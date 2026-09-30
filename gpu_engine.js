/**
 * gpu_engine.js — Gerçek WebGPU Secp256k1 Point Multiplication Engine
 *
 * GERÇEK implementasyon — hiçbir sahte kod yoktur.
 * WGSL 1.0 uyumlu: Sadece u32, bool, vec2<u32>. (64-bit tip YOK)
 *
 * Mimari:
 *   GPU: privkey (256-bit) → compressed pubkey (33 byte)  [WGSL compute shader]
 *   CPU: pubkey hex → SHA256 → RIPEMD160  [CryptoJS, zaten window'da]
 *
 * navigator.gpu yoksa → init() false döner. Çağıran kod CPU'ya fallback yapar.
 *
 * Konsolda doğrulama:
 *   await initWebGpuEngine()   // true = gerçek GPU
 *   startGpuHunting()
 *   setTimeout(()=>console.log(gpuKeysPerSec+'key/s'), 5000)
 */

// =============================================================================
// WGSL SHADER — secp256k1 nokta çarpımı, WGSL 1.0 (u32 only)
// U256 = 8 x u32, little-endian (limbs[0] = en küçük anlamlı word)
// =============================================================================
const GPU_WGSL = `
// ----------------------------------------------------------------------------
// Veri yapıları
// ----------------------------------------------------------------------------
struct U256 { l: array<u32, 8> }
struct U512 { lo: U256, hi: U256 }
struct JacobianPoint { x: U256, y: U256, z: U256, is_inf: u32 }

// Input: 8 u32 per key (LE), Output: 9 u32 per key (compressed pubkey)
@group(0) @binding(0) var<storage, read>       privkeys: array<u32>;
@group(0) @binding(1) var<storage, read_write> pubkeys:  array<u32>;

// ----------------------------------------------------------------------------
// u32 x u32 → 64-bit result represented as vec2<u32>(lo, hi)
// WGSL 1.0 uyumlu: 16-bit parçalara böleriz (sadece u32)
// ----------------------------------------------------------------------------
fn mul32(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;

    let ll = a_lo * b_lo;
    let lh = a_lo * b_hi;
    let hl = a_hi * b_lo;
    let hh = a_hi * b_hi;

    // cross terms mid
    let mid = (ll >> 16u) + (lh & 0xFFFFu) + (hl & 0xFFFFu);
    let lo  = (ll & 0xFFFFu) | (mid << 16u);
    let hi  = hh + (lh >> 16u) + (hl >> 16u) + (mid >> 16u);
    return vec2<u32>(lo, hi);
}

// a32 + b32 + c32 → vec2<u32>(sum_lo, carry)  (all u32, no overflow)
fn add32c(a: u32, b: u32, c: u32) -> vec2<u32> {
    let s1 = a + b;
    let c1 = select(0u, 1u, s1 < a);
    let s2 = s1 + c;
    let c2 = select(0u, 1u, s2 < s1);
    return vec2<u32>(s2, c1 + c2);
}

// ----------------------------------------------------------------------------
// U256 helpers
// ----------------------------------------------------------------------------
fn u256_zero() -> U256 {
    var r: U256;
    for (var i=0u; i<8u; i++) { r.l[i]=0u; }
    return r;
}
fn u256_one() -> U256 {
    var r: U256;
    r.l[0]=1u;
    for (var i=1u; i<8u; i++) { r.l[i]=0u; }
    return r;
}
fn u256_is_zero(a: U256) -> bool {
    for (var i=0u; i<8u; i++) { if(a.l[i]!=0u){return false;} }
    return true;
}
// Compare: true if a < b
fn u256_lt(a: U256, b: U256) -> bool {
    for (var i=7u; i>=1u; i--) {
        if(a.l[i] < b.l[i]) { return true; }
        if(a.l[i] > b.l[i]) { return false; }
    }
    return a.l[0] < b.l[0];
}
fn u256_eq(a: U256, b: U256) -> bool {
    for (var i=0u; i<8u; i++) { if(a.l[i]!=b.l[i]){return false;} }
    return true;
}

// Raw add (no mod) — returns carry in extra field (ignored, overflow handled by caller)
fn u256_add_raw(a: U256, b: U256) -> U256 {
    var r: U256;
    var carry: u32 = 0u;
    for (var i=0u; i<8u; i++) {
        let s1 = a.l[i] + b.l[i];
        let c1 = select(0u, 1u, s1 < a.l[i]);
        let s2 = s1 + carry;
        let c2 = select(0u, 1u, s2 < s1);
        r.l[i] = s2;
        carry = c1 + c2;
    }
    return r;
}

// Raw sub (assumes a >= b)
fn u256_sub_raw(a: U256, b: U256) -> U256 {
    var r: U256;
    var borrow: u32 = 0u;
    for (var i=0u; i<8u; i++) {
        let s = a.l[i] - b.l[i] - borrow;
        // borrow if a.l[i] < b.l[i]+borrow
        let nb_cond1 = a.l[i] < b.l[i];
        let nb_cond2 = (borrow == 1u) && (a.l[i] == b.l[i]);
        borrow = select(0u, 1u, nb_cond1 || nb_cond2);
        r.l[i] = s;
    }
    return r;
}

// secp256k1 prime p = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF FFFFFFFEFFFFFC2F
fn p_val() -> U256 {
    var p: U256;
    p.l[0]=0xFFFFFC2Fu; p.l[1]=0xFFFFFFFEu;
    p.l[2]=0xFFFFFFFFu; p.l[3]=0xFFFFFFFFu;
    p.l[4]=0xFFFFFFFFu; p.l[5]=0xFFFFFFFFu;
    p.l[6]=0xFFFFFFFFu; p.l[7]=0xFFFFFFFFu;
    return p;
}

// ----------------------------------------------------------------------------
// Field arithmetic mod p
// ----------------------------------------------------------------------------
fn fp_add(a: U256, b: U256) -> U256 {
    let p = p_val();
    var r: U256;
    var carry: u32 = 0u;
    for (var i = 0u; i < 8u; i++) {
        let s1 = a.l[i] + b.l[i];
        let c1 = select(0u, 1u, s1 < a.l[i]);
        let s2 = s1 + carry;
        let c2 = select(0u, 1u, s2 < s1);
        r.l[i] = s2;
        carry = c1 + c2;
    }
    if (carry != 0u) {
        // a+b >= 2^256; add (2^256 - p) = {977, 1, 0, 0, 0, 0, 0, 0}
        let s0 = add32c(r.l[0], 977u, 0u); r.l[0] = s0.x;
        let s1 = add32c(r.l[1], 1u, s0.y); r.l[1] = s1.x;
        var cc = s1.y;
        for (var i = 2u; i < 8u; i++) {
            if (cc == 0u) { break; }
            let s = r.l[i] + cc;
            cc = select(0u, 1u, s < r.l[i]);
            r.l[i] = s;
        }
    } else if (!u256_lt(r, p)) {
        r = u256_sub_raw(r, p);
    }
    return r;
}
fn fp_sub(a: U256, b: U256) -> U256 {
    if (u256_lt(a, b)) { return u256_sub_raw(p_val(), u256_sub_raw(b, a)); }
    return u256_sub_raw(a, b);
}

// 256x256 → 512-bit (schoolbook, all u32 via mul32)
fn u256_mul_full(a: U256, b: U256) -> U512 {
    // result[k] for k in 0..15 (use two U256)
    var t: array<u32, 16>;
    for (var k=0u; k<16u; k++) { t[k]=0u; }

    for (var i=0u; i<8u; i++) {
        for (var j=0u; j<8u; j++) {
            let k = i+j;
            let prod = mul32(a.l[i], b.l[j]);
            // prod.x → t[k], prod.y → t[k+1], with carry propagation
            let r1 = add32c(t[k], prod.x, 0u);
            t[k] = r1.x;
            let r2 = add32c(t[k+1u], prod.y, r1.y);
            t[k+1u] = r2.x;
            var carry: u32 = r2.y;
            var kk = k + 2u;
            loop {
                if(carry == 0u || kk >= 16u) { break; }
                let ss = t[kk] + carry;
                carry = select(0u, 1u, ss < t[kk]);
                t[kk] = ss;
                kk++;
            }
        }
    }

    var lo: U256; var hi: U256;
    for (var i=0u; i<8u; i++) { lo.l[i]=t[i]; hi.l[i]=t[i+8u]; }
    return U512(lo, hi);
}

// Reduce 512-bit mod secp256k1 p using: 2^256 ≡ 2^32 + 977 (mod p)
fn u512_mod_p(t: U512) -> U256 {
    let h = t.hi;
    let p = p_val();

    // 10-limb accumulator: acc = t.lo (8 limbs) + h*(2^32+977) (can overflow into limbs 8,9)
    var acc: array<u32, 10>;
    for (var i = 0u; i < 10u; i++) { acc[i] = 0u; }
    for (var i = 0u; i < 8u; i++) { acc[i] = t.lo.l[i]; }

    // Add h * 977
    var carry: u32 = 0u;
    for (var i = 0u; i < 8u; i++) {
        let prod = mul32(h.l[i], 977u);
        let s = add32c(acc[i], prod.x, carry);
        acc[i] = s.x;
        carry = prod.y + s.y;
    }
    let tmp8 = add32c(acc[8], carry, 0u); acc[8] = tmp8.x; acc[9] = tmp8.y;

    // Add h * 2^32 (shift h left one limb position)
    carry = 0u;
    for (var i = 0u; i < 8u; i++) {
        let s = add32c(acc[i + 1u], h.l[i], carry);
        acc[i + 1u] = s.x;
        carry = s.y;
    }
    acc[9] = acc[9] + carry;

    // Fold acc[8..9] back: 2^256 ≡ 2^32+977, so ov*2^256 ≡ ov*(2^32+977)
    let ov8 = acc[8]; let ov9 = acc[9];
    acc[8] = 0u; acc[9] = 0u;

    if (ov8 != 0u || ov9 != 0u) {
        // ov8 * 977
        let q8 = mul32(ov8, 977u);
        // ov8 * 2^32 → adds to position [1]
        // ov9 * 977 → adds to position [0..1]
        // ov9 * 2^32 → adds to position [1..2] (ov9 is usually 0 or 1)
        let r0 = add32c(acc[0], q8.x, 0u); acc[0] = r0.x;
        // q8.y = high word of ov8*977, ov8 = ov8*1 for the *2^32 term
        // q8.y + ov8 could overflow u32, so use add32c
        let qc = add32c(q8.y, ov8, 0u);
        let r1 = add32c(acc[1], qc.x, r0.y); acc[1] = r1.x;
        carry = qc.y + r1.y;
        for (var i = 2u; i < 8u; i++) {
            if (carry == 0u) { break; }
            let s = acc[i] + carry;
            carry = select(0u, 1u, s < acc[i]);
            acc[i] = s;
        }
        if (ov9 != 0u) {
            let q9 = mul32(ov9, 977u);
            let s0 = add32c(acc[0], q9.x, 0u); acc[0] = s0.x;
            let sc = add32c(q9.y, ov9, 0u);
            let s1 = add32c(acc[1], sc.x, s0.y); acc[1] = s1.x;
            carry = sc.y + s1.y;
            for (var i = 2u; i < 8u; i++) {
                if (carry == 0u) { break; }
                let s = acc[i] + carry;
                carry = select(0u, 1u, s < acc[i]);
                acc[i] = s;
            }
        }
    }

    var r: U256;
    for (var i = 0u; i < 8u; i++) { r.l[i] = acc[i]; }
    if (!u256_lt(r, p)) { r = u256_sub_raw(r, p); }
    if (!u256_lt(r, p)) { r = u256_sub_raw(r, p); }
    return r;
}

fn fp_mul(a: U256, b: U256) -> U256 { return u512_mod_p(u256_mul_full(a, b)); }
fn fp_sq(a: U256) -> U256 { return fp_mul(a, a); }

// Modular inverse: a^(p-2) mod p  [Fermat's little theorem]
// p-2 = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFBFFFFFC2D
fn fp_inv(a: U256) -> U256 {
    var exp: U256;
    exp.l[0]=0xFFFFFC2Du; exp.l[1]=0xFFFFFFFEu;
    exp.l[2]=0xFFFFFFFFu; exp.l[3]=0xFFFFFFFFu;
    exp.l[4]=0xFFFFFFFFu; exp.l[5]=0xFFFFFFFFu;
    exp.l[6]=0xFFFFFFFFu; exp.l[7]=0xFFFFFFFFu;
    var base = a;
    var result = u256_one();
    for (var i=0u; i<256u; i++) {
        let word = i / 32u;
        let bit  = i % 32u;
        if ((exp.l[word] >> bit) & 1u) != 0u {
            result = fp_mul(result, base);
        }
        base = fp_sq(base);
    }
    return result;
}

// ----------------------------------------------------------------------------
// Jacobian arithmetic
// ----------------------------------------------------------------------------
fn jac_double(pt: JacobianPoint) -> JacobianPoint {
    if (pt.is_inf != 0u) { return pt; }
    // Standard Jacobian doubling for a=0 (secp256k1)
    // S = 4*X*Y^2
    // M = 3*X^2
    // X' = M^2 - 2*S
    // Y' = M*(S - X') - 8*Y^4
    // Z' = 2*Y*Z
    let y2 = fp_sq(pt.y);
    let x2 = fp_sq(pt.x);
    // M = 3*X^2
    let m  = fp_add(x2, fp_add(x2, x2));
    // S = 4*X*Y^2
    let xy2 = fp_mul(pt.x, y2);
    let s   = fp_add(fp_add(xy2, xy2), fp_add(xy2, xy2));  // 4*X*Y^2
    // X' = M^2 - 2*S
    let m2  = fp_sq(m);
    let x3  = fp_sub(m2, fp_add(s, s));
    // Y' = M*(S-X') - 8*Y^4
    let y4  = fp_sq(y2);
    let y4x8 = fp_add(fp_add(fp_add(y4,y4),fp_add(y4,y4)),fp_add(fp_add(y4,y4),fp_add(y4,y4)));  // 8*Y^4
    let y3  = fp_sub(fp_mul(m, fp_sub(s, x3)), y4x8);
    // Z' = 2*Y*Z
    let z3  = fp_mul(fp_add(pt.y, pt.y), pt.z);
    return JacobianPoint(x3, y3, z3, 0u);
}

fn jac_add(p1: JacobianPoint, p2: JacobianPoint) -> JacobianPoint {
    if (p1.is_inf != 0u) { return p2; }
    if (p2.is_inf != 0u) { return p1; }

    let z1sq = fp_sq(p1.z);
    let z2sq = fp_sq(p2.z);
    let u1 = fp_mul(p1.x, z2sq);
    let u2 = fp_mul(p2.x, z1sq);
    let s1 = fp_mul(p1.y, fp_mul(p2.z, z2sq));
    let s2 = fp_mul(p2.y, fp_mul(p1.z, z1sq));
    let h  = fp_sub(u2, u1);
    let r  = fp_sub(s2, s1);

    if (u256_is_zero(h)) {
        if (u256_is_zero(r)) { return jac_double(p1); }
        var inf: JacobianPoint; inf.is_inf = 1u; return inf;
    }

    let h2   = fp_sq(h);
    let h3   = fp_mul(h, h2);
    let u1h2 = fp_mul(u1, h2);
    let r2   = fp_sq(r);
    var x3   = fp_sub(r2, h3);
    x3       = fp_sub(x3, fp_add(u1h2, u1h2));
    let y3   = fp_sub(fp_mul(r, fp_sub(u1h2, x3)), fp_mul(s1, h3));
    let z3   = fp_mul(h, fp_mul(p1.z, p2.z));
    return JacobianPoint(x3, y3, z3, 0u);
}

// secp256k1 generator G
fn G_point() -> JacobianPoint {
    var Gx: U256;
    Gx.l[0]=0x16F81798u; Gx.l[1]=0x59F2815Bu; Gx.l[2]=0x2DCE28D9u; Gx.l[3]=0x029BFCDBu;
    Gx.l[4]=0xCE870B07u; Gx.l[5]=0x55A06295u; Gx.l[6]=0xF9DCBBACu; Gx.l[7]=0x79BE667Eu;
    var Gy: U256;
    Gy.l[0]=0xFB10D4B8u; Gy.l[1]=0x9C47D08Fu; Gy.l[2]=0xA6855419u; Gy.l[3]=0xFD17B448u;
    Gy.l[4]=0x0E1108A8u; Gy.l[5]=0x5DA4FBFCu; Gy.l[6]=0x26A3C465u; Gy.l[7]=0x483ADA77u;
    return JacobianPoint(Gx, Gy, u256_one(), 0u);
}

// Find highest non-zero bit of k (MSB)
fn u256_top_bit(k: U256) -> u32 {
    for (var i = 0u; i < 8u; i++) {
        let w = 7u - i;
        let val = k.l[w];
        if (val != 0u) {
            for (var b = 31u; b > 0u; b--) {
                if ((val & (1u << b)) != 0u) {
                    return w * 32u + b;
                }
            }
            return w * 32u;
        }
    }
    return 0u;
}

// k * G  (double-and-add, MSB first — puzzle bit sayısına göre 256 yerine sadece gerekli adım kadar çalışır)
fn scalar_mul_G(k: U256) -> JacobianPoint {
    if (u256_is_zero(k)) {
        var inf: JacobianPoint; inf.is_inf = 1u; return inf;
    }
    let top_bit = u256_top_bit(k);
    var result = G_point();
    let G = G_point();
    var bit_idx = top_bit;
    loop {
        if (bit_idx == 0u) { break; }
        bit_idx--;
        let word = bit_idx / 32u;
        let bit  = bit_idx % 32u;
        result = jac_double(result);
        if (((k.l[word] >> bit) & 1u) != 0u) {
            result = jac_add(result, G);
        }
    }
    return result;
}

// Jacobian → affine (inline, main içinde kullanılır — WGSL 1.0'da generic return yoktur)

// Store 33-byte compressed + 65-byte uncompressed pubkey at output slot [base]
// pubkeys buffer: 26 u32 per key = 104 bytes (9 words compressed, 17 words uncompressed)
fn store_pubkey(base: u32, x: U256, y: U256) {
    let prefix = 2u + (y.l[0] & 1u);  // 02=even, 03=odd
    let off = base * 26u;

    // 1. Sıkıştırılmış (Compressed) 33-byte: [prefix, X (32 bytes)] -> 9 u32 kelimesi
    var c_bytes: array<u32, 33>;
    c_bytes[0] = prefix;
    for (var li=0u; li<8u; li++) {
        let limb = x.l[7u - li];  // MSB limb first
        c_bytes[1u + li*4u + 0u] = (limb >> 24u) & 0xFFu;
        c_bytes[1u + li*4u + 1u] = (limb >> 16u) & 0xFFu;
        c_bytes[1u + li*4u + 2u] = (limb >>  8u) & 0xFFu;
        c_bytes[1u + li*4u + 3u] =  limb         & 0xFFu;
    }
    for (var w=0u; w<9u; w++) {
        let b0 = select(0u, c_bytes[w*4u+0u], w*4u+0u < 33u);
        let b1 = select(0u, c_bytes[w*4u+1u], w*4u+1u < 33u);
        let b2 = select(0u, c_bytes[w*4u+2u], w*4u+2u < 33u);
        let b3 = select(0u, c_bytes[w*4u+3u], w*4u+3u < 33u);
        pubkeys[off + w] = (b0 << 24u) | (b1 << 16u) | (b2 << 8u) | b3;
    }

    // 2. Sıkıştırılmamış (Uncompressed) 65-byte: [0x04, X (32 bytes), Y (32 bytes)] -> 17 u32 kelimesi
    var u_bytes: array<u32, 65>;
    u_bytes[0] = 4u; // 0x04 uncompressed
    for (var li=0u; li<8u; li++) {
        let lx = x.l[7u - li];
        u_bytes[1u + li*4u + 0u] = (lx >> 24u) & 0xFFu;
        u_bytes[1u + li*4u + 1u] = (lx >> 16u) & 0xFFu;
        u_bytes[1u + li*4u + 2u] = (lx >>  8u) & 0xFFu;
        u_bytes[1u + li*4u + 3u] =  lx         & 0xFFu;
        let ly = y.l[7u - li];
        u_bytes[33u + li*4u + 0u] = (ly >> 24u) & 0xFFu;
        u_bytes[33u + li*4u + 1u] = (ly >> 16u) & 0xFFu;
        u_bytes[33u + li*4u + 2u] = (ly >>  8u) & 0xFFu;
        u_bytes[33u + li*4u + 3u] =  ly         & 0xFFu;
    }
    for (var w=0u; w<17u; w++) {
        let b0 = select(0u, u_bytes[w*4u+0u], w*4u+0u < 65u);
        let b1 = select(0u, u_bytes[w*4u+1u], w*4u+1u < 65u);
        let b2 = select(0u, u_bytes[w*4u+2u], w*4u+2u < 65u);
        let b3 = select(0u, u_bytes[w*4u+3u], w*4u+3u < 65u);
        pubkeys[off + 9u + w] = (b0 << 24u) | (b1 << 16u) | (b2 << 8u) | b3;
    }
}

// ----------------------------------------------------------------------------
// Compute entry point
// ----------------------------------------------------------------------------
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx   = gid.x;
    let total = arrayLength(&privkeys) / 8u;
    if (idx >= total) { return; }

    // Load private key (8 u32, LE)
    var k: U256;
    for (var i=0u; i<8u; i++) { k.l[i] = privkeys[idx * 8u + i]; }
    if (u256_is_zero(k)) { return; }

    // k * G
    let pt = scalar_mul_G(k);
    if (pt.is_inf != 0u) { return; }

    // Jacobian → affine
    let zinv  = fp_inv(pt.z);
    let zinv2 = fp_sq(zinv);
    let zinv3 = fp_mul(zinv, zinv2);
    let ax    = fp_mul(pt.x, zinv2);
    let ay    = fp_mul(pt.y, zinv3);

    store_pubkey(idx, ax, ay);
}
`;

// =============================================================================
// JavaScript WebGPU Engine
// =============================================================================

class GpuEngine {
    constructor() {
        this.device      = null;
        this.pipeline    = null;
        this.privkeyBuf  = null;
        this.pubkeyBuf   = null;
        this.readbackBuf = null;
        this.bindGroup   = null;
        this.maxBatchSize = 2048;
        this.batchSize   = (typeof gpuCurrentIntensity !== 'undefined' && gpuCurrentIntensity === 'ECO') ? 64
                         : ((typeof gpuCurrentIntensity !== 'undefined' && gpuCurrentIntensity === 'MAX') ? 1024 : 256);
        this.isInitialized = false;
        this.adapterInfo = null;
    }

    async init() {
        if (!navigator.gpu) {
            console.log('[GPU] navigator.gpu yok — WebGPU desteklenmiyor');
            return false;
        }
        try {
            let adapter = null;
            try {
                adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
            } catch(e) {}
            if (!adapter) {
                try {
                    adapter = await navigator.gpu.requestAdapter();
                } catch(e) {}
            }
            if (!adapter) {
                console.log('[GPU] Adapter bulunamadı');
                return false;
            }
            this.adapterInfo = (adapter.info && adapter.info.description) ? adapter.info.description
                             : (adapter.name || 'WebGPU GPU');

            let device = null;
            try {
                device = await adapter.requestDevice();
            } catch(e) {
                console.warn('[GPU] Standart requestDevice başarısız, temel limitler ile deneniyor:', e);
                device = await adapter.requestDevice({ requiredLimits: {} });
            }
            this.device = device;

            this.device.lost.then(info => {
                console.warn('[GPU] Device lost:', info.reason, info.message);
                this.isInitialized = false;
            });

            // Shader derleme
            const shaderModule = this.device.createShaderModule({ code: GPU_WGSL });
            const compileInfo  = await shaderModule.getCompilationInfo();
            const errors = compileInfo.messages.filter(m => m.type === 'error');
            if (errors.length > 0) {
                console.error('[GPU] WGSL derleme HATA:\n' + errors.map(e => `  Line ${e.lineNum}: ${e.message}`).join('\n'));
                return false;
            }
            if (compileInfo.messages.length > 0) {
                compileInfo.messages.forEach(m => console.warn('[GPU][WGSL]', m.type, 'line', m.lineNum, ':', m.message));
            }

            // Asenkron pipeline oluşturma (tarayıcıyı ve UI'ı dondurmaz)
            if (typeof this.device.createComputePipelineAsync === 'function') {
                this.pipeline = await this.device.createComputePipelineAsync({
                    layout: 'auto',
                    compute: { module: shaderModule, entryPoint: 'main' }
                });
            } else {
                this.pipeline = this.device.createComputePipeline({
                    layout: 'auto',
                    compute: { module: shaderModule, entryPoint: 'main' }
                });
            }

            const privBytes = this.maxBatchSize * 8 * 4;   // 2048*8*4 = 65,536 bytes
            const pubBytes  = this.maxBatchSize * 26 * 4;  // 2048*26*4 = 212,992 bytes (33b comp + 65b uncomp)

            this.privkeyBuf = this.device.createBuffer({
                size: privBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            });
            this.pubkeyBuf = this.device.createBuffer({
                size: pubBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
            });
            this.readbackBuf = this.device.createBuffer({
                size: pubBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });
            this.bindGroup = this.device.createBindGroup({
                layout: this.pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.privkeyBuf } },
                    { binding: 1, resource: { buffer: this.pubkeyBuf  } }
                ]
            });

            this.isInitialized = true;

            // Isınma (Warmup) adımı: GPU driver JIT derlemesini ilk aramadan önce tamamlar
            try {
                await this.computePubkeysRaw([1n]);
            } catch(we) {
                console.warn('[GPU] Warmup uyarısı:', we);
            }

            console.log('[GPU] ✅ WebGPU yüksek performansla başlatıldı! Adapter:', this.adapterInfo);
            console.log('[GPU] Batch boyutu:', this.batchSize, 'key/dispatch (' + Math.ceil(this.batchSize / 64) + ' workgroups)');
            return true;
        } catch(e) {
            console.error('[GPU] Init hatası:', e);
            return false;
        }
    }

    /**
     * Ham Uint32Array olarak pubkey kelimelerini döner (sıfır ara nesne tahsisi)
     */
    async computePubkeysRaw(privkeysBig) {
        if (!this.isInitialized) throw new Error('GPU başlatılmamış');
        const n = privkeysBig.length;

        // BigInt → Uint32Array (LE, 8 u32 per key)
        const privData = new Uint32Array(n * 8);
        for (let i = 0; i < n; i++) {
            let k = privkeysBig[i];
            const base = i * 8;
            for (let j = 0; j < 8; j++) {
                privData[base + j] = Number(k & 0xFFFFFFFFn);
                k >>= 32n;
            }
        }

        this.device.queue.writeBuffer(this.privkeyBuf, 0, privData.buffer, 0, n * 32);

        const encoder = this.device.createCommandEncoder();
        encoder.clearBuffer(this.pubkeyBuf, 0, n * 104);

        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.dispatchWorkgroups(Math.ceil(n / 64));
        pass.end();

        encoder.copyBufferToBuffer(this.pubkeyBuf, 0, this.readbackBuf, 0, n * 104);
        this.device.queue.submit([encoder.finish()]);

        await this.readbackBuf.mapAsync(GPUMapMode.READ, 0, n * 104);
        const raw = new Uint32Array(this.readbackBuf.getMappedRange(0, n * 104).slice(0));
        this.readbackBuf.unmap();
        return raw;
    }

    /**
     * Batch private key'leri GPU ile işle → 33-byte compressed pubkey dizisi
     * (testGpu ve konsol uyumluluğu için)
     * @param {BigInt[]} privkeysBig
     * @returns {Uint8Array[]}
     */
    async computePubkeys(privkeysBig) {
        const raw = await this.computePubkeysRaw(privkeysBig);
        const n = privkeysBig.length;
        const pubkeys = [];
        for (let i = 0; i < n; i++) {
            const bytes = new Uint8Array(33);
            const base = i * 26;
            for (let w = 0; w < 9; w++) {
                const word = raw[base + w];
                for (let b = 0; b < 4; b++) {
                    const bi = w * 4 + b;
                    if (bi < 33) bytes[bi] = (word >> (24 - b * 8)) & 0xFF;
                }
            }
            pubkeys.push(bytes);
        }
        return pubkeys;
    }

    destroy() {
        try {
            if (this.privkeyBuf)  this.privkeyBuf.destroy();
            if (this.pubkeyBuf)   this.pubkeyBuf.destroy();
            if (this.readbackBuf) this.readbackBuf.destroy();
            if (this.device)      this.device.destroy();
        } catch(e) {}
        this.isInitialized = false;
        this.device = null;
    }
}

// =============================================================================
// GPU Hunt Loop
// =============================================================================

let gpuEngine     = null;
let gpuHuntRunning = false;
let gpuKeysPerSec  = 0;
let gpuTotalKeys   = 0;

async function initWebGpuEngine() {
    gpuEngine = new GpuEngine();
    const ok = await gpuEngine.init();
    if (!ok) { gpuEngine = null; return false; }
    return true;
}

function pubkeyBytesToHex(bytes) {
    let hex = '';
    for (let i = 0; i < bytes.length; i++) hex += bytes[i].toString(16).padStart(2, '0');
    return hex;
}

/** Hash160 (SHA256 + RIPEMD160) from hex pubkey — uses window.CryptoJS */
function pubhexToHash160(pubHex) {
    const cjs = window.CryptoJS;
    if (!cjs) return '';
    const wa = cjs.enc.Hex.parse(pubHex);
    return cjs.RIPEMD160(cjs.SHA256(wa)).toString();
}

let gpuAutoStepCounter = 0;
let gpuAutoIndex = 0;
let gpuPrefixJumpCounter = 0;
let gpuActivePrefix = '';

function toBigIntSafeGpu(val) {
    if (val === null || typeof val === 'undefined') return 0n;
    if (typeof val === 'bigint') return val;
    let s = val.toString().trim().toLowerCase();
    if (s.startsWith('0x')) return BigInt(s);
    return BigInt('0x' + s);
}

/**
 * GPU Arama Uzayını ve Algoritmaları Yöneten Gelişmiş Üretici
 * (Kaydıraç / Slider, Ön Ek / Prefix ve Seçilen Algoritmayı %100 Çalıştırır)
 */
function genBatchForGpu(cfg, size) {
    const target = cfg.activeTarget;
    const baseStart = toBigIntSafeGpu(target.origStart || target.start || '1');
    const baseEnd   = toBigIntSafeGpu(target.origEnd   || target.end   || 'fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364140');
    const totalR    = baseEnd - baseStart;

    // 1. Kaydıraç (Dual Slider) & Hex Range
    let boundMin = baseStart;
    let boundMax = baseEnd;

    if (cfg.isDualSliderActive && cfg.activeRangeMode !== 'HEX') {
        if (cfg.isCircularMode) {
            const minStep = (cfg.sliderMinStep !== undefined && cfg.sliderMinStep !== null) ? BigInt(Math.round(cfg.sliderMinStep)) : 0n;
            const maxStep = (cfg.sliderMaxStep !== undefined && cfg.sliderMaxStep !== null) ? BigInt(Math.round(cfg.sliderMaxStep)) : 1000000000000n;
            if (minStep > 0n || maxStep < 1000000000000n) {
                boundMin = baseStart + (totalR * minStep / 1000000000000n);
                boundMax = baseStart + (totalR * maxStep / 1000000000000n);
                if (boundMax <= boundMin) {
                    boundMax = (boundMin + (totalR / 100n) <= baseEnd) ? (boundMin + (totalR / 100n)) : baseEnd;
                    if (boundMin >= boundMax) { boundMin = baseStart; boundMax = baseEnd; }
                }
            }
        } else if (cfg.sliderMinHex && cfg.sliderMaxHex) {
            const sMin = toBigIntSafeGpu(cfg.sliderMinHex);
            const sMax = toBigIntSafeGpu(cfg.sliderMaxHex);
            if (sMin >= baseStart && sMin < baseEnd &&
                sMax > baseStart && sMax <= baseEnd &&
                sMax > sMin) {
                boundMin = sMin;
                boundMax = sMax;
            }
        }
    }

    if (cfg.activeRangeMode === 'HEX' && cfg.userHexStart && cfg.userHexEnd) {
        const hMin = toBigIntSafeGpu(cfg.userHexStart);
        const hMax = toBigIntSafeGpu(cfg.userHexEnd);
        boundMin = (hMin < boundMin) ? boundMin : (hMin > boundMax ? boundMin : hMin);
        boundMax = (hMax > boundMax) ? boundMax : (hMax < boundMin ? boundMax : hMax);
    } else if (cfg.activeRangeMode === 'PERCENT' && cfg.sliceModeActive && totalR > 10000n) {
        const effRange = boundMax - boundMin;
        const sBig = BigInt(Math.round((cfg.sliceStartPct || 0) * 100000000));
        const eBig = BigInt(Math.round((cfg.sliceEndPct || 100) * 100000000));
        boundMin = boundMin + (effRange * sBig / 10000000000n);
        boundMax = boundMin + (effRange * eBig / 10000000000n);
    }
    if (boundMax <= boundMin) boundMax = boundMin + 1n;

    // 2. Fizik ve Kaos Formülleri Alt-Aralık Hesabı ([subMin, subMax])
    let subMin = boundMin;
    let subMax = boundMax;
    const f = cfg.activeFormula;
    const curSpan = (boundMax > boundMin) ? (boundMax - boundMin) : 1n;
    const isNarrowRange = curSpan <= 10000000n; // 10M altı test ve dar aralıklarda mikro dilimleme YAPILMAZ!
    const isPrefixMode  = (f === 'PREFIX' || cfg.activeRangeMode === 'PREFIX' || cfg.isPrefixFilterActive);
    const isHexRange    = (cfg.activeRangeMode === 'HEX');

    if (!isNarrowRange && !isPrefixMode && !isHexRange &&
        (f === 'OMNI_CHAOS' || f === 'DETERMINISTIC_CHAOS' || f === 'QUANTUM_TUNNEL' || f === 'RIEMANN_ZETA' || f === 'GOLDEN_SINGULARITY' || cfg.activeRangeMode === 'PHYSICS')) {
        if (typeof window.getNextPhysicsPercent === 'function') {
            const centerPct = window.getNextPhysicsPercent();
            const microWidth = 0.5; // Geniş uzayda anlamlı pencere genişliği
            const sPct = Math.max(0.0, centerPct - microWidth);
            const ePct = Math.min(100.0, centerPct + microWidth);
            const startBig = BigInt(Math.round(sPct * 100000000));
            const endBig   = BigInt(Math.round(ePct * 100000000));
            subMin = boundMin + (curSpan * startBig / 10000000000n);
            subMax = boundMin + (curSpan * endBig   / 10000000000n);
            if (subMax <= subMin) subMax = subMin + 1n;
        }
    }

    // 3. Algoritma Seçimi
    let activeAlgo = cfg.selectedAlgorithm || 'RANDOM';
    if (activeAlgo === 'AUTO') {
        gpuAutoStepCounter++;
        const autoList = ['RANDOM', 'SOBOL', 'VD_CORPUT', 'WEYL_GOLDEN', 'COPRIME_STRIDE', 'HILBERT', 'WEAK_ENTROPY', 'CHAOS', 'KANGAROO'];
        if (gpuAutoStepCounter % 25 === 0) {
            gpuAutoIndex = (gpuAutoIndex + 1) % autoList.length;
        }
        activeAlgo = autoList[gpuAutoIndex] || 'RANDOM';
    }
    if (activeAlgo === 'WEAK_ENTROPY' && cfg.selectedEntropyBits === -1) {
        activeAlgo = 'RANDOM';
    }
    if (activeAlgo === 'KANGAROO' && typeof window.isKangarooModeActive === 'function' && !window.isKangarooModeActive()) {
        activeAlgo = 'RANDOM';
    }

    // 4. Ön Ek Hazırlığı (Prefix)
    const isEffectivePrefix = (cfg.isPrefixFilterActive !== false && (Boolean(cfg.userPrefix) || cfg.isPrefixAutoJump));
    if (isEffectivePrefix) {
        if (cfg.isPrefixAutoJump) {
            gpuPrefixJumpCounter++;
            if (gpuPrefixJumpCounter % 25 === 0 || !gpuActivePrefix) {
                const totalLen = baseEnd.toString(16).length;
                const sHexPadded = boundMin.toString(16).padStart(totalLen, '0');
                const eHexPadded = boundMax.toString(16).padStart(totalLen, '0');
                let commonLen = 0;
                while (commonLen < totalLen && sHexPadded[commonLen] === eHexPadded[commonLen]) commonLen++;
                const pLen = commonLen > 0 ? Math.min(totalLen - 1, commonLen + 1) : Math.max(1, Math.min(4, Math.floor(totalLen / 4)));
                const sPrefixInt = parseInt(sHexPadded.substring(0, pLen), 16) || 0;
                const ePrefixInt = parseInt(eHexPadded.substring(0, pLen), 16) || 15;
                const span = Math.max(1, ePrefixInt - sPrefixInt + 1);
                gpuActivePrefix = (sPrefixInt + Math.floor(Math.random() * span)).toString(16).padStart(pLen, '0');
            }
        } else {
            gpuActivePrefix = (cfg.userPrefix || '').trim().replace(/^0x/i, '');
        }
    }

    // 5. Anahtarları Üret (Seçilen Algoritmaya Göre)
    const keys = new Array(size);
    const effSpan = subMax - subMin;
    const targetLen = baseEnd.toString(16).length;

    let rndBuf = null;
    let rndIdx = 0;

    for (let i = 0; i < size; i++) {
        let currentKey;

        if (activeAlgo === 'SOBOL' && typeof window.generateSobolKey === 'function') {
            currentKey = window.generateSobolKey(subMin, subMax);
        } else if (activeAlgo === 'VD_CORPUT' && typeof window.generateVanDerCorputKey === 'function') {
            currentKey = window.generateVanDerCorputKey(subMin, subMax);
        } else if (activeAlgo === 'WEYL_GOLDEN' && typeof window.generateWeylGoldenKey === 'function') {
            currentKey = window.generateWeylGoldenKey(subMin, subMax);
        } else if (activeAlgo === 'COPRIME_STRIDE' && typeof window.generateCoprimeStrideKey === 'function') {
            currentKey = window.generateCoprimeStrideKey(subMin, subMax);
        } else if (activeAlgo === 'HILBERT' && typeof window.generateHilbertKey === 'function') {
            currentKey = window.generateHilbertKey(subMin, subMax);
        } else if (activeAlgo === 'CHAOS' && typeof window.generateDeterministicChaosKey === 'function') {
            currentKey = window.generateDeterministicChaosKey(subMin, subMax);
        } else if (activeAlgo === 'WEAK_ENTROPY' && typeof window.generateWeakEntropyKey === 'function' && cfg.selectedEntropyBits !== -1) {
            currentKey = window.generateWeakEntropyKey(subMin, subMax, cfg.selectedEntropyBits);
        } else if (activeAlgo === 'KANGAROO' && typeof window.generateKangarooKey === 'function') {
            currentKey = window.generateKangarooKey(subMin, subMax);
        } else if (typeof window.randomBigIntInRange === 'function') {
            currentKey = window.randomBigIntInRange(subMin, subMax);
        } else {
            if (!rndBuf || rndIdx >= rndBuf.length) {
                rndBuf = new Uint8Array(256 * 32);
                crypto.getRandomValues(rndBuf);
                rndIdx = 0;
            }
            let rand = 0n;
            for (let j = 0; j < 32; j++) rand = (rand << 8n) | BigInt(rndBuf[rndIdx + j]);
            rndIdx += 32;
            currentKey = subMin + (rand % (effSpan + 1n));
        }

        // Ön Ek Kilitleme
        if (isEffectivePrefix && gpuActivePrefix) {
            const keyHex = currentKey.toString(16).padStart(targetLen, '0');
            const pLen = Math.min(gpuActivePrefix.length, targetLen - 1);
            const prefixedHex = gpuActivePrefix.substring(0, pLen) + keyHex.substring(pLen);
            const pKey = BigInt('0x' + prefixedHex);
            if (pKey >= boundMin && pKey <= boundMax) {
                currentKey = pKey;
            }
        }

        // Kaydıraç Sınır Denetimi & Sarma (Clamp & Wrap)
        if (cfg.isDualSliderActive && (boundMin > baseStart || boundMax < baseEnd)) {
            if (currentKey < boundMin || currentKey > boundMax) {
                currentKey = effSpan > 0n ? (boundMin + ((currentKey > boundMin ? (currentKey - boundMin) : (boundMin - currentKey)) % effSpan)) : boundMin;
            }
        }

        // Sezgisel Ajan Popcount Sınırlaması (varsa)
        if (cfg && cfg.agentPopcountRange && typeof window.applyPopcountConstraint === 'function') {
            currentKey = window.applyPopcountConstraint(currentKey, boundMin, boundMax, cfg.agentPopcountRange);
        }

        keys[i] = currentKey;
    }

    return {
        keys: keys,
        boundMin: boundMin,
        boundMax: boundMax,
        subMin: subMin,
        subMax: subMax,
        activeAlgo: activeAlgo,
        activePrefix: isEffectivePrefix ? gpuActivePrefix : '',
        baseStart: baseStart,
        baseEnd: baseEnd
    };
}

// Hedef hash160'ı 5 tamsayı kelimesi olarak önbellekleme (string ayrıştırma yükünü sıfırlar)
let cachedTargetH160Str = '';
let targetW0 = 0, targetW1 = 0, targetW2 = 0, targetW3 = 0, targetW4 = 0;
function ensureTargetWords(h160) {
    if (h160 === cachedTargetH160Str) return;
    cachedTargetH160Str = h160;
    if (h160 && h160.length === 40) {
        targetW0 = parseInt(h160.substr(0, 8), 16) | 0;
        targetW1 = parseInt(h160.substr(8, 8), 16) | 0;
        targetW2 = parseInt(h160.substr(16, 8), 16) | 0;
        targetW3 = parseInt(h160.substr(24, 8), 16) | 0;
        targetW4 = parseInt(h160.substr(32, 8), 16) | 0;
    } else {
        targetW0 = targetW1 = targetW2 = targetW3 = targetW4 = 0;
    }
}

async function gpuHuntBatch() {
    if (!gpuHuntRunning || !gpuEngine || !gpuEngine.isInitialized) return;

    const cfg = (typeof getCurrentWorkerConfig === 'function') ? getCurrentWorkerConfig() : null;
    if (!cfg || !cfg.activeTarget) {
        if (!window._gpuCfgRetry) window._gpuCfgRetry = 0;
        window._gpuCfgRetry++;
        if (window._gpuCfgRetry > 50) {
            console.warn('[GPU] ⚠️ Hedef konfigürasyon alınamadı, GPU arama durduruldu.');
            gpuHuntRunning = false;
            window._gpuCfgRetry = 0;
            return;
        }
        setTimeout(gpuHuntBatch, 100);
        return;
    }
    window._gpuCfgRetry = 0;

    const target     = cfg.activeTarget;
    const targetH160 = target.targetHash160 || '';
    const BATCH      = gpuEngine.batchSize;
    const t0         = performance.now();

    try {
        const batchInfo = genBatchForGpu(cfg, BATCH);
        const keys = batchInfo.keys;
        const raw  = await gpuEngine.computePubkeysRaw(keys);

        ensureTargetWords(targetH160);
        const cjs = window.CryptoJS;
        if (!cjs) return;

        // Tek bir WordArray nesnesini yeniden kullanarak bellek çöpünü önleme
        const waWords = [0, 0, 0, 0, 0, 0, 0, 0, 0];
        const wa = { words: waWords, sigBytes: 33 };

        const waUWords = new Array(17);
        const waU = { words: waUWords, sigBytes: 65 };

        const isUncompTarget = Boolean(target && (target.isUncompressed || target.id === 'ilave' || target.id === 'TEST_64'));
        const checkUncomp = isUncompTarget || Boolean(cfg.isCustomPoolActive);

        let found = false;
        for (let i = 0; i < BATCH && gpuHuntRunning; i++) {
            const off = i * 26;
            const w0 = raw[off];
            const prefix = (w0 >>> 24) & 0xFF;
            if (prefix !== 2 && prefix !== 3) continue;

            waWords[0] = w0;
            waWords[1] = raw[off + 1];
            waWords[2] = raw[off + 2];
            waWords[3] = raw[off + 3];
            waWords[4] = raw[off + 4];
            waWords[5] = raw[off + 5];
            waWords[6] = raw[off + 6];
            waWords[7] = raw[off + 7];
            waWords[8] = raw[off + 8] & 0xFF000000;

            const sha = cjs.SHA256(wa);
            const rmd = cjs.RIPEMD160(sha);
            const rw = rmd.words;

            gpuTotalKeys++;

            // 🎯 1. Sıkıştırılmış (Compressed) Hedef Eşleşme Kontrolü
            if (targetW0 !== 0 && rw[0] === targetW0 && rw[1] === targetW1 && rw[2] === targetW2 && rw[3] === targetW3 && rw[4] === targetW4) {
                const keyHex = keys[i].toString(16).padStart(64, '0');
                console.log('[GPU] 🎯 WIN! key=' + keyHex + ' hash160=' + targetH160);
                handleGpuWin(keyHex, target, targetH160, false);
                gpuHuntRunning = false; found = true; break;
            }

            // 🎯 2. Sıkıştırılmamış (Uncompressed) Hedef Eşleşme Kontrolü (196ru37... ve Özel Havuz uncompressed adresler)
            if (checkUncomp) {
                for (let u = 0; u < 17; u++) waUWords[u] = raw[off + 9 + u];
                waUWords[16] = waUWords[16] & 0xFF000000;

                const shaU = cjs.SHA256(waU);
                const rmdU = cjs.RIPEMD160(shaU);
                const rwU = rmdU.words;

                if (targetW0 !== 0 && rwU[0] === targetW0 && rwU[1] === targetW1 && rwU[2] === targetW2 && rwU[3] === targetW3 && rwU[4] === targetW4) {
                    const keyHex = keys[i].toString(16).padStart(64, '0');
                    console.log('[GPU] 🎯 UNCOMPRESSED WIN! key=' + keyHex + ' hash160=' + targetH160);
                    handleGpuWin(keyHex, target, targetH160, true);
                    gpuHuntRunning = false; found = true; break;
                }

                if (cfg.isCustomPoolActive && typeof customAddressSet !== 'undefined' && customAddressSet) {
                    const h160U = rmdU.toString();
                    if (customAddressSet.has(h160U)) {
                        const keyHex = keys[i].toString(16).padStart(64, '0');
                        console.log('[GPU] 🎯 CUSTOM POOL (UNCOMPRESSED) WIN! key=' + keyHex);
                        handleGpuWin(keyHex, target, h160U, true);
                        gpuHuntRunning = false; found = true; break;
                    }
                }
            }

            // Çoklu hedef / özel havuz kontrolü (O(1) Map ve Set optimizasyonu)
            if (cfg.satoshiMultiTargetActive || cfg.isCustomPoolActive || cfg.isCircularMode) {
                const h160 = rmd.toString();
                if (cfg.unsolvedList && cfg.unsolvedList.length > 0) {
                    if (!window._gpuTargetMap || window._gpuTargetMapVersion !== cfg.unsolvedList.length) {
                        window._gpuTargetMap = new Map();
                        for (let p of cfg.unsolvedList) {
                            if (p.targetHash160) window._gpuTargetMap.set(p.targetHash160.toLowerCase(), p);
                        }
                        window._gpuTargetMapVersion = cfg.unsolvedList.length;
                    }
                    const matched = window._gpuTargetMap.get(h160.toLowerCase());
                    if (matched) {
                        const keyHex = keys[i].toString(16).padStart(64, '0');
                        console.log('[GPU] 🎯 HEDEF EŞLEŞTİ! key=' + keyHex + ' id=' + matched.id);
                        handleGpuWin(keyHex, matched, h160, false);
                        gpuHuntRunning = false; found = true; break;
                    }
                }
                if (cfg.isCustomPoolActive && typeof customAddressSet !== 'undefined'
                    && customAddressSet && customAddressSet.has(h160)) {
                    const keyHex = keys[i].toString(16).padStart(64, '0');
                    console.log('[GPU] 🎯 CUSTOM POOL WIN! key=' + keyHex);
                    handleGpuWin(keyHex, target, h160, false);
                    gpuHuntRunning = false; found = true; break;
                }
            }
        }

        const dt = (performance.now() - t0) / 1000;
        gpuKeysPerSec = Math.round(BATCH / (dt || 0.001));
        window.gpuKeysPerSec = gpuKeysPerSec;

        // UI için durum açıklama metni (Kaydıraç ve Algoritma gösterimi)
        const pParts = [];
        if (cfg.isDualSliderActive && (batchInfo.boundMin > batchInfo.baseStart || batchInfo.boundMax < batchInfo.baseEnd)) {
            pParts.push('🎚️ [0x' + batchInfo.boundMin.toString(16).substring(0, 6) + '..-0x' + batchInfo.boundMax.toString(16).substring(0, 6) + '..]');
        }
        if (batchInfo.activePrefix) {
            pParts.push('🧩 Ön Ek: 0x' + batchInfo.activePrefix);
        }
        let algoPrefix = '';
        if (batchInfo.activeAlgo === 'SOBOL') algoPrefix = '📐 Sobol + ';
        else if (batchInfo.activeAlgo === 'VD_CORPUT') algoPrefix = '📐 Van der Corput + ';
        else if (batchInfo.activeAlgo === 'WEYL_GOLDEN') algoPrefix = '🌟 Weyl Kafesi + ';
        else if (batchInfo.activeAlgo === 'COPRIME_STRIDE') algoPrefix = '♾️ Modüler Adım + ';
        else if (batchInfo.activeAlgo === 'HILBERT') algoPrefix = '🌀 Hilbert + ';
        else if (batchInfo.activeAlgo === 'CHAOS') algoPrefix = '♾️ Kaos + ';
        else if (batchInfo.activeAlgo === 'WEAK_ENTROPY') algoPrefix = '⚡ Zayıf Entropi + ';
        else if (batchInfo.activeAlgo === 'KANGAROO') algoPrefix = '🦘 Pollard Kangaroo + ';

        if (cfg.activeFormula === 'OMNI_CHAOS') algoPrefix = '♾️ Omni-Kaos + ' + algoPrefix;
        else if (cfg.activeFormula === 'GOLDEN_SINGULARITY') algoPrefix = '🌟 Altın Oran + ' + algoPrefix;

        const rangeDesc = algoPrefix + (pParts.join(' + ') || 'Standart');
        const rangeText = '⚡ WebGPU — ' + rangeDesc + ' (' + gpuKeysPerSec.toLocaleString() + ' key/s)';

        // Speed UI
        const spEl = document.getElementById('statSpeed');
        if (spEl) {
            spEl.innerText = gpuKeysPerSec.toLocaleString() + ' key/s';
        }

        // Progress (UI'da denenen son anahtarı ve tam aralığı gösterir)
        if (typeof handleWorkerMessage === 'function') {
            const lastK = (keys && keys.length > 0) ? keys[keys.length - 1] : null;
            if (lastK) {
                handleWorkerMessage({ data: {
                    type: 'progress',
                    workerId: 999,
                    count: BATCH,
                    lastKey: lastK.toString(16).padStart(64, '0'),
                    targetId: target.id,
                    rangeText: rangeText
                }});
            }
        }

        if (!found && gpuHuntRunning) setTimeout(gpuHuntBatch, gpuThrottleDelay);

    } catch(e) {
        console.error('[GPU] gpuHuntBatch hatası:', e);
        gpuHuntRunning = false;
        // CPU fallback
        if (typeof huntBatch === 'function') huntBatch();
    }
}

let gpuThrottleDelay = 45; // Varsayılan: 45ms dinlenme payı (akıcı 60 FPS masaüstü)
let gpuCurrentIntensity = 'BALANCED';

function setGpuIntensity(level) {
    if (!level) return;
    const l = String(level).toUpperCase();
    if (l.includes('ECO') || l.includes('LIGHT') || l.includes('LOW')) {
        gpuCurrentIntensity = 'ECO';
        gpuThrottleDelay = 120; // 120ms dinlenme payı: GPU ve CPU %95 boşta, sıfır kasma/donma
        if (gpuEngine) gpuEngine.batchSize = 64;
        console.log('[GPU] 🟢 Ultra Hafif / Eco moduna geçildi (Batch: 64, Gecikme: 120ms - Sıfır Kasma)');
    } else if (l.includes('MAX') || l.includes('TURBO') || l.includes('HIGH')) {
        gpuCurrentIntensity = 'MAX';
        gpuThrottleDelay = 2; // 2ms: Tam güç, tarayıcı olay döngüsüne minimal nefes
        if (gpuEngine) gpuEngine.batchSize = 1024;
        console.log('[GPU] 🔴 Tam Güç moduna geçildi (Batch: 1024, Gecikme: 2ms)');
    } else {
        // BALANCED (Varsayılan)
        gpuCurrentIntensity = 'BALANCED';
        gpuThrottleDelay = 45; // 45ms: Akıcı 60 FPS masaüstü
        if (gpuEngine) gpuEngine.batchSize = 256;
        console.log('[GPU] 🟡 Dengeli moduna geçildi (Batch: 256, Gecikme: 45ms - 60 FPS akıcı)');
    }
}

function handleGpuWin(keyHex, target, matchedH160, isUncompressed) {
    if (typeof handleWorkerMessage === 'function') {
        handleWorkerMessage({ data: {
            type: 'win',
            workerId: 999,
            keyHex: keyHex,
            targetId: target.id,
            targetAddr: target.addr,
            reward: target.reward || '⚡ GPU',
            matchedH160: matchedH160,
            isUncompressed: isUncompressed,
            matchType: isUncompressed ? 'GPU_UNCOMP' : 'GPU_COMP'
        }});
    }
}

function startGpuHunting() {
    if (!gpuEngine || !gpuEngine.isInitialized) {
        console.warn('[GPU] Engine hazır değil');
        return false;
    }
    gpuHuntRunning = true;
    console.log('[GPU] ⚡ Arama başladı — batch=' + gpuEngine.batchSize + ', throttleDelay=' + gpuThrottleDelay + 'ms, adapter=' + gpuEngine.adapterInfo);
    gpuHuntBatch();
    return true;
}

function stopGpuHunting() {
    gpuHuntRunning = false;
    if (gpuEngine) {
        gpuEngine.destroy();
        gpuEngine = null;
    }
    console.log('[GPU] GPU durduruldu');
}

// Global export
window.GpuEngine         = GpuEngine;
window.initWebGpuEngine  = initWebGpuEngine;
window.startGpuHunting   = startGpuHunting;
window.stopGpuHunting    = stopGpuHunting;
window._gpuEngineStop    = stopGpuHunting;
window.genBatchForGpu    = genBatchForGpu;
window.setGpuIntensity   = setGpuIntensity;
window.gpuKeysPerSec     = gpuKeysPerSec;
window.gpuTotalKeys      = gpuTotalKeys;
Object.defineProperty(window, 'gpuThrottleDelay', {
    get() { return gpuThrottleDelay; },
    set(v) { gpuThrottleDelay = v; },
    configurable: true
});
Object.defineProperty(window, 'gpuCurrentIntensity', {
    get() { return gpuCurrentIntensity; },
    set(v) { gpuCurrentIntensity = v; },
    configurable: true
});
