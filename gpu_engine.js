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

struct GpuParams {
    batch_size: u32,
    has_target: u32,
    target_w0: u32,
    target_w1: u32,
    target_w2: u32,
    target_w3: u32,
    target_w4: u32,
    _pad0: u32,
}

// Input: 8 u32 per key (LE), Output: 26 u32 per key (pubkey)
@group(0) @binding(0) var<storage, read>       privkeys:     array<u32>;
@group(0) @binding(1) var<storage, read_write> pubkeys:      array<u32>;
@group(0) @binding(2) var<uniform>             params:       GpuParams;
@group(0) @binding(3) var<storage, read_write> match_result: array<u32>;

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
            let s1 = add32c(acc[1], q9.x, 0u); acc[1] = s1.x;
            let qc = add32c(q9.y, ov9, 0u);
            let s2 = add32c(acc[2], qc.x, s1.y); acc[2] = s2.x;
            carry = qc.y + s2.y;
            for (var i = 3u; i < 8u; i++) {
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
// SHA-256 and RIPEMD-160 on GPU (WGSL 1.0)
// ----------------------------------------------------------------------------
const SHA256_K = array<u32, 64>(
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
);

const RL = array<u32, 80>(
    0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u,
    7u, 4u, 13u, 1u, 10u, 6u, 15u, 3u, 12u, 0u, 9u, 5u, 2u, 14u, 11u, 8u,
    3u, 10u, 14u, 4u, 9u, 15u, 8u, 1u, 2u, 7u, 0u, 6u, 13u, 11u, 5u, 12u,
    1u, 9u, 11u, 10u, 0u, 8u, 12u, 4u, 13u, 3u, 7u, 15u, 14u, 5u, 6u, 2u,
    4u, 0u, 5u, 9u, 7u, 12u, 2u, 10u, 14u, 1u, 3u, 8u, 11u, 6u, 15u, 13u
);
const RR = array<u32, 80>(
    5u, 14u, 7u, 0u, 9u, 2u, 11u, 4u, 13u, 6u, 15u, 8u, 1u, 10u, 3u, 12u,
    6u, 11u, 3u, 7u, 0u, 13u, 5u, 10u, 14u, 15u, 8u, 12u, 4u, 9u, 1u, 2u,
    15u, 5u, 1u, 3u, 7u, 14u, 6u, 9u, 11u, 8u, 12u, 2u, 10u, 0u, 4u, 13u,
    8u, 6u, 4u, 1u, 3u, 11u, 15u, 0u, 5u, 12u, 2u, 13u, 9u, 7u, 10u, 14u,
    12u, 15u, 10u, 4u, 1u, 5u, 8u, 7u, 6u, 2u, 13u, 14u, 0u, 3u, 9u, 11u
);
const SL = array<u32, 80>(
    11u, 14u, 15u, 12u, 5u, 8u, 7u, 9u, 11u, 13u, 14u, 15u, 6u, 7u, 9u, 8u,
    7u, 6u, 8u, 13u, 11u, 9u, 7u, 15u, 7u, 12u, 15u, 9u, 11u, 7u, 13u, 12u,
    11u, 13u, 6u, 7u, 14u, 9u, 13u, 15u, 14u, 8u, 13u, 6u, 5u, 12u, 7u, 5u,
    11u, 12u, 14u, 15u, 14u, 15u, 9u, 8u, 9u, 14u, 5u, 6u, 8u, 6u, 5u, 12u,
    9u, 15u, 5u, 11u, 6u, 8u, 13u, 12u, 5u, 12u, 13u, 14u, 11u, 8u, 5u, 6u
);
const SR = array<u32, 80>(
    8u, 9u, 9u, 11u, 13u, 15u, 15u, 5u, 7u, 7u, 8u, 11u, 14u, 14u, 12u, 6u,
    9u, 13u, 15u, 7u, 12u, 8u, 9u, 11u, 7u, 7u, 12u, 7u, 6u, 15u, 13u, 11u,
    9u, 7u, 15u, 11u, 8u, 6u, 6u, 14u, 12u, 13u, 5u, 14u, 13u, 13u, 7u, 5u,
    15u, 5u, 8u, 11u, 14u, 14u, 6u, 14u, 6u, 9u, 12u, 9u, 12u, 5u, 15u, 8u,
    8u, 5u, 12u, 9u, 12u, 5u, 14u, 6u, 8u, 13u, 6u, 5u, 15u, 13u, 11u, 11u
);

fn rotl(v: u32, n: u32) -> u32 {
    return (v << n) | (v >> (32u - n));
}

fn bswap32(w: u32) -> u32 {
    return ((w & 0xFFu) << 24u) | (((w >> 8u) & 0xFFu) << 16u) | (((w >> 16u) & 0xFFu) << 8u) | ((w >> 24u) & 0xFFu);
}

fn f_ripemd(round: u32, x: u32, y: u32, z: u32) -> u32 {
    if (round == 0u) { return x ^ y ^ z; }
    if (round == 1u) { return (x & y) | ((~x) & z); }
    if (round == 2u) { return (x | (~y)) ^ z; }
    if (round == 3u) { return (x & z) | (y & (~z)); }
    return x ^ (y | (~z));
}

// ----------------------------------------------------------------------------
// Compute entry point
// ----------------------------------------------------------------------------
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx   = gid.x;
    let total = arrayLength(&privkeys) / 8u;
    if (idx >= total || idx >= params.batch_size) { return; }

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

    let prefix = 2u + (ay.l[0] & 1u);

    // 1. SHA-256 for 33-byte compressed public key
    var W: array<u32, 64>;
    W[0] = (prefix << 24u) | (ax.l[7] >> 8u);
    W[1] = (ax.l[7] << 24u) | (ax.l[6] >> 8u);
    W[2] = (ax.l[6] << 24u) | (ax.l[5] >> 8u);
    W[3] = (ax.l[5] << 24u) | (ax.l[4] >> 8u);
    W[4] = (ax.l[4] << 24u) | (ax.l[3] >> 8u);
    W[5] = (ax.l[3] << 24u) | (ax.l[2] >> 8u);
    W[6] = (ax.l[2] << 24u) | (ax.l[1] >> 8u);
    W[7] = (ax.l[1] << 24u) | (ax.l[0] >> 8u);
    W[8] = ((ax.l[0] & 0xFFu) << 24u) | 0x00800000u;
    for (var t = 9u; t < 15u; t++) { W[t] = 0u; }
    W[15] = 264u;

    for (var t = 16u; t < 64u; t++) {
        let s0 = ((W[t - 15u] >> 7u) | (W[t - 15u] << 25u)) ^ ((W[t - 15u] >> 18u) | (W[t - 15u] << 14u)) ^ (W[t - 15u] >> 3u);
        let s1 = ((W[t - 2u] >> 17u) | (W[t - 2u] << 15u)) ^ ((W[t - 2u] >> 19u) | (W[t - 2u] << 13u)) ^ (W[t - 2u] >> 10u);
        W[t] = W[t - 16u] + s0 + W[t - 7u] + s1;
    }

    var a = 0x6a09e667u;
    var b = 0xbb67ae85u;
    var c = 0x3c6ef372u;
    var d = 0xa54ff53au;
    var e = 0x510e527fu;
    var f = 0x9b05688cu;
    var g = 0x1f83d9abu;
    var h = 0x5be0cd19u;

    for (var t = 0u; t < 64u; t++) {
        let S1 = ((e >> 6u) | (e << 26u)) ^ ((e >> 11u) | (e << 21u)) ^ ((e >> 25u) | (e << 7u));
        let ch = (e & f) ^ ((~e) & g);
        let temp1 = h + S1 + ch + SHA256_K[t] + W[t];
        let S0 = ((a >> 2u) | (a << 30u)) ^ ((a >> 13u) | (a << 19u)) ^ ((a >> 22u) | (a << 10u));
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = S0 + maj;

        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }

    // 2. RIPEMD-160 for 32-byte SHA-256 output
    var X: array<u32, 16>;
    X[0] = bswap32(a + 0x6a09e667u);
    X[1] = bswap32(b + 0xbb67ae85u);
    X[2] = bswap32(c + 0x3c6ef372u);
    X[3] = bswap32(d + 0xa54ff53au);
    X[4] = bswap32(e + 0x510e527fu);
    X[5] = bswap32(f + 0x9b05688cu);
    X[6] = bswap32(g + 0x1f83d9abu);
    X[7] = bswap32(h + 0x5be0cd19u);
    X[8] = 0x00000080u;
    for (var xi = 9u; xi < 14u; xi++) { X[xi] = 0u; }
    X[14] = 256u;
    X[15] = 0u;

    var al = 0x67452301u; var bl = 0xefcdab89u; var cl = 0x98badcfeu; var dl = 0x10325476u; var el = 0xc3d2e1f0u;
    var ar = 0x67452301u; var br = 0xefcdab89u; var cr = 0x98badcfeu; var dr = 0x10325476u; var er = 0xc3d2e1f0u;

    for (var i = 0u; i < 80u; i++) {
        let round_l = i / 16u;
        var kl = 0u;
        if (round_l == 1u) { kl = 0x5a827999u; }
        else if (round_l == 2u) { kl = 0x6ed9eba1u; }
        else if (round_l == 3u) { kl = 0x8f1bbcdcu; }
        else if (round_l == 4u) { kl = 0xa953fd4eu; }

        let fl = f_ripemd(round_l, bl, cl, dl);
        var tl = al + fl + X[RL[i]] + kl;
        tl = rotl(tl, SL[i]) + el;
        al = el; el = dl; dl = rotl(cl, 10u); cl = bl; bl = tl;

        let round_r = 4u - round_l;
        var kr = 0u;
        if (round_r == 4u) { kr = 0x50a28be6u; }
        else if (round_r == 3u) { kr = 0x5c4dd124u; }
        else if (round_r == 2u) { kr = 0x6d703ef3u; }
        else if (round_r == 1u) { kr = 0x7a6d76e9u; }

        let fr = f_ripemd(round_r, br, cr, dr);
        var tr = ar + fr + X[RR[i]] + kr;
        tr = rotl(tr, SR[i]) + er;
        ar = er; er = dr; dr = rotl(cr, 10u); cr = br; br = tr;
    }

    let r0 = bswap32(0xefcdab89u + cl + dr);
    let r1 = bswap32(0x98badcfeu + dl + er);
    let r2 = bswap32(0x10325476u + el + ar);
    let r3 = bswap32(0xc3d2e1f0u + al + br);
    let r4 = bswap32(0x67452301u + bl + cr);

    // 3. Target Hash160 Matching on GPU
    if (params.has_target == 1u) {
        if (r0 == params.target_w0 && r1 == params.target_w1 && r2 == params.target_w2 && r3 == params.target_w3 && r4 == params.target_w4) {
            match_result[0] = 1u;
            match_result[1] = idx;
            for (var wi = 0u; wi < 8u; wi++) {
                match_result[2u + wi] = k.l[wi];
            }
            match_result[10] = r0;
            match_result[11] = r1;
            match_result[12] = r2;
            match_result[13] = r3;
            match_result[14] = r4;
            match_result[15] = 0u;
        }
    }

    // Report last key for UI status
    if (idx == params.batch_size - 1u) {
        for (var wi = 0u; wi < 8u; wi++) {
            match_result[16u + wi] = k.l[wi];
        }
    }
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
        this.paramsBuf   = null;
        this.readbackBuf = null;
        this.bindGroup   = null;
        this.maxBatchSize = 2048;
        this.batchSize   = (typeof gpuCurrentIntensity !== 'undefined' && gpuCurrentIntensity === 'ECO') ? 64
                         : ((typeof gpuCurrentIntensity !== 'undefined' && gpuCurrentIntensity === 'MAX') ? 256 : 128);
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
            return await this.initWithDevice(device, this.adapterInfo);
        } catch(e) {
            console.error('[GPU] WebGPU init hatası:', e);
            return false;
        }
    }

    async initWithDevice(device, adapterInfo) {
        if (!device) return false;
        try {
            this.device = device;
            this.adapterInfo = adapterInfo || 'WebGPU GPU';

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

            const privBytes = this.maxBatchSize * 8 * 4;
            const pubBytes  = this.maxBatchSize * 26 * 4;

            this.privkeyBuf = this.device.createBuffer({
                size: privBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            });
            this.pubkeyBuf = this.device.createBuffer({
                size: pubBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
            });
            this.readbackBuf = this.device.createBuffer({
                size: pubBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });
            this.paramsBuf = this.device.createBuffer({
                size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            });
            this.matchBuf = this.device.createBuffer({
                size: 128, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
            });
            this.readbackMatchBuf = this.device.createBuffer({
                size: 128, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });

            this.bindGroup = this.device.createBindGroup({
                layout: this.pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.privkeyBuf } },
                    { binding: 1, resource: { buffer: this.pubkeyBuf  } },
                    { binding: 2, resource: { buffer: this.paramsBuf  } },
                    { binding: 3, resource: { buffer: this.matchBuf   } }
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
     * ⚡ GPU-Native HASH160 Arama Çekirdeği (SHA256 + RIPEMD160 + Eşleşme Denetimi %100 GPU İçinde)
     */
    async huntBatchGpu(privkeysBig, targetWords) {
        if (!this.isInitialized) throw new Error('GPU başlatılmamış');
        const n = privkeysBig.length;
        if (n === 0) return { matched: false, keyHex: '', lastKeyHex: '' };

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

        // matchBuf sıfırlama (128 bytes = 32 x u32)
        const zeroMatch = new Uint32Array(32);
        this.device.queue.writeBuffer(this.matchBuf, 0, zeroMatch.buffer);

        const hasTarget = (targetWords && targetWords.length === 5) ? 1 : 0;
        const pData = new Uint32Array([
            n,
            hasTarget,
            hasTarget ? targetWords[0] : 0,
            hasTarget ? targetWords[1] : 0,
            hasTarget ? targetWords[2] : 0,
            hasTarget ? targetWords[3] : 0,
            hasTarget ? targetWords[4] : 0,
            0
        ]);
        this.device.queue.writeBuffer(this.paramsBuf, 0, pData.buffer);

        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.dispatchWorkgroups(Math.ceil(n / 64));
        pass.end();

        // SADECE 128 byte'lık sonuç tamponunu geri oku (Sıfır PCIe darboğazı)
        encoder.copyBufferToBuffer(this.matchBuf, 0, this.readbackMatchBuf, 0, 128);
        this.device.queue.submit([encoder.finish()]);

        await this.readbackMatchBuf.mapAsync(GPUMapMode.READ, 0, 128);
        try {
            const mapped = new Uint32Array(this.readbackMatchBuf.getMappedRange(0, 128));
            const isFound = mapped[0] === 1;
            let winningKeyHex = '';
            if (isFound) {
                let winK = 0n;
                for (let wi = 7; wi >= 0; wi--) {
                    winK = (winK << 32n) | BigInt(mapped[2 + wi] >>> 0);
                }
                winningKeyHex = winK.toString(16).padStart(64, '0');
            }
            let lastK = 0n;
            for (let wi = 7; wi >= 0; wi--) {
                lastK = (lastK << 32n) | BigInt(mapped[16 + wi] >>> 0);
            }
            const lastKeyHex = lastK ? lastK.toString(16).padStart(64, '0') : '';
            return {
                matched: isFound,
                keyHex: winningKeyHex,
                lastKeyHex: lastKeyHex
            };
        } finally {
            this.readbackMatchBuf.unmap();
        }
    }

    /**
     * Ham Uint32Array olarak pubkey kelimelerini döner (Geriye dönük uyumluluk)
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
        const paramsData = new Uint32Array([n, 0, 0, 0, 0, 0, 0, 0]);
        this.device.queue.writeBuffer(this.paramsBuf, 0, paramsData.buffer);

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
        try {
            const raw = new Uint32Array(this.readbackBuf.getMappedRange(0, n * 104).slice(0));
            return raw;
        } finally {
            this.readbackBuf.unmap();
        }
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
            if (this.privkeyBuf)        this.privkeyBuf.destroy();
            if (this.pubkeyBuf)         this.pubkeyBuf.destroy();
            if (this.readbackBuf)       this.readbackBuf.destroy();
            if (this.paramsBuf)         this.paramsBuf.destroy();
            if (this.matchBuf)          this.matchBuf.destroy();
            if (this.readbackMatchBuf)  this.readbackMatchBuf.destroy();
            if (this.device)            this.device.destroy();
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

let gpuWorker = null;
let isGpuWorkerActive = false;
let gpuWorkerBlobUrl = null;
let gpuWorkerAdapterInfo = '';

const GPU_WORKER_RUNNER_LOGIC = `
let workerDevice = null;
let workerAdapterInfo = '';
let workerGpuEngine = null;
let workerKangarooEngine = null;
let workerHuntRunning = false;

let workerIntensity = 'BALANCED';
let workerThrottleDelay = 35;
let workerStepsPerCall = 16;
let workerBatchSize = 128;

let workerCfg = null;
let workerTotalKeys = 0;
let workerLastTime = 0;
let workerAccumSteps = 0;

let workerTargetWords0 = new Set();
let workerTargetMap = new Map();
let workerCustomAddressSet = null;

let targetW0 = 0, targetW1 = 0, targetW2 = 0, targetW3 = 0, targetW4 = 0;
let hasTargetHash160 = false;
let cachedTargetH160Str = '';

function ensureTargetWords(h160) {
    if (h160 === cachedTargetH160Str) return;
    cachedTargetH160Str = h160;
    if (h160 && h160.length === 40) {
        targetW0 = parseInt(h160.substr(0, 8), 16) | 0;
        targetW1 = parseInt(h160.substr(8, 8), 16) | 0;
        targetW2 = parseInt(h160.substr(16, 8), 16) | 0;
        targetW3 = parseInt(h160.substr(24, 8), 16) | 0;
        targetW4 = parseInt(h160.substr(32, 8), 16) | 0;
        hasTargetHash160 = true;
    } else {
        targetW0 = targetW1 = targetW2 = targetW3 = targetW4 = 0;
        hasTargetHash160 = false;
    }
}

async function workerKangarooLoop() {
    if (!workerHuntRunning || !workerKangarooEngine) return;
    try {
        const batchRes = await workerKangarooEngine.dispatchBatch();
        const steps = (typeof batchRes === 'object') ? batchRes.steps : Number(batchRes);
        const lastKey = (typeof batchRes === 'object' && batchRes.lastKey) ? batchRes.lastKey : '';
        workerAccumSteps += steps;
        workerTotalKeys += steps;

        const now = performance.now();
        const elapsed = (now - workerLastTime) / 1000;
        if (elapsed >= 0.25) {
            const rate = Math.round(workerAccumSteps / (elapsed || 0.001));
            const trapCount = workerKangarooEngine.tameTraps.size + workerKangarooEngine.wildTraps.size;

            self.postMessage({
                type: 'progress',
                workerId: 999,
                count: workerAccumSteps,
                rate: rate,
                trapCount: trapCount,
                lastKey: lastKey,
                rangeText: '⚡ WebGPU Worker Kangaroo — 2048 Sürü (' + rate.toLocaleString() + ' key/s)'
            });
            workerAccumSteps = 0;
            workerLastTime = now;
        }

        if (workerKangarooEngine.isSolved && workerKangarooEngine.solvedKey) {
            const keyHex = workerKangarooEngine.solvedKey.toString(16).padStart(64, '0');
            const target = workerCfg ? workerCfg.activeTarget : null;
            self.postMessage({
                type: 'win',
                workerId: 999,
                keyHex: keyHex,
                targetId: target ? target.id : '',
                targetAddr: target ? target.addr : '',
                reward: (target && target.reward ? target.reward : '⚡ GPU') + ' (WebGPU Worker Kangaroo)',
                matchedH160: target ? target.targetHash160 : '',
                isUncompressed: false,
                matchType: 'GPU_KANGAROO'
            });
            workerHuntRunning = false;
            return;
        }

        if (workerHuntRunning) {
            setTimeout(workerKangarooLoop, workerThrottleDelay);
        }
    } catch(err) {
        console.error('[WebGPU Worker Kangaroo Error]:', err);
        self.postMessage({ type: 'error', error: err.message });
        workerHuntRunning = false;
    }
}

async function workerHuntLoop() {
    if (!workerHuntRunning || !workerGpuEngine) return;
    try {
        const t0 = performance.now();
        const batchInfo = genBatchForGpu(workerCfg, workerGpuEngine.batchSize);
        const keys = batchInfo.keys;
        const BATCH = keys.length;
        if (BATCH === 0) {
            if (workerHuntRunning) setTimeout(workerHuntLoop, workerThrottleDelay);
            return;
        }

        const target = workerCfg ? workerCfg.activeTarget : null;
        const hasTargetH160 = Boolean(target && target.targetHash160);
        let targetH160 = hasTargetH160 ? target.targetHash160.trim().toLowerCase() : '';
        ensureTargetWords(targetH160);

        // ⚡ GPU computes Pubkeys + SHA-256 + RIPEMD-160 + Target Match in single kernel!
        const gpuRes = await workerGpuEngine.huntBatchGpu(
            keys,
            hasTargetHash160 ? [targetW0, targetW1, targetW2, targetW3, targetW4] : null
        );

        let found = false;
        if (gpuRes && gpuRes.matched) {
            const keyHex = gpuRes.keyHex || (keys[0] ? keys[0].toString(16).padStart(64, '0') : '');
            self.postMessage({
                type: 'win',
                workerId: 999,
                keyHex: keyHex,
                targetId: target ? target.id : '',
                targetAddr: target ? target.addr : '',
                reward: (target && target.reward) ? target.reward : '⚡ GPU',
                matchedH160: targetH160,
                isUncompressed: false,
                matchType: 'GPU_SHADER_HASH160'
            });
            workerHuntRunning = false;
            found = true;
            return;
        }

        // Çoklu Hedef veya İlave Havuz (Fallback Kontrolü)
        const hasMultiTarget = Boolean(workerCustomAddressSet && workerCustomAddressSet.size > 0) || Boolean(workerTargetMap && workerTargetMap.size > 1);
        if (hasMultiTarget) {
            const raw = await workerGpuEngine.computePubkeysRaw(keys);
            const cjs = (typeof self !== 'undefined' && self.CryptoJS) ? self.CryptoJS : (typeof CryptoJS !== 'undefined' ? CryptoJS : null);
            if (cjs && raw) {
                const waWords = [0, 0, 0, 0, 0, 0, 0, 0, 0];
                const wa = { words: waWords, sigBytes: 33 };

                for (let i = 0; i < BATCH && workerHuntRunning; i++) {
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

                    if (workerTargetMap && workerTargetMap.size > 0) {
                        const w0Match = workerTargetWords0.size === 0 || workerTargetWords0.has(rw[0] >>> 0);
                        if (w0Match) {
                            const h160 = rmd.toString().toLowerCase();
                            const matched = workerTargetMap.get(h160);
                            if (matched) {
                                const keyHex = keys[i].toString(16).padStart(64, '0');
                                self.postMessage({
                                    type: 'win',
                                    workerId: 999,
                                    keyHex: keyHex,
                                    targetId: matched.id,
                                    targetAddr: matched.addr,
                                    reward: matched.reward || '⚡ GPU',
                                    matchedH160: h160,
                                    isUncompressed: false,
                                    matchType: 'GPU_MULTI_TARGET'
                                });
                                workerHuntRunning = false; found = true; break;
                            }
                        }
                    }
                }
            }
        }

        workerAccumSteps += BATCH;
        workerTotalKeys += BATCH;

        const now = performance.now();
        const elapsed = (now - workerLastTime) / 1000;
        if (elapsed >= 0.25) {
            const rate = Math.round(workerAccumSteps / (elapsed || 0.001));
            const lastK = (gpuRes && gpuRes.lastKeyHex) ? gpuRes.lastKeyHex : (keys && keys.length > 0 ? keys[keys.length - 1].toString(16).padStart(64, '0') : '');
            self.postMessage({
                type: 'progress',
                workerId: 999,
                count: workerAccumSteps,
                rate: rate,
                lastKey: lastK,
                targetId: target ? target.id : '',
                rangeText: '⚡ WebGPU HASH160 — ' + (batchInfo.rangeDesc || 'Standart') + ' (' + rate.toLocaleString() + ' key/s)'
            });
            workerAccumSteps = 0;
            workerLastTime = now;
        }

        if (!found && workerHuntRunning) {
            setTimeout(workerHuntLoop, workerThrottleDelay);
        }
    } catch(err) {
        console.error('[WebGPU Worker Standard Loop Error]:', err);
        self.postMessage({ type: 'error', error: err.message });
        workerHuntRunning = false;
    }
}

self.onmessage = async function(e) {
    const data = e.data;
    if (!data) return;

    if (data.cmd === 'init') {
        try {
            if (!navigator.gpu) {
                self.postMessage({ type: 'gpu_unsupported', reason: 'Worker navigator.gpu desteklenmiyor' });
                return;
            }
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                self.postMessage({ type: 'gpu_unsupported', reason: 'GPU Adapter bulunamadı' });
                return;
            }
            workerDevice = await adapter.requestDevice();
            workerAdapterInfo = (adapter.info && adapter.info.description) ? adapter.info.description : (adapter.name || 'WebGPU GPU');
            self.postMessage({ type: 'gpu_ready', adapterInfo: workerAdapterInfo });
        } catch(err) {
            self.postMessage({ type: 'gpu_unsupported', reason: err.message });
        }
    } else if (data.cmd === 'start') {
        workerCfg = data.config;
        workerIntensity = data.intensity || 'BALANCED';
        workerThrottleDelay = data.throttleDelay || 35;
        workerStepsPerCall = data.stepsPerCall || 16;
        workerBatchSize = (workerIntensity === 'ECO') ? 512 : ((workerIntensity === 'MAX') ? 4096 : 2048);

        workerTotalKeys = 0;
        workerLastTime = performance.now();
        workerAccumSteps = 0;
        workerHuntRunning = true;

        workerTargetMap.clear();
        workerTargetWords0.clear();
        if (workerCfg && workerCfg.unsolvedList) {
            for (const p of workerCfg.unsolvedList) {
                if (p.targetHash160) {
                    const cleanH = p.targetHash160.trim().toLowerCase();
                    workerTargetMap.set(cleanH, p);
                    workerTargetWords0.add(parseInt(cleanH.substr(0, 8), 16) | 0);
                }
            }
        }
        if (data.customAddresses && Array.isArray(data.customAddresses)) {
            workerCustomAddressSet = new Set(data.customAddresses);
        }

        const isKangaroo = data.isKangaroo;
        const target = workerCfg ? workerCfg.activeTarget : null;

        if (isKangaroo && data.targetPubHex) {
            let s = toBigIntSafeGpu(target.start);
            let e = toBigIntSafeGpu(target.end);
            if (workerCfg && workerCfg.isDualSliderActive && workerCfg.sliderMinHex && workerCfg.sliderMaxHex) {
                const sMin = toBigIntSafeGpu(workerCfg.sliderMinHex);
                const sMax = toBigIntSafeGpu(workerCfg.sliderMaxHex);
                if (sMin >= s && sMax <= e && sMax > sMin) {
                    s = sMin; e = sMax;
                }
            }

            workerKangarooEngine = new WebGpuKangarooEngine();
            workerKangarooEngine.stepsPerCall = workerStepsPerCall;
            const ok = await workerKangarooEngine.init(workerDevice, data.targetPubHex, s, e);
            if (ok) {
                workerKangarooEngine.setStepsPerCall(workerStepsPerCall);
                workerKangarooLoop();
            } else {
                self.postMessage({ type: 'error', error: 'Kangaroo Engine başlatılamadı' });
                workerHuntRunning = false;
            }
        } else {
            workerGpuEngine = new GpuEngine();
            workerGpuEngine.batchSize = workerBatchSize;
            const ok = await workerGpuEngine.initWithDevice(workerDevice, workerAdapterInfo);
            if (ok) {
                workerHuntLoop();
            } else {
                self.postMessage({ type: 'error', error: 'GpuEngine başlatılamadı' });
                workerHuntRunning = false;
            }
        }
    } else if (data.cmd === 'stop') {
        workerHuntRunning = false;
        if (workerKangarooEngine) {
            workerKangarooEngine.destroy();
            workerKangarooEngine = null;
        }
        if (workerGpuEngine) {
            workerGpuEngine.destroyBuffers();
            workerGpuEngine = null;
        }
    } else if (data.cmd === 'setIntensity') {
        workerIntensity = data.intensity || workerIntensity;
        workerThrottleDelay = data.throttleDelay || workerThrottleDelay;
        workerStepsPerCall = data.stepsPerCall || workerStepsPerCall;
        workerBatchSize = (workerIntensity === 'ECO') ? 512 : ((workerIntensity === 'MAX') ? 4096 : 2048);

        if (workerKangarooEngine) {
            workerKangarooEngine.setStepsPerCall(workerStepsPerCall);
        }
        if (workerGpuEngine) {
            workerGpuEngine.batchSize = workerBatchSize;
        }
    }
};
`;

function createGpuWorker() {
    if (gpuWorker) {
        try { gpuWorker.terminate(); } catch(e) {}
        gpuWorker = null;
    }

    // 1. Önce doğrudan gpu_worker.js dosyasını dene (HTTP / Web Server ortamı)
    try {
        const w = new Worker('gpu_worker.js');
        return w;
    } catch(err) {
        console.warn('[GPU] Doğrudan Worker("gpu_worker.js") açılamadı (file:// CORS kısıtlaması olabilir), Blob Worker deneniyor...', err.message);
    }

    // 2. Local file:// altında CORS hatasını aşmak için DOM'daki kütüphanelerle dinamik Blob Worker oluştur
    try {
        if (gpuWorkerBlobUrl) {
            try { URL.revokeObjectURL(gpuWorkerBlobUrl); } catch(e) {}
            gpuWorkerBlobUrl = null;
        }

        let coreLibsCode = '';
        const coreTag = (typeof document !== 'undefined') ? document.getElementById('core-libs') : null;
        if (coreTag) {
            coreLibsCode = coreTag.textContent;
        }

        const workerSource = `(function() {
${coreLibsCode}
}).call(self);

const SECP256K1_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141n;
const GPU_WGSL = ${JSON.stringify(GPU_WGSL)};
const WGSL_KANGAROO_SHADER = ${typeof WGSL_KANGAROO_SHADER !== 'undefined' ? JSON.stringify(WGSL_KANGAROO_SHADER) : "''"};
${typeof WebGpuKangarooEngine !== 'undefined' ? WebGpuKangarooEngine.toString() : ''};
${GpuEngine.toString()};
${toBigIntSafeGpu.toString()};

${typeof GPU_KEY_GENERATORS_SOURCE !== 'undefined' ? GPU_KEY_GENERATORS_SOURCE : ''}

${genBatchForGpu.toString()};

${GPU_WORKER_RUNNER_LOGIC}
`;
        const blob = new Blob([workerSource], { type: 'application/javascript' });
        gpuWorkerBlobUrl = URL.createObjectURL(blob);
        return new Worker(gpuWorkerBlobUrl);
    } catch(err) {
        console.warn('[GPU] Blob Worker oluşturulamadı:', err);
        return null;
    }
}

function setupGpuWorkerListeners(worker) {
    if (!worker) return;
    worker.onmessage = function(e) {
        const data = e.data;
        if (!data) return;

        if (data.type === 'progress') {
            gpuKeysPerSec = data.rate || 0;
            gpuTotalKeys = (gpuTotalKeys || 0) + (data.count || 0);
            window.gpuKeysPerSec = gpuKeysPerSec;
            window.gpuTotalKeys = gpuTotalKeys;

            const spEl = document.getElementById('statSpeed');
            if (spEl) {
                spEl.innerText = gpuKeysPerSec.toLocaleString() + ' key/s';
            }

            if (typeof handleWorkerMessage === 'function') {
                handleWorkerMessage(e);
            }
        } else if (data.type === 'win') {
            gpuHuntRunning = false;
            if (typeof handleWorkerMessage === 'function') {
                handleWorkerMessage(e);
            }
        } else if (data.type === 'error') {
            console.error('[GPU Worker Error]:', data.error);
        }
    };

    worker.onerror = function(err) {
        console.error('[GPU Worker Runtime Error]:', err.message);
    };
}

async function initWebGpuEngine() {
    isGpuWorkerActive = false;

    // 1. Web Worker içinde WebGPU'yu test et ve başlat (İzole Thread)
    try {
        const w = createGpuWorker();
        if (w) {
            const workerInitOk = await new Promise((resolve) => {
                const timeout = setTimeout(() => {
                    resolve(false);
                }, 2000);

                w.onmessage = function(e) {
                    const data = e.data;
                    if (!data) return;
                    if (data.type === 'gpu_ready') {
                        clearTimeout(timeout);
                        gpuWorkerAdapterInfo = data.adapterInfo || 'WebGPU GPU (Worker)';
                        console.log('[GPU] 🚀 WebGPU Web Worker Başarıyla Aktif! (' + gpuWorkerAdapterInfo + ')');
                        resolve(true);
                    } else if (data.type === 'gpu_unsupported') {
                        clearTimeout(timeout);
                        console.warn('[GPU] Worker içinde WebGPU desteklenmiyor:', data.reason);
                        resolve(false);
                    }
                };

                w.onerror = function(err) {
                    clearTimeout(timeout);
                    console.warn('[GPU] WebGPU Worker hatası:', err.message);
                    resolve(false);
                };

                w.postMessage({ cmd: 'init' });
            });

            if (workerInitOk) {
                gpuWorker = w;
                isGpuWorkerActive = true;
                setupGpuWorkerListeners(gpuWorker);
                return true;
            } else {
                try { w.terminate(); } catch(e) {}
            }
        }
    } catch(err) {
        console.warn('[GPU] Worker başlatma sırasında istisna:', err);
    }

    // 2. Worker WebGPU başarısız olursa (Fallback: Ana thread)
    console.log('[GPU] ℹ️ Ana thread WebGPU motoru başlatılıyor (Fallback)...');
    isGpuWorkerActive = false;
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

if (typeof SECP_P === 'undefined') {
    var SECP_P      = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2Fn;
    var SECP_N      = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141n;
    var SECP_BETA   = 0x7AE96A2B657C07106E64479EAC3434E99CF0497512F58995C1396C28719501EEn;
    var SECP_LAMBDA = 0x5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72n;
}

function toBigIntSafeGpu(val) {
    if (val === null || typeof val === 'undefined') return 0n;
    if (typeof val === 'bigint') return val;
    let s = val.toString().trim().toLowerCase();
    if (!s || s === '0x') return 0n;
    if (s.startsWith('0x')) return BigInt(s);
    return BigInt('0x' + s);
}

const GPU_KEY_GENERATORS_SOURCE = `
let sobolStep = 1;
function generateSobolKey(rangeStart, rangeEnd) {
    sobolStep++;
    const totalRange = rangeEnd - rangeStart;
    if (totalRange <= 0n) return rangeStart;
    let n = BigInt(sobolStep);
    let rev = 0n;
    let temp = n & 0xFFFFFFFFFFFFFFFFn;
    for (let i = 0; i < 64; i++) {
        rev = (rev << 1n) | (temp & 1n);
        temp >>= 1n;
    }
    const g = n ^ (n >> 1n);
    const mix = (rev ^ (g * 0x9E3779B97F4A7C15n)) & 0xFFFFFFFFFFFFFFFFn;
    const spanBits = totalRange.toString(2).length;
    let offset;
    if (spanBits > 60) {
        const g2 = ((n * 0x517cc1b727220a95n) ^ 0x6c62272e07bb0142n) & 0xFFFFFFFFFFFFFFFFn;
        const fullMix = (mix << 64n) | g2;
        offset = (fullMix * (totalRange + 1n)) >> 128n;
    } else {
        offset = (mix * (totalRange + 1n)) >> 64n;
    }
    const cand = rangeStart + offset;
    return cand > rangeEnd ? rangeEnd : cand;
}

let scrambledSobolStep = 1;
function generateScrambledSobolKey(rangeStart, rangeEnd) {
    scrambledSobolStep++;
    const totalRange = rangeEnd - rangeStart;
    if (totalRange <= 0n) return rangeStart;
    let n = BigInt(scrambledSobolStep);
    const g = n ^ (n >> 1n);
    let s1 = (g ^ 0x6a09e667f3bcc908n) * 0x9e3779b97f4a7c15n;
    s1 ^= (s1 >> 32n);
    s1 = (s1 * 0xbf58476d1ce4e5b9n) & 0xFFFFFFFFFFFFFFFFn;
    s1 ^= (s1 >> 27n);
    let rev = 0n, temp = s1;
    for (let i = 0; i < 64; i++) {
        rev = (rev << 1n) | (temp & 1n);
        temp >>= 1n;
    }
    const spanBits = totalRange.toString(2).length;
    let offset;
    if (spanBits > 60) {
        let s2 = (g ^ 0xbb67ae8584caa73bn) * 0x517cc1b727220a95n;
        s2 ^= (s2 >> 29n);
        const fullMix = (rev << 64n) | (s2 & 0xFFFFFFFFFFFFFFFFn);
        offset = (fullMix * (totalRange + 1n)) >> 128n;
    } else {
        offset = (rev * (totalRange + 1n)) >> 64n;
    }
    const cand = rangeStart + offset;
    return cand > rangeEnd ? rangeEnd : cand;
}

let haltonStep = 1n;
function generateHaltonKey(rangeStart, rangeEnd) {
    haltonStep++;
    if (haltonStep > 1000000000000n) haltonStep = 1n;
    const totalRange = rangeEnd - rangeStart;
    if (totalRange <= 0n) return rangeStart;
    // Halton: taban 2 ve taban 3 radikal tersleri sırayla kullanılır (her biri tek başına düzgün dağılımlıdır)
    const m = haltonStep >> 1n;
    let value;
    if ((haltonStep & 1n) === 0n) {
        let rev2 = 0n, temp = m & 0xFFFFFFFFFFFFFFFFn;
        for (let i = 0; i < 64; i++) {
            rev2 = (rev2 << 1n) | (temp & 1n);
            temp >>= 1n;
        }
        value = Number(rev2 >> 11n) / 9007199254740992.0;
    } else {
        let rev3 = 0.0, f3 = 1.0 / 3.0, n3 = m;
        while (n3 > 0n) {
            const d = Number(n3 % 3n);
            rev3 += d * f3;
            n3 /= 3n;
            f3 /= 3.0;
        }
        value = rev3;
    }
    const scale = BigInt(Math.floor(value * 1000000000000000.0));
    const cand = rangeStart + (totalRange * scale) / 1000000000000000n;
    return cand > rangeEnd ? rangeEnd : cand;
}

let vdCorputStep = 1n;

let weylStep = 1n;
function generateWeylGoldenKey(rangeStart, rangeEnd) {
    weylStep++;
    const totalRange = rangeEnd - rangeStart;
    if (totalRange <= 0n) return rangeStart;
    const weylFrac = (weylStep * 0x9E3779B97F4A7C15n) & 0xFFFFFFFFFFFFFFFFn;
    const spanBits = totalRange.toString(2).length;
    let offset;
    if (spanBits > 60) {
        const weylFrac2 = (weylStep * 0xC6A4A7935BD1E995n) & 0xFFFFFFFFFFFFFFFFn;
        const fullFrac = (weylFrac << 64n) | weylFrac2;
        offset = (fullFrac * (totalRange + 1n)) >> 128n;
    } else {
        offset = (weylFrac * (totalRange + 1n)) >> 64n;
    }
    const cand = rangeStart + offset;
    return cand > rangeEnd ? rangeEnd : cand;
}

let coprimeStep = 1n;
function generateCoprimeStrideKey(rangeStart, rangeEnd) {
    coprimeStep++;
    const totalRange = rangeEnd - rangeStart;
    if (totalRange <= 0n) return rangeStart;
    const mod = totalRange + 1n;
    const stride = (totalRange > 0xFFFFFFFFFFFFFFFFn) ? (0xC6A4A7935BD1E995n * 0x9E3779B97F4A7C15n) : 0xC6A4A7935BD1E995n;
    const offset = (coprimeStep * (stride | 1n)) % mod;
    return rangeStart + (offset < 0n ? -offset : offset);
}

let hilbertStepVal = 0n;
function hilbertD2XY(order, d) {
    let n = 1 << order;
    let rx, ry, s, t = Number(d % BigInt(n * n));
    let x = 0, y = 0;
    for (s = 1; s < n; s <<= 1) {
        rx = 1 & Math.floor(t / 2);
        ry = 1 & (t ^ rx);
        if (ry === 0) {
            if (rx === 1) {
                x = s - 1 - x;
                y = s - 1 - y;
            }
            let tmp = x; x = y; y = tmp;
        }
        x += s * rx;
        y += s * ry;
    }
    return { x: x, y: y };
}
function generateHilbertKey(rangeStart, rangeEnd) {
    hilbertStepVal++;
    const totalRange = rangeEnd - rangeStart;
    if (totalRange <= 0n) return rangeStart;
    const HILBERT_TOTAL = 65536n;
    const HILBERT_STRIDE = 10007n;
    const cellIndex = (hilbertStepVal * HILBERT_STRIDE) % HILBERT_TOTAL;
    // Hücre sınırları tam bölüşümle hesaplanır (tamsayı yuvarlaması aralığın üst ucunu dışarıda bırakmaz)
    const spanCount = totalRange + 1n;
    const cellStart = rangeStart + (cellIndex * spanCount) / HILBERT_TOTAL;
    const cellSize = (rangeStart + ((cellIndex + 1n) * spanCount) / HILBERT_TOTAL) - cellStart;
    let microOffset = 0n;
    if (cellSize > 1000000n) {
        const r = BigInt(Math.floor(Math.random() * 1000000));
        microOffset = (cellSize * r) / 1000000n;
    } else if (cellSize > 1n) {
        microOffset = BigInt(Math.floor(Math.random() * Number(cellSize)));
    }
    const key = cellStart + microOffset;
    return (key >= rangeStart && key <= rangeEnd) ? key : rangeStart + (key % (totalRange + 1n));
}

let chaosX = 0.141592653589793, chaosY = 0.271828182845904, chaosStep = 0;
function generateDeterministicChaosKey(min, max) {
    if (min >= max) return min;
    const span = max - min;
    // Lojistik harita bit akışı; aralık genişliği + 64 bit taşma payı (modulo yanlılığı ihmal edilebilir)
    const bitCount = Math.min(span.toString(2).length + 64, 320);
    const wordCount = Math.ceil(bitCount / 32);
    let hex = '';
    for (let w = 0; w < wordCount; w++) {
        let u32 = 0;
        for (let b = 0; b < 32; b++) {
            chaosStep++;
            chaosX = 4.0 * chaosX * (1.0 - chaosX);
            if (chaosX <= 0.000001 || chaosX >= 0.999999 || isNaN(chaosX)) {
                chaosX = 0.14159265 + ((chaosStep * 0.031) % 0.8);
            }
            u32 = (u32 << 1) | (chaosX > 0.5 ? 1 : 0);
        }
        hex += (u32 >>> 0).toString(16).padStart(8, '0');
    }
    return min + (BigInt('0x' + hex) % (span + 1n));
}

let weakEntropyCounter = 0;
function generateWeakEntropyKey(rangeStart, rangeEnd, entropyBits) {
    const entBits = Number((entropyBits !== undefined && entropyBits !== null) ? entropyBits : 0);
    if (entBits === -1) return randomBigIntInRange(rangeStart, rangeEnd);
    weakEntropyCounter++;
    const span = rangeEnd - rangeStart;
    if (span <= 0n) return rangeStart;
    const totalBits = BigInt(span.toString(2).length);
    const c = BigInt(weakEntropyCounter);
    if (entBits === 32) {
        let state = (c * 1103515245n + 12345n) & 0xFFFFFFFFn;
        let val = 0n, bits = 0n;
        while (bits < totalBits) {
            state = (state * 1664525n + 1013904223n) & 0xFFFFFFFFn;
            val = (val << 32n) | state;
            bits += 32n;
        }
        return rangeStart + (val % (span + 1n));
    } else if (entBits === 64) {
        let z = (c * 0x9e3779b97f4a7c15n + 1n) & 0xFFFFFFFFFFFFFFFFn;
        let val = 0n, bits = 0n;
        while (bits < totalBits) {
            z = (z + 0x9e3779b97f4a7c15n) & 0xFFFFFFFFFFFFFFFFn;
            let temp = z;
            temp = ((temp ^ (temp >> 30n)) * 0xbf58476d1ce4e5b9n) & 0xFFFFFFFFFFFFFFFFn;
            val = (val << 64n) | ((temp ^ (temp >> 31n)) & 0xFFFFFFFFFFFFFFFFn);
            bits += 64n;
        }
        return rangeStart + (val % (span + 1n));
    }
    const cat = Number(c % 3n);
    if (cat === 0) {
        let state = (c * 1103515245n + 12345n) & 0xFFFFFFFFn;
        let val = 0n, bits = 0n;
        while (bits < totalBits) {
            state = (state * 1664525n + 1013904223n) & 0xFFFFFFFFn;
            val = (val << 32n) | state;
            bits += 32n;
        }
        return rangeStart + (val % (span + 1n));
    } else if (cat === 1) {
        const b1 = (c * 7n) % totalBits;
        const b2 = (c * 13n) % totalBits;
        const val = (1n << b1) | (1n << b2);
        return rangeStart + (val % (span + 1n));
    } else {
        const pat = 'deadbeefcafebabe';
        const hexLen = Math.max(1, Math.ceil(Number(totalBits) / 4));
        const fullHex = pat.repeat(Math.floor(hexLen / pat.length) + 2).slice(0, hexLen);
        const val = BigInt('0x' + fullHex) ^ ((c * 13n) % 1024n);
        return rangeStart + (val % (span + 1n));
    }
}

const KRONECKER_CONSTANTS = [
    0x6A09E667F3BCC908n, 0xBB67AE8584CAA73Bn, 0x3C6EF372FE94F82Bn, 0xA54FF53A5F1D36F1n,
    0x510E527FADE682D1n, 0x9B05688C2B3E6C1Fn, 0x1F83D9ABFB41BD6Bn, 0x5BD0AAFF55490AEFn
];
let kroneckerStep = 1n;
function generateKroneckerKey(rangeStart, rangeEnd) {
    kroneckerStep++;
    const span = rangeEnd - rangeStart;
    if (span <= 0n) return rangeStart;
    const idx = Number(kroneckerStep % 8n);
    const alpha = KRONECKER_CONSTANTS[idx];
    const frac = (kroneckerStep * alpha) & 0xFFFFFFFFFFFFFFFFn;
    return rangeStart + ((frac * (span + 1n)) >> 64n);
}

let grayCodeStep = 1n;
function generateGrayCodeKey(rangeStart, rangeEnd) {
    grayCodeStep++;
    const span = rangeEnd - rangeStart;
    if (span <= 0n) return rangeStart;
    const g = grayCodeStep ^ (grayCodeStep >> 1n);
    const frac = (g * 0x9E3779B97F4A7C15n) & 0xFFFFFFFFFFFFFFFFn;
    return rangeStart + ((frac * (span + 1n)) >> 64n);
}

let rndBuf = null;
let rndIdx = 0;
function randomBigIntInRange(min, max) {
    if (min >= max) return min;
    const span = max - min;
    if (!rndBuf || rndIdx >= rndBuf.length) {
        rndBuf = new Uint8Array(256 * 32);
        const c = (typeof crypto !== 'undefined') ? crypto : (typeof self !== 'undefined' ? self.crypto : null);
        if (c && typeof c.getRandomValues === 'function') c.getRandomValues(rndBuf);
        else for (let k = 0; k < rndBuf.length; k++) rndBuf[k] = Math.floor(Math.random() * 256);
        rndIdx = 0;
    }
    let rand = 0n;
    for (let j = 0; j < 32; j++) rand = (rand << 8n) | BigInt(rndBuf[rndIdx + j]);
    rndIdx += 32;
    return min + (rand % (span + 1n));
}

let riemannT = 0.0;
let quantumPhase = 0.0;
function getNextPhysicsPercent(formula) {
    if (formula === 'DETERMINISTIC_CHAOS' || formula === 'CHAOS') {
        chaosStep++;
        chaosX = 4.0 * chaosX * (1.0 - chaosX);
        if (chaosX <= 0.0001 || chaosX >= 0.9999 || isNaN(chaosX)) chaosX = 0.14159265;
        chaosY = (chaosY < 0.5) ? (2.0 * chaosY) : (2.0 * (1.0 - chaosY));
        if (chaosY <= 0.0001 || chaosY >= 0.9999 || isNaN(chaosY)) chaosY = 0.27182818;
        return Math.max(0.001, Math.min(99.999, (chaosX * 0.618 + chaosY * 0.382) * 100.0));
    }
    if (formula === 'GOLDEN_SINGULARITY' || formula === 'WEYL_GOLDEN') {
        riemannT += 1.0;
        return Math.max(0.001, Math.min(99.999, (((riemannT * 0.618033988749895) % 1.0) * 100.0)));
    }
    if (formula === 'QUANTUM_TUNNEL' || formula === 'QUANTUM') {
        quantumPhase = (quantumPhase + 0.08) % (Math.PI * 2);
        const alpha = (Math.sin(quantumPhase) + 1) / 2;
        return 100.0 * (1 - Math.exp(-3.5 * alpha));
    }
    riemannT += 0.05;
    const s1 = Math.sin(riemannT * Math.log(2));
    const s2 = Math.sin(riemannT * Math.log(3));
    const s3 = Math.sin(riemannT * Math.log(5));
    const val = (s1 + s2 + s3 + 3) / 6;
    return Math.max(0.001, Math.min(99.999, val * 100.0));
}
`;

try { (new Function(GPU_KEY_GENERATORS_SOURCE))(); } catch(e) {}

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
        const baseMin = boundMin;
        const effRange = boundMax - boundMin;
        const sBig = BigInt(Math.round((cfg.sliceStartPct || 0) * 100000000));
        const eBig = BigInt(Math.round((cfg.sliceEndPct || 100) * 100000000));
        boundMin = baseMin + (effRange * sBig / 10000000000n);
        boundMax = baseMin + (effRange * eBig / 10000000000n);
    }
    if (boundMax <= boundMin) boundMax = boundMin + 1n;

    // 2. Fizik ve Kaos Formülleri Alt-Aralık Hesabı ([subMin, subMax])
    let subMin = boundMin;
    let subMax = boundMax;
    const f = cfg.activeFormula;
    const curSpan = (boundMax > boundMin) ? (boundMax - boundMin) : 1n;
    const isNarrowRange = curSpan <= 10000000n;
    const isPrefixMode  = (f === 'PREFIX' || cfg.activeRangeMode === 'PREFIX' || cfg.isPrefixFilterActive);
    const isHexRange    = (cfg.activeRangeMode === 'HEX');

    if (!isNarrowRange && !isPrefixMode && !isHexRange &&
        (f === 'OMNI_CHAOS' || f === 'DETERMINISTIC_CHAOS' || f === 'QUANTUM_TUNNEL' || f === 'RIEMANN_ZETA' || f === 'GOLDEN_SINGULARITY' || cfg.activeRangeMode === 'PHYSICS')) {
        if (typeof getNextPhysicsPercent === 'function') {
            const centerPct = getNextPhysicsPercent(f);
            const microWidth = 0.5;
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
        const autoList = ['RANDOM', 'SOBOL', 'WEYL_GOLDEN', 'COPRIME_STRIDE', 'HILBERT', 'WEAK_ENTROPY', 'CHAOS', 'HALTON', 'SCRAMBLED_SOBOL', 'KRONECKER_LATTICE', 'GRAY_CODE_WALK'];
        if (gpuAutoStepCounter % 25 === 0) {
            gpuAutoIndex = (gpuAutoIndex + 1) % autoList.length;
        }
        activeAlgo = autoList[gpuAutoIndex] || 'RANDOM';
    }
    if (activeAlgo === 'WEAK_ENTROPY' && cfg.selectedEntropyBits === -1) {
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

    // 5. Anahtarları Üret (Seçilen Algoritmaya Göre - 100% Bağımsız ve Hatasız)
    const keys = new Array(size);
    const effSpan = (subMax > subMin) ? (subMax - subMin) : 1n;
    const targetLen = baseEnd.toString(16).length;

    for (let i = 0; i < size; i++) {
        let currentKey;

        if (activeAlgo === 'SOBOL') {
            currentKey = (typeof generateSobolKey === 'function') ? generateSobolKey(subMin, subMax) : randomBigIntInRange(subMin, subMax);
        } else if (activeAlgo === 'SCRAMBLED_SOBOL') {
            currentKey = (typeof generateScrambledSobolKey === 'function') ? generateScrambledSobolKey(subMin, subMax) : randomBigIntInRange(subMin, subMax);
        } else if (activeAlgo === 'HALTON') {
            currentKey = (typeof generateHaltonKey === 'function') ? generateHaltonKey(subMin, subMax) : randomBigIntInRange(subMin, subMax);
        } else if (activeAlgo === 'WEYL_GOLDEN') {
            currentKey = (typeof generateWeylGoldenKey === 'function') ? generateWeylGoldenKey(subMin, subMax) : randomBigIntInRange(subMin, subMax);
        } else if (activeAlgo === 'COPRIME_STRIDE') {
            currentKey = (typeof generateCoprimeStrideKey === 'function') ? generateCoprimeStrideKey(subMin, subMax) : randomBigIntInRange(subMin, subMax);
        } else if (activeAlgo === 'HILBERT') {
            currentKey = (typeof generateHilbertKey === 'function') ? generateHilbertKey(subMin, subMax) : randomBigIntInRange(subMin, subMax);
        } else if (activeAlgo === 'CHAOS') {
            currentKey = (typeof generateDeterministicChaosKey === 'function') ? generateDeterministicChaosKey(subMin, subMax) : randomBigIntInRange(subMin, subMax);
        } else if (activeAlgo === 'WEAK_ENTROPY') {
            currentKey = (typeof generateWeakEntropyKey === 'function') ? generateWeakEntropyKey(subMin, subMax, cfg.selectedEntropyBits) : randomBigIntInRange(subMin, subMax);
        } else if (activeAlgo === 'KRONECKER_LATTICE') {
            currentKey = (typeof generateKroneckerKey === 'function') ? generateKroneckerKey(subMin, subMax) : randomBigIntInRange(subMin, subMax);
        } else if (activeAlgo === 'GRAY_CODE_WALK') {
            currentKey = (typeof generateGrayCodeKey === 'function') ? generateGrayCodeKey(subMin, subMax) : randomBigIntInRange(subMin, subMax);
        } else {
            currentKey = randomBigIntInRange(subMin, subMax);
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
                currentKey = effSpan > 0n ? (boundMin + ((currentKey > boundMin ? (currentKey - boundMin) : (boundMin - currentKey)) % (effSpan + 1n))) : boundMin;
            }
        }

        keys[i] = currentKey;
    }

    let algoPrefix = '';
    if (activeAlgo === 'SOBOL') algoPrefix = '📐 Sobol + ';
    else if (activeAlgo === 'WEYL_GOLDEN') algoPrefix = '🌟 Weyl Kafesi + ';
    else if (activeAlgo === 'COPRIME_STRIDE') algoPrefix = '♾️ Modüler Adım + ';
    else if (activeAlgo === 'HILBERT') algoPrefix = '🌀 Hilbert + ';
    else if (activeAlgo === 'CHAOS') algoPrefix = '♾️ Kaos + ';
    else if (activeAlgo === 'WEAK_ENTROPY') algoPrefix = '⚡ Zayıf Entropi + ';
    else if (activeAlgo === 'HALTON') algoPrefix = '🎲 Halton + ';
    else if (activeAlgo === 'SCRAMBLED_SOBOL') algoPrefix = '🔀 S.Sobol + ';
    else if (activeAlgo === 'KRONECKER_LATTICE') algoPrefix = '🌟 Kronecker Kafesi + ';
    else if (activeAlgo === 'GRAY_CODE_WALK') algoPrefix = '🧬 Gray Kodu (H=1) + ';

    const pParts = [];
    if (cfg.isDualSliderActive && (boundMin > baseStart || boundMax < baseEnd)) {
        pParts.push('🎚️ [0x' + boundMin.toString(16).substring(0, 6) + '..-0x' + boundMax.toString(16).substring(0, 6) + '..]');
    }
    if (isEffectivePrefix && gpuActivePrefix) {
        pParts.push('🧩 Ön Ek: 0x' + gpuActivePrefix);
    }
    const rangeDesc = algoPrefix + (pParts.join(' + ') || 'Standart');

    return {
        keys: keys,
        boundMin: boundMin,
        boundMax: boundMax,
        subMin: subMin,
        subMax: subMax,
        activeAlgo: activeAlgo,
        activePrefix: isEffectivePrefix ? gpuActivePrefix : '',
        rangeDesc: rangeDesc,
        baseStart: baseStart,
        baseEnd: baseEnd
    };
}

// Hedef hash160'ı 5 tamsayı kelimesi olarak önbellekleme (string ayrıştırma yükünü sıfırlar)
let cachedTargetH160Str = '';
let targetW0 = 0, targetW1 = 0, targetW2 = 0, targetW3 = 0, targetW4 = 0;
let hasTargetHash160 = false;
function ensureTargetWords(h160) {
    if (h160 === cachedTargetH160Str) return;
    cachedTargetH160Str = h160;
    if (h160 && h160.length === 40) {
        targetW0 = parseInt(h160.substr(0, 8), 16) | 0;
        targetW1 = parseInt(h160.substr(8, 8), 16) | 0;
        targetW2 = parseInt(h160.substr(16, 8), 16) | 0;
        targetW3 = parseInt(h160.substr(24, 8), 16) | 0;
        targetW4 = parseInt(h160.substr(32, 8), 16) | 0;
        hasTargetHash160 = true;
    } else {
        targetW0 = targetW1 = targetW2 = targetW3 = targetW4 = 0;
        hasTargetHash160 = false;
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

        ensureTargetWords(targetH160);

        // ⚡ GPU computes Pubkeys + SHA-256 + RIPEMD-160 + Target Match in single kernel!
        const gpuRes = await gpuEngine.huntBatchGpu(
            keys,
            hasTargetHash160 ? [targetW0, targetW1, targetW2, targetW3, targetW4] : null
        );

        let found = false;
        if (gpuRes && gpuRes.matched) {
            const keyHex = gpuRes.keyHex || (keys[0] ? keys[0].toString(16).padStart(64, '0') : '');
            console.log('[GPU] 🎯 WIN! key=' + keyHex + ' hash160=' + targetH160);
            handleGpuWin(keyHex, target, targetH160, false);
            gpuHuntRunning = false;
            found = true;
            return;
        }

        // Çoklu hedef / özel havuz / sıkıştırılmamış kontrolü (İkincil Kontrol)
        const isUncompTarget = Boolean(target && (target.isUncompressed || target.id === 'ilave' || target.id === 'TEST_64'));
        const checkUncomp = isUncompTarget || Boolean(cfg.isCustomPoolActive);
        const hasMultiTarget = checkUncomp || Boolean(cfg.isCustomPoolActive) || Boolean(cfg.satoshiMultiTargetActive) || Boolean(cfg.isCircularMode);

        if (hasMultiTarget) {
            const raw = await gpuEngine.computePubkeysRaw(keys);
            const cjs = window.CryptoJS;
            if (cjs && raw) {
                const waWords = [0, 0, 0, 0, 0, 0, 0, 0, 0];
                const wa = { words: waWords, sigBytes: 33 };

                const waUWords = new Array(17);
                const waU = { words: waUWords, sigBytes: 65 };

                const customBloom = (cfg.isCustomPoolActive && typeof customBloomFilter !== 'undefined') ? customBloomFilter : null;

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

                    // 🎯 Sıkıştırılmamış Eşleşme Kontrolü
                    if (checkUncomp) {
                        for (let u = 0; u < 17; u++) waUWords[u] = raw[off + 9 + u];
                        waUWords[16] = waUWords[16] & 0xFF000000;

                        const shaU = cjs.SHA256(waU);
                        const rmdU = cjs.RIPEMD160(shaU);
                        const rwU = rmdU.words;

                        if (hasTargetHash160 && rwU[0] === targetW0 && rwU[1] === targetW1 && rwU[2] === targetW2 && rwU[3] === targetW3 && rwU[4] === targetW4) {
                            const keyHex = keys[i].toString(16).padStart(64, '0');
                            console.log('[GPU] 🎯 UNCOMPRESSED WIN! key=' + keyHex + ' hash160=' + targetH160);
                            handleGpuWin(keyHex, target, targetH160, true);
                            gpuHuntRunning = false; found = true; break;
                        }

                        if (cfg.isCustomPoolActive && typeof customAddressSet !== 'undefined' && customAddressSet && customAddressSet.size > 0) {
                            let bloomHitU = true;
                            if (customBloom && typeof customBloom.hasWords5 === 'function') {
                                bloomHitU = customBloom.hasWords5(rwU[0] >>> 0, rwU[1] >>> 0, rwU[2] >>> 0, rwU[3] >>> 0, rwU[4] >>> 0);
                            }
                            if (bloomHitU) {
                                const h160U = rmdU.toString();
                                if (customAddressSet.has(h160U)) {
                                    const keyHex = keys[i].toString(16).padStart(64, '0');
                                    console.log('[GPU] 🎯 CUSTOM POOL (UNCOMPRESSED) WIN! key=' + keyHex);
                                    handleGpuWin(keyHex, target, h160U, true, '🎯 İlave Havuz Cüzdanı (Uncompressed)');
                                    gpuHuntRunning = false; found = true; break;
                                }
                            }
                        }
                    }

                    // Çoklu hedef / özel havuz kontrolü
                    if (cfg.satoshiMultiTargetActive || cfg.isCustomPoolActive || cfg.isCircularMode) {
                        if (cfg.unsolvedList && cfg.unsolvedList.length > 0) {
                            const h160 = rmd.toString();
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
                        if (cfg.isCustomPoolActive && typeof customAddressSet !== 'undefined' && customAddressSet && customAddressSet.size > 0) {
                            let bloomHit = true;
                            if (customBloom && typeof customBloom.hasWords5 === 'function') {
                                bloomHit = customBloom.hasWords5(rw[0] >>> 0, rw[1] >>> 0, rw[2] >>> 0, rw[3] >>> 0, rw[4] >>> 0);
                            }
                            if (bloomHit) {
                                const h160 = rmd.toString();
                                if (customAddressSet.has(h160)) {
                                    const keyHex = keys[i].toString(16).padStart(64, '0');
                                    console.log('[GPU] 🎯 CUSTOM POOL WIN! key=' + keyHex);
                                    handleGpuWin(keyHex, target, h160, false, '🎯 İlave Havuz Cüzdanı');
                                    gpuHuntRunning = false; found = true; break;
                                }
                            }
                        }
                    }
                }
            }
        }

        gpuTotalKeys += BATCH;

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
        else if (batchInfo.activeAlgo === 'WEYL_GOLDEN') algoPrefix = '🌟 Weyl Kafesi + ';
        else if (batchInfo.activeAlgo === 'COPRIME_STRIDE') algoPrefix = '♾️ Modüler Adım + ';
        else if (batchInfo.activeAlgo === 'HILBERT') algoPrefix = '🌀 Hilbert + ';
        else if (batchInfo.activeAlgo === 'CHAOS') algoPrefix = '♾️ Kaos + ';
        else if (batchInfo.activeAlgo === 'WEAK_ENTROPY') algoPrefix = '⚡ Zayıf Entropi + ';
        else if (batchInfo.activeAlgo === 'KANGAROO') algoPrefix = '🦘 Pollard Kangaroo + ';
        else if (batchInfo.activeAlgo === 'KANGAROO_SOBOL') algoPrefix = '🦘📐 Sobol-Kangaroo + ';
        else if (batchInfo.activeAlgo === 'KANGAROO_GLV') algoPrefix = '⚡🦘 GLV-Kangaroo + ';
        else if (batchInfo.activeAlgo === 'HALTON') algoPrefix = '🎲 Halton + ';
        else if (batchInfo.activeAlgo === 'SCRAMBLED_SOBOL') algoPrefix = '🔀 S.Sobol + ';
        else if (batchInfo.activeAlgo === 'MCMC') algoPrefix = '🎰 MCMC + ';
        else if (batchInfo.activeAlgo === 'WEAK_PRNG_LCG') algoPrefix = '🎲 Zayıf PRNG (LCG) + ';
        else if (batchInfo.activeAlgo === 'KRONECKER_LATTICE') algoPrefix = '🌟 Kronecker Kafesi + ';
        else if (batchInfo.activeAlgo === 'GRAY_CODE_WALK') algoPrefix = '🧬 Gray Kodu (H=1) + ';

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
        const wasRunning = gpuHuntRunning;
        gpuHuntRunning = false;
        // CPU fallback - only if hunt was actively running and not stopped
        if (wasRunning && typeof huntRunning !== 'undefined' && huntRunning && typeof huntBatch === 'function') {
            huntBatch();
        }
    }
}

let gpuThrottleDelay = 35; // Varsayılan: 35ms dinlenme payı (akıcı 60 FPS masaüstü, AMD DWM uyumlu)
let gpuCurrentIntensity = 'BALANCED';
let gpuCurrentStepsPerCall = 16;

function setGpuIntensity(level) {
    if (!level) return;
    const l = String(level).toUpperCase();
    if (l.includes('ECO') || l.includes('LIGHT') || l.includes('LOW')) {
        gpuCurrentIntensity = 'ECO';
        gpuThrottleDelay = 80; // 80ms dinlenme payı: GPU ve CPU %90+ serbest, sıfır kasma/donma
        gpuCurrentStepsPerCall = 8;
        if (gpuEngine) gpuEngine.batchSize = 512;
        if (webGpuKangarooInstance && typeof webGpuKangarooInstance.setStepsPerCall === 'function') {
            webGpuKangarooInstance.setStepsPerCall(8);
        }
        console.log('[GPU] 🟢 Ultra Hafif / Eco moduna geçildi (Batch: 512, Steps: 8, Gecikme: 80ms - Sıfır Kasma)');
    } else if (l.includes('MAX') || l.includes('TURBO') || l.includes('HIGH')) {
        gpuCurrentIntensity = 'MAX';
        gpuThrottleDelay = 15; // 15ms: Windows DWM ve fare imlecinin akıcı kalması için AMD uyumlu nefes payı
        gpuCurrentStepsPerCall = 32;
        if (gpuEngine) gpuEngine.batchSize = 4096;
        if (webGpuKangarooInstance && typeof webGpuKangarooInstance.setStepsPerCall === 'function') {
            webGpuKangarooInstance.setStepsPerCall(32);
        }
        console.log('[GPU] 🔴 Tam Güç moduna geçildi (Batch: 4096, Steps: 32, Gecikme: 15ms)');
    } else {
        // BALANCED (Varsayılan)
        gpuCurrentIntensity = 'BALANCED';
        gpuThrottleDelay = 35; // 35ms: Akıcı 60 FPS masaüstü, AMD WDDM preemption uyumlu
        gpuCurrentStepsPerCall = 16;
        if (gpuEngine) gpuEngine.batchSize = 2048;
        if (webGpuKangarooInstance && typeof webGpuKangarooInstance.setStepsPerCall === 'function') {
            webGpuKangarooInstance.setStepsPerCall(16);
        }
        console.log('[GPU] 🟡 Dengeli moduna geçildi (Batch: 2048, Steps: 16, Gecikme: 35ms - 60 FPS akıcı)');
    }

    if (isGpuWorkerActive && gpuWorker) {
        try {
            gpuWorker.postMessage({
                cmd: 'setIntensity',
                intensity: gpuCurrentIntensity,
                throttleDelay: gpuThrottleDelay,
                stepsPerCall: gpuCurrentStepsPerCall
            });
        } catch(e) {}
    }
}

function handleGpuWin(keyHex, target, matchedH160, isUncompressed, extraNote) {
    if (typeof handleWorkerMessage === 'function') {
        const rewardText = (target.reward || '⚡ GPU') + (extraNote ? (' ' + extraNote) : '');
        handleWorkerMessage({ data: {
            type: 'win',
            workerId: 999,
            keyHex: keyHex,
            targetId: target.id,
            targetAddr: target.addr,
            reward: rewardText,
            matchedH160: matchedH160,
            isUncompressed: isUncompressed,
            matchType: isUncompressed ? 'GPU_UNCOMP' : 'GPU_COMP'
        }});
    }
}

let webGpuKangarooInstance = null;
let gpuKangarooLastTime = 0;
let gpuKangarooAccumSteps = 0;

async function gpuKangarooLoop() {
    if (!gpuHuntRunning || !webGpuKangarooInstance) return;

    try {
        const batchRes = await webGpuKangarooInstance.dispatchBatch();
        const steps = (typeof batchRes === 'object') ? batchRes.steps : Number(batchRes);
        const lastKey = (typeof batchRes === 'object' && batchRes.lastKey) ? batchRes.lastKey : '';
        gpuKangarooAccumSteps += steps;
        gpuTotalKeys += steps;

        const now = performance.now();
        const elapsed = (now - gpuKangarooLastTime) / 1000;
        if (elapsed >= 0.25) {
            gpuKeysPerSec = Math.round(gpuKangarooAccumSteps / (elapsed || 0.001));
            gpuKangarooAccumSteps = 0;
            gpuKangarooLastTime = now;

            const trapCount = webGpuKangarooInstance.tameTraps.size + webGpuKangarooInstance.wildTraps.size;

            if (typeof handleWorkerMessage === 'function') {
                handleWorkerMessage({ data: {
                    type: 'progress',
                    workerId: 999,
                    count: steps,
                    rate: gpuKeysPerSec,
                    trapCount: trapCount,
                    lastKey: lastKey,
                    rangeText: `⚡ WebGPU Kangaroo — 2048 Sürü (${gpuKeysPerSec.toLocaleString()} key/s)`
                }});
            }
        }

        if (webGpuKangarooInstance.isSolved && webGpuKangarooInstance.solvedKey) {
            const keyHex = webGpuKangarooInstance.solvedKey.toString(16).padStart(64, '0');
            const cfg = (typeof getCurrentWorkerConfig === 'function') ? getCurrentWorkerConfig() : null;
            const target = cfg ? cfg.activeTarget : null;
            console.log('🎉 [GPU KANGAROO BINGO!] Hedef Çözüldü:', keyHex);
            handleGpuWin(keyHex, target, target ? target.targetHash160 : '', false);
            gpuHuntRunning = false;
            return;
        }

        if (gpuHuntRunning) {
            setTimeout(gpuKangarooLoop, gpuThrottleDelay);
        }
    } catch(err) {
        console.error('[WebGPU Kangaroo Loop Error]:', err);
        gpuHuntRunning = false;
    }
}

async function startGpuHunting() {
    gpuHuntRunning = true;

    const cfg = (typeof getCurrentWorkerConfig === 'function') ? getCurrentWorkerConfig() : null;
    const algoStr = (cfg && (cfg.selectedAlgorithm || cfg.activeAlgo)) ? String(cfg.selectedAlgorithm || cfg.activeAlgo).toUpperCase() : '';
    const isExplicitNonKangaroo = algoStr && algoStr !== 'AUTO' && !algoStr.startsWith('KANGAROO') && algoStr !== 'GLV_LATTICE_2D';
    const isKangaroo = !isExplicitNonKangaroo && Boolean(algoStr.startsWith('KANGAROO') || algoStr === 'GLV_LATTICE_2D' || (typeof isKangarooModeActive === 'function' && isKangarooModeActive()));

    let targetPub = (typeof kangarooTargetPubHex !== 'undefined' && kangarooTargetPubHex) ? kangarooTargetPubHex : null;
    const target = cfg ? cfg.activeTarget : null;
    if (!targetPub && target && typeof EXPOSED_KANGAROO_PUBKEYS !== 'undefined') {
        targetPub = EXPOSED_KANGAROO_PUBKEYS[target.id] || EXPOSED_KANGAROO_PUBKEYS[parseInt(target.id, 10)] || null;
    }
    if (!targetPub && target) {
        targetPub = target.pubHex || target.pubkeyHex || null;
    }

    if (isKangaroo && !targetPub) {
        console.warn('[GPU] Hedef açık anahtar (pubkey) bulunamadı. Standart GPU moduna geçiliyor.');
        if (typeof showToast === 'function') {
            showToast(`ℹ️ <strong>Pollard Kangaroo Açık Anahtar Gerektirir</strong><br><span style="font-size:11px;">#${target ? target.id : ''} için açık anahtar ifşa edilmediğinden GPU Hızlı Arama (${cfg?.selectedAlgorithm || 'Sobol'}) modunda çalışıyor.</span>`, 4500);
        }
    }

    const effectiveKangaroo = isKangaroo && Boolean(targetPub);

    if (isGpuWorkerActive && gpuWorker) {
        console.log(`[GPU] ⚡ İzole Web Worker üzerinden GPU avı başlatılıyor... (Kangaroo=${effectiveKangaroo}, Intensity=${gpuCurrentIntensity}, Delay=${gpuThrottleDelay}ms, Steps=${gpuCurrentStepsPerCall})`);

        const customAddrs = (cfg && cfg.isCustomPoolActive && typeof customAddressSet !== 'undefined' && customAddressSet)
            ? Array.from(customAddressSet)
            : [];

        gpuWorker.postMessage({
            cmd: 'start',
            config: cfg,
            intensity: gpuCurrentIntensity,
            throttleDelay: gpuThrottleDelay,
            stepsPerCall: gpuCurrentStepsPerCall,
            isKangaroo: effectiveKangaroo,
            targetPubHex: targetPub,
            customAddresses: customAddrs
        });
        return true;
    }

    if (!gpuEngine || !gpuEngine.isInitialized) {
        console.warn('[GPU] Engine hazır değil');
        return false;
    }

    if (effectiveKangaroo && typeof WebGpuKangarooEngine !== 'undefined') {
        if (targetPub) {
            console.log('[GPU] 🦘 Pollard Kangaroo WebGPU Çekirdeği Başlatılıyor... Hedef Açık Anahtar:', targetPub);
            let s = (typeof toBigIntSafe === 'function') ? toBigIntSafe(target.start) : BigInt('0x' + target.start);
            let e = (typeof toBigIntSafe === 'function') ? toBigIntSafe(target.end) : BigInt('0x' + target.end);

            // 🎚️ İkili Kaydıraç (Uzay Dilimleme / Kesme) aktifse Kanguru aralığını bu daraltılmış dilime kilitle
            if (cfg && cfg.isDualSliderActive && cfg.sliderMinHex && cfg.sliderMaxHex) {
                const sMin = (typeof toBigIntSafe === 'function') ? toBigIntSafe(cfg.sliderMinHex) : BigInt('0x' + cfg.sliderMinHex);
                const sMax = (typeof toBigIntSafe === 'function') ? toBigIntSafe(cfg.sliderMaxHex) : BigInt('0x' + cfg.sliderMaxHex);
                if (sMin >= s && sMax <= e && sMax > sMin) {
                    s = sMin;
                    e = sMax;
                    console.log(`[GPU Kangaroo] ✂️ Uzay Kesildi! Dilim: 0x${s.toString(16)} — 0x${e.toString(16)}`);
                }
            }

            gpuKangarooLastTime = performance.now();
            gpuKangarooAccumSteps = 0;

            webGpuKangarooInstance = new WebGpuKangarooEngine();
            if (typeof gpuCurrentStepsPerCall !== 'undefined') {
                webGpuKangarooInstance.stepsPerCall = gpuCurrentStepsPerCall;
            }
            const ok = await webGpuKangarooInstance.init(gpuEngine.device, targetPub, s, e);
            if (ok) {
                if (typeof gpuCurrentStepsPerCall !== 'undefined' && typeof webGpuKangarooInstance.setStepsPerCall === 'function') {
                    webGpuKangarooInstance.setStepsPerCall(gpuCurrentStepsPerCall);
                }
                console.log(`[GPU] 🦘 WebGPU Kangaroo Döngüsü Aktif! (stepsPerCall=${webGpuKangarooInstance.stepsPerCall}, delay=${gpuThrottleDelay}ms)`);
                gpuKangarooLoop();
                return true;
            } else {
                console.warn('[GPU] WebGpuKangarooEngine başlatılamadı, standart GPU aramasına dönülüyor.');
                webGpuKangarooInstance = null;
            }
        } else {
            console.warn('[GPU] Hedef açık anahtar (pubkey) bulunamadı. Standart GPU moduna geçiliyor.');
            if (typeof showToast === 'function') {
                showToast(`ℹ️ <strong>Pollard Kangaroo Açık Anahtar Gerektirir</strong><br><span style="font-size:11px;">#${target ? target.id : ''} için açık anahtar ifşa edilmediğinden GPU Hızlı Arama (Sobol/Kaos) modunda çalışıyor.</span>`, 4500);
            }
        }
    }

    console.log('[GPU] ⚡ Arama başladı — batch=' + gpuEngine.batchSize + ', throttleDelay=' + gpuThrottleDelay + 'ms, adapter=' + gpuEngine.adapterInfo);
    gpuHuntBatch();
    return true;
}

function stopGpuHunting(immediate) {
    gpuHuntRunning = false;
    if (isGpuWorkerActive && gpuWorker) {
        try {
            gpuWorker.postMessage({ cmd: 'stop', immediate: !!immediate });
        } catch(e) {}
    }
    if (webGpuKangarooInstance) {
        webGpuKangarooInstance = null;
    }
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
Object.defineProperty(window, 'isGpuWorkerActive', {
    get() { return isGpuWorkerActive; },
    configurable: true
});
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
Object.defineProperty(window, 'gpuCurrentStepsPerCall', {
    get() { return gpuCurrentStepsPerCall; },
    set(v) { gpuCurrentStepsPerCall = v; },
    configurable: true
});
