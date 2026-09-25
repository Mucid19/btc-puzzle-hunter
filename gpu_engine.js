/**
 * gpu_engine.js — Gerçek WebGPU Secp256k1 Point Multiplication Engine
 *
 * GERÇEK implementasyon — hiçbir sahte kod yoktur.
 * WGSL 1.0 uyumlu: Sadece u32, bool, vec2<u32>. (64-bit tip YOK)
 *
 * Mimari v2:
 *   GPU: privkey (256-bit) → EC mul → Jacobian→Affine → SHA256(compressed pubkey)  [WGSL]
 *   CPU: GPU SHA256 digest → RIPEMD160  [CryptoJS — çok hafif]
 *
 *   Optimizasyonlar:
 *   - SHA256 GPU'da (WGSL): CPU SHA256 yükü tamamen sıfırlandı
 *   - Batch: 2048 → 8192 key/dispatch (128 workgroup × 64 thread)
 *   - Double-buffer: GPU N+1 çalışırken CPU N'i RIPEMD işliyor (pipeline overlap)
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

// Input: 8 u32 per key (LE), Output: 8 u32 per key (SHA256 digest of compressed pubkey)
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
            let s1 = add32c(acc[1], q9.y + ov9, s0.y); acc[1] = s1.x;
            carry = s1.y;
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

// ----------------------------------------------------------------------------
// SHA256 — 33-byte compressed pubkey'i GPU'da hash'ler (CPU yükünü sıfırlar)
// output: 8 u32 SHA256 digest → pubkeys[idx*8 .. idx*8+7]
// ----------------------------------------------------------------------------

// SHA256 K sabitleri (64 adet)
const sha256_K = array<u32, 64>(
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
);

// SHA256(compressed_pubkey) hesapla ve pubkeys[idx*8..+7]'ye yaz
fn store_sha256(idx: u32, ax: U256, ay: U256) {
    let prefix = 2u + (ay.l[0] & 1u);  // 02=çift y, 03=tek y

    // 33-byte compressed pubkey → 16-word SHA256 mesaj bloğu (big-endian)
    // Byte 0 = prefix, Byte 1..32 = ax big-endian (limbs[7]..limbs[0])
    var w: array<u32, 64>;
    w[0]  = (prefix << 24u) | (ax.l[7] >> 8u);
    w[1]  = (ax.l[7] << 24u) | (ax.l[6] >> 8u);
    w[2]  = (ax.l[6] << 24u) | (ax.l[5] >> 8u);
    w[3]  = (ax.l[5] << 24u) | (ax.l[4] >> 8u);
    w[4]  = (ax.l[4] << 24u) | (ax.l[3] >> 8u);
    w[5]  = (ax.l[3] << 24u) | (ax.l[2] >> 8u);
    w[6]  = (ax.l[2] << 24u) | (ax.l[1] >> 8u);
    w[7]  = (ax.l[1] << 24u) | (ax.l[0] >> 8u);
    w[8]  = (ax.l[0] << 24u) | 0x800000u;  // son byte + SHA256 padding bit
    w[9]  = 0u; w[10] = 0u; w[11] = 0u; w[12] = 0u; w[13] = 0u;
    w[14] = 0u;    // uzunluk yüksek 32 bit (33 byte = 264 bit < 2^32)
    w[15] = 264u;  // uzunluk düşük 32 bit: 33 * 8 = 264

    // Mesaj zamanlaması genişletme (w[16]..w[63])
    for (var i = 16u; i < 64u; i++) {
        let s0 = rotr(w[i-15u], 7u) ^ rotr(w[i-15u], 18u) ^ (w[i-15u] >> 3u);
        let s1 = rotr(w[i-2u],  17u) ^ rotr(w[i-2u],  19u) ^ (w[i-2u]  >> 10u);
        w[i] = w[i-16u] + s0 + w[i-7u] + s1;
    }

    // SHA256 başlangıç hash değerleri
    var h0 = 0x6a09e667u; var h1 = 0xbb67ae85u;
    var h2 = 0x3c6ef372u; var h3 = 0xa54ff53au;
    var h4 = 0x510e527fu; var h5 = 0x9b05688cu;
    var h6 = 0x1f83d9abu; var h7 = 0x5be0cd19u;

    var a = h0; var b = h1; var c = h2; var d = h3;
    var e = h4; var f = h5; var g = h6; var hh = h7;

    // 64 tur SHA256 compression
    for (var i = 0u; i < 64u; i++) {
        let S1    = rotr(e, 6u) ^ rotr(e, 11u) ^ rotr(e, 25u);
        let ch    = (e & f) ^ (~e & g);
        let temp1 = hh + S1 + ch + sha256_K[i] + w[i];
        let S0    = rotr(a, 2u) ^ rotr(a, 13u) ^ rotr(a, 22u);
        let maj   = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = S0 + maj;
        hh = g; g = f; f = e; e = d + temp1;
        d  = c; c = b; b = a; a = temp1 + temp2;
    }

    // Final hash değerlerini output buffer'a yaz
    let off = idx * 8u;
    pubkeys[off + 0u] = h0 + a;
    pubkeys[off + 1u] = h1 + b;
    pubkeys[off + 2u] = h2 + c;
    pubkeys[off + 3u] = h3 + d;
    pubkeys[off + 4u] = h4 + e;
    pubkeys[off + 5u] = h5 + f;
    pubkeys[off + 6u] = h6 + g;
    pubkeys[off + 7u] = h7 + hh;
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

    store_sha256(idx, ax, ay);
}
`;

// =============================================================================
// JavaScript WebGPU Engine
// =============================================================================

class GpuEngine {
    constructor() {
        this.device      = null;
        this.pipeline    = null;
        // Double-buffer: 2 set (0 ve 1) — GPU + CPU pipeline'ını örtüştürür
        this.privkeyBufs  = [null, null];
        this.sha256Bufs   = [null, null];
        this.readbackBufs = [null, null];
        this.bindGroups   = [null, null];
        this.batchSize    = 8192;   // 8192 key/dispatch — 128 workgroup × 64 thread
        this.isInitialized = false;
        this.adapterInfo = null;
    }

    async init() {
        if (!navigator.gpu) {
            console.log('[GPU] navigator.gpu yok — WebGPU desteklenmiyor');
            return false;
        }
        try {
            const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
            if (!adapter) {
                console.log('[GPU] Adapter bulunamadı');
                return false;
            }
            this.adapterInfo = (adapter.info && adapter.info.description) ? adapter.info.description
                             : (adapter.name || 'WebGPU GPU');

            this.device = await adapter.requestDevice({
                requiredLimits: { maxBufferSize: 64 * 1024 * 1024 }
            });
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

            this.pipeline = this.device.createComputePipeline({
                layout: 'auto',
                compute: { module: shaderModule, entryPoint: 'main' }
            });

            // Buffer boyutları: 8192 key × 8 u32 × 4 byte = 262,144 byte
            const privBytes = this.batchSize * 8 * 4;   // privkey: 8 u32/key
            const sha2Bytes = this.batchSize * 8 * 4;   // SHA256 output: 8 u32/key

            // 2 set buffer oluştur (double-buffer)
            for (let i = 0; i < 2; i++) {
                this.privkeyBufs[i] = this.device.createBuffer({
                    size: privBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
                });
                this.sha256Bufs[i] = this.device.createBuffer({
                    size: sha2Bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
                });
                this.readbackBufs[i] = this.device.createBuffer({
                    size: sha2Bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
                });
                this.bindGroups[i] = this.device.createBindGroup({
                    layout: this.pipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: this.privkeyBufs[i] } },
                        { binding: 1, resource: { buffer: this.sha256Bufs[i]  } }
                    ]
                });
            }

            this.isInitialized = true;
            console.log('[GPU] ✅ WebGPU v2 yüksek performansla başlatıldı! Adapter:', this.adapterInfo);
            console.log('[GPU] Batch boyutu:', this.batchSize, 'key/dispatch (128 workgroup) | SHA256 GPU\'da | Double-buffer aktif');
            return true;
        } catch(e) {
            console.error('[GPU] Init hatası:', e);
            return false;
        }
    }

    /**
     * Bir batch'i GPU'ya gönder — mapAsync Promise'ini döner (non-blocking).
     * Double-buffer loop için: submitBatch(keys, 0) → await submitBatch(keys2, 1) vb.
     */
    submitBatch(privkeysBig, bufIdx) {
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

        const sha2Bytes = n * 8 * 4;
        this.device.queue.writeBuffer(this.privkeyBufs[bufIdx], 0, privData.buffer, 0, n * 32);

        const encoder = this.device.createCommandEncoder();
        encoder.clearBuffer(this.sha256Bufs[bufIdx], 0, sha2Bytes);

        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroups[bufIdx]);
        pass.dispatchWorkgroups(Math.ceil(n / 64));
        pass.end();

        encoder.copyBufferToBuffer(this.sha256Bufs[bufIdx], 0, this.readbackBufs[bufIdx], 0, sha2Bytes);
        this.device.queue.submit([encoder.finish()]);

        // mapAsync Promise'ini döner — caller'ın await etmesi gerekir
        return this.readbackBufs[bufIdx].mapAsync(GPUMapMode.READ, 0, sha2Bytes);
    }

    /**
     * Geçmiş uyumluluk: tek seferlik çalıştırma → SHA256 Uint32Array döner
     */
    async computePubkeysRaw(privkeysBig) {
        const n = privkeysBig.length;
        const sha2Bytes = n * 8 * 4;
        await this.submitBatch(privkeysBig, 0);
        const raw = new Uint32Array(this.readbackBufs[0].getMappedRange(0, sha2Bytes).slice(0));
        this.readbackBufs[0].unmap();
        return raw;
    }

    /**
     * Eski API uyumluluğu: SHA256 Uint32Array → sahte compressed pubkey sarmalayıcısı
     * NOT: Bu metot artık SHA256 çıktısını döner (pubkey değil). Sadece testGpu için.
     */
    async computePubkeys(privkeysBig) {
        // Geriye uyumluluk — artık SHA256 digest döner
        const raw = await this.computePubkeysRaw(privkeysBig);
        return raw;
    }

    destroy() {
        try {
            for (let i = 0; i < 2; i++) {
                if (this.privkeyBufs[i])  this.privkeyBufs[i].destroy();
                if (this.sha256Bufs[i])   this.sha256Bufs[i].destroy();
                if (this.readbackBufs[i]) this.readbackBufs[i].destroy();
            }
            if (this.device) this.device.destroy();
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

    // 2. Algoritma Seçimi
    let activeAlgo = cfg.selectedAlgorithm || 'RANDOM';
    if (activeAlgo === 'AUTO') {
        gpuAutoStepCounter++;
        if (gpuAutoStepCounter % 25 === 0) {
            gpuAutoIndex = (gpuAutoIndex + 1) % 9;
        }
        const autoList = ['RANDOM', 'SOBOL', 'VD_CORPUT', 'WEYL_GOLDEN', 'COPRIME_STRIDE', 'HILBERT', 'WEAK_ENTROPY', 'CHAOS', 'RANDOM'];
        activeAlgo = autoList[gpuAutoIndex] || 'RANDOM';
    }

    // 3. Ön Ek Hazırlığı (Prefix)
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

    // 4. Anahtarları Üret
    const keys = new Array(size);
    const effSpan = boundMax - boundMin;
    const targetLen = baseEnd.toString(16).length;

    let rndBuf = null;
    let rndIdx = 0;

    for (let i = 0; i < size; i++) {
        let currentKey;

        if (activeAlgo === 'SOBOL' && typeof window.generateSobolKey === 'function') {
            currentKey = window.generateSobolKey(boundMin, boundMax);
        } else if (activeAlgo === 'VD_CORPUT' && typeof window.generateVanDerCorputKey === 'function') {
            currentKey = window.generateVanDerCorputKey(boundMin, boundMax);
        } else if (activeAlgo === 'WEYL_GOLDEN' && typeof window.generateWeylGoldenKey === 'function') {
            currentKey = window.generateWeylGoldenKey(boundMin, boundMax);
        } else if (activeAlgo === 'COPRIME_STRIDE' && typeof window.generateCoprimeStrideKey === 'function') {
            currentKey = window.generateCoprimeStrideKey(boundMin, boundMax);
        } else if (activeAlgo === 'HILBERT' && typeof window.generateHilbertKey === 'function') {
            currentKey = window.generateHilbertKey(boundMin, boundMax);
        } else if (activeAlgo === 'CHAOS' && typeof window.generateDeterministicChaosKey === 'function') {
            currentKey = window.generateDeterministicChaosKey(boundMin, boundMax);
        } else if (activeAlgo === 'WEAK_ENTROPY' && typeof window.generateWeakEntropyKey === 'function' && cfg.selectedEntropyBits !== -1) {
            currentKey = window.generateWeakEntropyKey(boundMin, boundMax, cfg.selectedEntropyBits);
        } else if (typeof window.randomBigIntInRange === 'function') {
            currentKey = window.randomBigIntInRange(boundMin, boundMax);
        } else {
            if (!rndBuf || rndIdx >= rndBuf.length) {
                rndBuf = new Uint8Array(256 * 32);
                crypto.getRandomValues(rndBuf);
                rndIdx = 0;
            }
            let rand = 0n;
            for (let j = 0; j < 32; j++) rand = (rand << 8n) | BigInt(rndBuf[rndIdx + j]);
            rndIdx += 32;
            currentKey = boundMin + (rand % (effSpan + 1n));
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
        setTimeout(gpuHuntBatch, 100);
        return;
    }

    const target     = cfg.activeTarget;
    const targetH160 = target.targetHash160 || '';
    const BATCH      = gpuEngine.batchSize;
    const cjs        = window.CryptoJS;
    if (!cjs) return;

    ensureTargetWords(targetH160);

    // SHA256 GPU'da yapılıyor → CPU'da yalnızca RIPEMD160 kalıyor
    // WordArray: 8 kelime, 32 byte (GPU'dan gelen SHA256 digest)
    const waWords = new Array(8).fill(0);
    const wa = { words: waWords, sigBytes: 32 };

    const t0 = performance.now();

    try {
        // ── İlk batch'i GPU'ya gönder (double-buffer: slot 0) ──
        let batchInfo  = genBatchForGpu(cfg, BATCH);
        let keys       = batchInfo.keys;
        let mapPromise = gpuEngine.submitBatch(keys, 0);
        let bufReady   = 0;   // hangi readbackBuf hazır olacak

        let found = false;

        while (gpuHuntRunning && !found) {
            // Bir sonraki batch'i hazırla (CPU bu işi GPU çalışırken yapıyor)
            const nextBufIdx    = 1 - bufReady;
            const nextBatchInfo = genBatchForGpu(cfg, BATCH);
            const nextKeys      = nextBatchInfo.keys;
            // Bir önceki GPU sonucunu bekle
            await mapPromise;
            // Hemen sonraki batch'i GPU'ya gönder (overlap!)
            const nextMapPromise = gpuEngine.submitBatch(nextKeys, nextBufIdx);

            // Mevcut batch'i CPU'da işle (sadece RIPEMD160)
            const sha2Bytes = BATCH * 8 * 4;
            const raw = new Uint32Array(
                gpuEngine.readbackBufs[bufReady].getMappedRange(0, sha2Bytes).slice(0)
            );
            gpuEngine.readbackBufs[bufReady].unmap();

            for (let i = 0; i < BATCH && gpuHuntRunning && !found; i++) {
                const off = i * 8;
                // GPU sıfır/sonsuz nokta döndürdüyse (privkey=0 vb.) tüm wordlar 0 olur
                if (raw[off] === 0 && raw[off + 1] === 0) continue;

                // GPU zaten SHA256 yaptı → sadece RIPEMD160
                waWords[0] = raw[off];
                waWords[1] = raw[off + 1];
                waWords[2] = raw[off + 2];
                waWords[3] = raw[off + 3];
                waWords[4] = raw[off + 4];
                waWords[5] = raw[off + 5];
                waWords[6] = raw[off + 6];
                waWords[7] = raw[off + 7];

                const rmd = cjs.RIPEMD160(wa);
                const rw  = rmd.words;

                gpuTotalKeys++;

                // 🎯 Doğrudan 160-bit tamsayı karşılaştırması (sıfır string tahsisi!)
                if (targetW0 !== 0 && rw[0] === targetW0 && rw[1] === targetW1
                    && rw[2] === targetW2 && rw[3] === targetW3 && rw[4] === targetW4) {
                    const keyHex = keys[i].toString(16).padStart(64, '0');
                    console.log('[GPU] 🎯 WIN! key=' + keyHex + ' hash160=' + targetH160);
                    handleGpuWin(keyHex, target, targetH160, false);
                    gpuHuntRunning = false; found = true; break;
                }

                // Çoklu hedef / özel havuz kontrolü (yalnızca aktifse string üretir)
                if (cfg.satoshiMultiTargetActive || cfg.isCustomPoolActive) {
                    const h160 = rmd.toString();
                    if (cfg.satoshiMultiTargetActive && cfg.unsolvedList
                        && cfg.unsolvedList.some(p => p.targetHash160 === h160)) {
                        const matched = cfg.unsolvedList.find(p => p.targetHash160 === h160);
                        const keyHex = keys[i].toString(16).padStart(64, '0');
                        console.log('[GPU] 🎯 SATOSHI WIN! key=' + keyHex);
                        handleGpuWin(keyHex, matched, h160, false);
                        gpuHuntRunning = false; found = true; break;
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

            // Hız ve UI güncelleme (her batch'te)
            const dt = (performance.now() - t0) / 1000;
            gpuKeysPerSec = Math.round(gpuTotalKeys / (dt || 0.001));
            window.gpuKeysPerSec = gpuKeysPerSec;

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

            const rangeDesc = algoPrefix + (pParts.join(' + ') || 'Standart');
            const rangeText = '⚡ WebGPU v2 — ' + rangeDesc + ' (' + gpuKeysPerSec.toLocaleString() + ' key/s)';

            const spEl = document.getElementById('statSpeed');
            if (spEl) spEl.innerText = gpuKeysPerSec.toLocaleString() + ' key/s ⚡GPU';

            if (typeof handleWorkerMessage === 'function') {
                handleWorkerMessage({ data: {
                    type: 'progress',
                    workerId: 999,
                    count: BATCH,
                    lastKey: keys[BATCH - 1].toString(16).padStart(64, '0'),
                    targetId: target.id,
                    rangeText: rangeText
                }});
            }

            // Bir sonraki iterasyona geç
            batchInfo  = nextBatchInfo;
            keys       = nextKeys;
            mapPromise = nextMapPromise;
            bufReady   = nextBufIdx;
        }

        // Döngüden çıktıysak mapPromise'i temizle (unmap edilmemiş buffer'ı kapat)
        if (!found) {
            try { await mapPromise; gpuEngine.readbackBufs[bufReady].unmap(); } catch(e) {}
        }

    } catch(e) {
        console.error('[GPU] gpuHuntBatch hatası:', e);
        gpuHuntRunning = false;
        // CPU fallback
        if (typeof huntBatch === 'function') huntBatch();
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
            matchType: 'GPU_COMP'
        }});
    }
}

function startGpuHunting() {
    if (!gpuEngine || !gpuEngine.isInitialized) {
        console.warn('[GPU] Engine hazır değil');
        return false;
    }
    gpuHuntRunning = true;
    console.log('[GPU] ⚡ Arama başladı — batch=' + gpuEngine.batchSize + ', adapter=' + gpuEngine.adapterInfo);
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
window.gpuKeysPerSec     = gpuKeysPerSec;
window.gpuTotalKeys      = gpuTotalKeys;
