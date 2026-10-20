/**
 * gpu_worker.js — İzole WebGPU Arama Çekirdeği (Dedicated Worker)
 * 
 * Mimari:
 *   - WebGPU compute shader'ları ana UI thread'inden tamamen izole ayrı bir işletim sistemi thread'inde çalışır.
 *   - Tarayıcı arayüzü (UI) ve Windows DWM asla kilitlenmez, takılmaz veya "Sayfa Yanıt Vermiyor" uyarısı vermez.
 *   - SHA256 ve RIPEMD160 hash hesaplamaları arka plan worker'ında sıfır gecikmeyle (micro-sleep olmadan) tam güç çalışır.
 *   - Pollard's Kangaroo ve Standart (Sobol, Kaos, Hilbert, Weyl vb.) modları %100 destekler.
 *   - Hem dosya tabanlı (HTTP) hem Blob URL tabanlı (100% Çevrimdışı / Local) çalışır.
 */

// 1. Kütüphane Yükleme (Blob Worker'da core-libs önceden yüklü gelir; bağımsız dosyada CDN fallback)
if (typeof CryptoJS === 'undefined' || typeof elliptic === 'undefined') {
    try {
        importScripts(
            'https://cdnjs.cloudflare.com/ajax/libs/crypto-js/4.1.1/crypto-js.min.js',
            'https://cdnjs.cloudflare.com/ajax/libs/crypto-js/4.1.1/ripemd160.min.js',
            'https://cdnjs.cloudflare.com/ajax/libs/elliptic/6.5.4/elliptic.min.js'
        );
    } catch(e) {
        // Çevrimdışı ortamda Blob URL üzerinden core-libs enjekte edilir.
    }
}

// 1.1 Çevrimdışı CryptoJS Fallback (SHA-256 & RIPEMD-160)
(function() {
    if (typeof CryptoJS !== 'undefined' && CryptoJS.SHA256 && CryptoJS.RIPEMD160) return;

    const SHA_K = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    ];

    function sha256Words(inWords, sigBytes) {
        const bitLen = sigBytes * 8;
        const totalWords = (((sigBytes + 8) >> 6) + 1) * 16;
        const w = new Uint32Array(totalWords);
        for (let i = 0; i < inWords.length && i < totalWords; i++) w[i] = inWords[i] >>> 0;
        const byteIndex = sigBytes;
        w[byteIndex >> 2] |= (0x80 << (24 - (byteIndex & 3) * 8)) >>> 0;
        w[totalWords - 1] = bitLen >>> 0;
        w[totalWords - 2] = Math.floor(bitLen / 0x100000000) >>> 0;

        let H0 = 0x6a09e667, H1 = 0xbb67ae85, H2 = 0x3c6ef372, H3 = 0xa54ff53a;
        let H4 = 0x510e527f, H5 = 0x9b05688c, H6 = 0x1f83d9ab, H7 = 0x5bd0aaff;
        const W = new Uint32Array(64);

        for (let chunk = 0; chunk < totalWords; chunk += 16) {
            for (let i = 0; i < 16; i++) W[i] = w[chunk + i];
            for (let i = 16; i < 64; i++) {
                const s0 = ((W[i-15] >>> 7) | (W[i-15] << 25)) ^ ((W[i-15] >>> 18) | (W[i-15] << 14)) ^ (W[i-15] >>> 3);
                const s1 = ((W[i-2] >>> 17) | (W[i-2] << 15)) ^ ((W[i-2] >>> 19) | (W[i-2] << 13)) ^ (W[i-2] >>> 10);
                W[i] = (W[i-16] + s0 + W[i-7] + s1) >>> 0;
            }

            let a = H0, b = H1, c = H2, d = H3, e = H4, f = H5, g = H6, h = H7;
            for (let i = 0; i < 64; i++) {
                const S1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
                const ch = (e & f) ^ (~e & g);
                const temp1 = (h + S1 + ch + SHA_K[i] + W[i]) >>> 0;
                const S0 = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
                const maj = (a & b) ^ (a & c) ^ (b & c);
                const temp2 = (S0 + maj) >>> 0;

                h = g; g = f; f = e; e = (d + temp1) >>> 0;
                d = c; c = b; b = a; a = (temp1 + temp2) >>> 0;
            }

            H0 = (H0 + a) >>> 0; H1 = (H1 + b) >>> 0; H2 = (H2 + c) >>> 0; H3 = (H3 + d) >>> 0;
            H4 = (H4 + e) >>> 0; H5 = (H5 + f) >>> 0; H6 = (H6 + g) >>> 0; H7 = (H7 + h) >>> 0;
        }

        return [H0, H1, H2, H3, H4, H5, H6, H7];
    }

    function ripemd160Words(inWords, sigBytes) {
        const byteLen = sigBytes;
        const totalWords = (((byteLen + 8) >> 6) + 1) * 16;
        const x = new Uint32Array(totalWords);
        for (let i = 0; i < inWords.length && (i * 4) < byteLen; i++) {
            const w = inWords[i];
            x[i] = ((w >>> 24) & 0xFF) | (((w >>> 16) & 0xFF) << 8) | (((w >>> 8) & 0xFF) << 16) | ((w & 0xFF) << 24);
        }
        const bytePos = byteLen;
        x[bytePos >> 2] |= (0x80 << ((bytePos & 3) * 8)) >>> 0;
        const bitLen = byteLen * 8;
        x[totalWords - 2] = bitLen >>> 0;
        x[totalWords - 1] = Math.floor(bitLen / 0x100000000) >>> 0;

        let h0 = 0x67452301, h1 = 0xefcdab89, h2 = 0x98badcfe, h3 = 0x10325476, h4 = 0xc3d2e1f0;

        const RL = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
            3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
            1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
            4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13
        ];
        const RR = [
            5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
            6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
            15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
            8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
            12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11
        ];
        const SL = [
            11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
            7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
            11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
            11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
            9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6
        ];
        const SR = [
            8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
            9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
            9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
            15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
            8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11
        ];

        function rotl(v, n) { return ((v << n) | (v >>> (32 - n))) >>> 0; }
        function f1(x, y, z) { return (x ^ y ^ z) >>> 0; }
        function f2(x, y, z) { return ((x & y) | (~x & z)) >>> 0; }
        function f3(x, y, z) { return ((x | ~y) ^ z) >>> 0; }
        function f4(x, y, z) { return ((x & z) | (y & ~z)) >>> 0; }
        function f5(x, y, z) { return (x ^ (y | ~z)) >>> 0; }

        for (let chunk = 0; chunk < totalWords; chunk += 16) {
            let al = h0, bl = h1, cl = h2, dl = h3, el = h4;
            let ar = h0, br = h1, cr = h2, dr = h3, er = h4;

            for (let i = 0; i < 80; i++) {
                let fl, kl;
                if (i < 16)      { fl = f1(bl, cl, dl); kl = 0x00000000; }
                else if (i < 32) { fl = f2(bl, cl, dl); kl = 0x5a827999; }
                else if (i < 48) { fl = f3(bl, cl, dl); kl = 0x6ed9eba1; }
                else if (i < 64) { fl = f4(bl, cl, dl); kl = 0x8f1bbcdc; }
                else             { fl = f5(bl, cl, dl); kl = 0xa953fd4e; }

                let tl = (al + fl + x[chunk + RL[i]] + kl) >>> 0;
                tl = (rotl(tl, SL[i]) + el) >>> 0;
                al = el; el = dl; dl = rotl(cl, 10); cl = bl; bl = tl;

                let fr, kr;
                if (i < 16)      { fr = f5(br, cr, dr); kr = 0x50a28be6; }
                else if (i < 32) { fr = f4(br, cr, dr); kr = 0x5c4dd124; }
                else if (i < 48) { fr = f3(br, cr, dr); kr = 0x6d703ef3; }
                else if (i < 64) { fr = f2(br, cr, dr); kr = 0x7a6d76e9; }
                else             { fr = f1(br, cr, dr); kr = 0x00000000; }

                let tr = (ar + fr + x[chunk + RR[i]] + kr) >>> 0;
                tr = (rotl(tr, SR[i]) + er) >>> 0;
                ar = er; er = dr; dr = rotl(cr, 10); cr = br; br = tr;
            }

            const t = (h1 + cl + dr) >>> 0;
            h1 = (h2 + dl + er) >>> 0;
            h2 = (h3 + el + ar) >>> 0;
            h3 = (h4 + al + br) >>> 0;
            h4 = (h0 + bl + cr) >>> 0;
            h0 = t;
        }

        function swapLE(w) {
            return (((w & 0xFF) << 24) | (((w >>> 8) & 0xFF) << 16) | (((w >>> 16) & 0xFF) << 8) | ((w >>> 24) & 0xFF)) >>> 0;
        }
        return [swapLE(h0), swapLE(h1), swapLE(h2), swapLE(h3), swapLE(h4)];
    }

    const fallbackCryptoJS = {
        SHA256: function(wa) {
            const inW = wa.words || [];
            const sig = wa.sigBytes !== undefined ? wa.sigBytes : inW.length * 4;
            const res = sha256Words(inW, sig);
            return {
                words: res,
                sigBytes: 32,
                toString: function() {
                    let hex = '';
                    for (let i = 0; i < 8; i++) hex += (res[i] >>> 0).toString(16).padStart(8, '0');
                    return hex;
                }
            };
        },
        RIPEMD160: function(wa) {
            const inW = wa.words || [];
            const sig = wa.sigBytes !== undefined ? wa.sigBytes : inW.length * 4;
            const res = ripemd160Words(inW, sig);
            return {
                words: res,
                sigBytes: 20,
                toString: function() {
                    let hex = '';
                    for (let i = 0; i < 5; i++) hex += (res[i] >>> 0).toString(16).padStart(8, '0');
                    return hex;
                }
            };
        }
    };

    if (typeof self !== 'undefined') self.CryptoJS = fallbackCryptoJS;
})();

const SECP256K1_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141n;

// =============================================================================
// WGSL SHADER 1: Secp256k1 Point Multiplication (Standart Mod)
// =============================================================================
const GPU_WGSL = `
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

@group(0) @binding(0) var<storage, read>       privkeys:     array<u32>;
@group(0) @binding(1) var<storage, read_write> pubkeys:      array<u32>;
@group(0) @binding(2) var<uniform>             params:       GpuParams;
@group(0) @binding(3) var<storage, read_write> match_result: array<u32>;

fn mul32(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;
    let ll = a_lo * b_lo;
    let lh = a_lo * b_hi;
    let hl = a_hi * b_lo;
    let hh = a_hi * b_hi;
    let mid = (ll >> 16u) + (lh & 0xFFFFu) + (hl & 0xFFFFu);
    let lo  = (ll & 0xFFFFu) | (mid << 16u);
    let hi  = hh + (lh >> 16u) + (hl >> 16u) + (mid >> 16u);
    return vec2<u32>(lo, hi);
}

fn u256_add(a: U256, b: U256) -> U256 {
    var out: U256;
    var carry: u32 = 0u;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let sum = a.l[i] + b.l[i] + carry;
        carry = select(0u, 1u, (sum < a.l[i]) || (carry == 1u && sum == a.l[i]));
        out.l[i] = sum;
    }
    return out;
}

fn u256_sub(a: U256, b: U256) -> U256 {
    var out: U256;
    var borrow: u32 = 0u;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let diff = a.l[i] - b.l[i] - borrow;
        borrow = select(0u, 1u, (a.l[i] < b.l[i]) || (borrow == 1u && a.l[i] == b.l[i]));
        out.l[i] = diff;
    }
    return out;
}

fn u256_gte(a: U256, b: U256) -> bool {
    for (var i: i32 = 7; i >= 0; i = i - 1) {
        if (a.l[u32(i)] > b.l[u32(i)]) { return true; }
        if (a.l[u32(i)] < b.l[u32(i)]) { return false; }
    }
    return true;
}

fn secp256k1_p() -> U256 {
    var p: U256;
    p.l[0] = 0xFFFFFC2Fu;
    p.l[1] = 0xFFFFFFFEu;
    p.l[2] = 0xFFFFFFFFu;
    p.l[3] = 0xFFFFFFFFu;
    p.l[4] = 0xFFFFFFFFu;
    p.l[5] = 0xFFFFFFFFu;
    p.l[6] = 0xFFFFFFFFu;
    p.l[7] = 0xFFFFFFFFu;
    return p;
}

fn add32c(a: u32, b: u32, c: u32) -> vec2<u32> {
    let s1 = a + b;
    let c1 = select(0u, 1u, s1 < a);
    let s2 = s1 + c;
    let c2 = select(0u, 1u, s2 < s1);
    return vec2<u32>(s2, c1 + c2);
}

fn u256_lt(a: U256, b: U256) -> bool {
    for (var i=7u; i>=1u; i--) {
        if(a.l[i] < b.l[i]) { return true; }
        if(a.l[i] > b.l[i]) { return false; }
    }
    return a.l[0] < b.l[0];
}

fn u256_sub_raw(a: U256, b: U256) -> U256 {
    var r: U256;
    var borrow: u32 = 0u;
    for (var i=0u; i<8u; i++) {
        let s = a.l[i] - b.l[i] - borrow;
        let nb_cond1 = a.l[i] < b.l[i];
        let nb_cond2 = (borrow == 1u) && (a.l[i] == b.l[i]);
        borrow = select(0u, 1u, nb_cond1 || nb_cond2);
        r.l[i] = s;
    }
    return r;
}

fn fp_add(a: U256, b: U256) -> U256 {
    let p = secp256k1_p();
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
    if (u256_lt(a, b)) { return u256_sub_raw(secp256k1_p(), u256_sub_raw(b, a)); }
    return u256_sub_raw(a, b);
}

fn u256_mul_full(a: U256, b: U256) -> U512 {
    var t: array<u32, 16>;
    for (var k=0u; k<16u; k++) { t[k]=0u; }

    for (var i=0u; i<8u; i++) {
        for (var j=0u; j<8u; j++) {
            let k = i+j;
            let prod = mul32(a.l[i], b.l[j]);
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

fn u512_mod_p(t: U512) -> U256 {
    let h = t.hi;
    let p = secp256k1_p();

    var acc: array<u32, 10>;
    for (var i = 0u; i < 10u; i++) { acc[i] = 0u; }
    for (var i = 0u; i < 8u; i++) { acc[i] = t.lo.l[i]; }

    var carry: u32 = 0u;
    for (var i = 0u; i < 8u; i++) {
        let prod = mul32(h.l[i], 977u);
        let s = add32c(acc[i], prod.x, carry);
        acc[i] = s.x;
        carry = prod.y + s.y;
    }
    let tmp8 = add32c(acc[8], carry, 0u); acc[8] = tmp8.x; acc[9] = tmp8.y;

    carry = 0u;
    for (var i = 0u; i < 8u; i++) {
        let s = add32c(acc[i + 1u], h.l[i], carry);
        acc[i + 1u] = s.x;
        carry = s.y;
    }
    acc[9] = acc[9] + carry;

    let ov8 = acc[8]; let ov9 = acc[9];
    acc[8] = 0u; acc[9] = 0u;

    if (ov8 != 0u || ov9 != 0u) {
        let q8 = mul32(ov8, 977u);
        let r0 = add32c(acc[0], q8.x, 0u); acc[0] = r0.x;
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
fn fp_sqr(a: U256) -> U256 { return fp_mul(a, a); }

fn fp_inv(a: U256) -> U256 {
    var out: U256;
    out.l[0] = 1u;
    for (var k: u32 = 1u; k < 8u; k = k + 1u) { out.l[k] = 0u; }
    let p = secp256k1_p();
    let exp = u256_sub(p, U256(array<u32, 8>(2u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)));
    var base = a;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let w = exp.l[i];
        for (var b: u32 = 0u; b < 32u; b = b + 1u) {
            if (((w >> b) & 1u) == 1u) { out = fp_mul(out, base); }
            base = fp_sqr(base);
        }
    }
    return out;
}

fn secp256k1_g() -> JacobianPoint {
    var pt: JacobianPoint;
    pt.x.l[0] = 0x16F81798u; pt.x.l[1] = 0x59F2815Bu;
    pt.x.l[2] = 0x2DCE28D9u; pt.x.l[3] = 0x029BFCDBu;
    pt.x.l[4] = 0xCE870B07u; pt.x.l[5] = 0x55A06295u;
    pt.x.l[6] = 0xF9DCBBACu; pt.x.l[7] = 0x79BE667Eu;
    pt.y.l[0] = 0xFB10D4B8u; pt.y.l[1] = 0x9C47D08Fu;
    pt.y.l[2] = 0xA6855419u; pt.y.l[3] = 0xFD17B448u;
    pt.y.l[4] = 0x0E1108A8u; pt.y.l[5] = 0x5DA4FBFCu;
    pt.y.l[6] = 0x26A3C465u; pt.y.l[7] = 0x483ADA77u;
    pt.z.l[0] = 1u;
    for (var k: u32 = 1u; k < 8u; k = k + 1u) { pt.z.l[k] = 0u; }
    pt.is_inf = 0u;
    return pt;
}

fn point_double(p: JacobianPoint) -> JacobianPoint {
    if (p.is_inf == 1u) { return p; }
    let a = fp_sqr(p.x);
    let b = fp_sqr(p.y);
    let c = fp_sqr(b);
    let t0 = fp_add(p.x, b);
    let t1 = fp_sqr(t0);
    let t2 = fp_sub(t1, a);
    let d  = fp_mul(fp_sub(t2, c), U256(array<u32, 8>(2u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)));
    let e  = fp_mul(a, U256(array<u32, 8>(3u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)));
    let f  = fp_sqr(e);
    let x3 = fp_sub(f, fp_mul(d, U256(array<u32, 8>(2u, 0u, 0u, 0u, 0u, 0u, 0u, 0u))));
    let y3 = fp_sub(fp_mul(e, fp_sub(d, x3)), fp_mul(c, U256(array<u32, 8>(8u, 0u, 0u, 0u, 0u, 0u, 0u, 0u))));
    let z3 = fp_mul(fp_mul(p.y, p.z), U256(array<u32, 8>(2u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)));
    return JacobianPoint(x3, y3, z3, 0u);
}

fn point_add_mixed(p: JacobianPoint, gx: U256, gy: U256) -> JacobianPoint {
    if (p.is_inf == 1u) {
        var z1: U256; z1.l[0] = 1u;
        for (var k: u32 = 1u; k < 8u; k = k + 1u) { z1.l[k] = 0u; }
        return JacobianPoint(gx, gy, z1, 0u);
    }
    let zsq = fp_sqr(p.z);
    let u2  = fp_mul(gx, zsq);
    let s2  = fp_mul(gy, fp_mul(p.z, zsq));
    let h   = fp_sub(u2, p.x);
    let r   = fp_sub(s2, p.y);
    let hsq = fp_sqr(h);
    let hcu = fp_mul(h, hsq);
    let v   = fp_mul(p.x, hsq);
    let x3  = fp_sub(fp_sub(fp_sqr(r), hcu), fp_mul(v, U256(array<u32, 8>(2u, 0u, 0u, 0u, 0u, 0u, 0u, 0u))));
    let y3  = fp_sub(fp_mul(r, fp_sub(v, x3)), fp_mul(p.y, hcu));
    let z3  = fp_mul(p.z, h);
    return JacobianPoint(x3, y3, z3, 0u);
}

struct AffinePoint {
    x: U256,
    y: U256,
}

fn jacobian_to_affine(p: JacobianPoint) -> AffinePoint {
    let zinv  = fp_inv(p.z);
    let zinv2 = fp_sqr(zinv);
    let zinv3 = fp_mul(zinv, zinv2);
    let ax    = fp_mul(p.x, zinv2);
    let ay    = fp_mul(p.y, zinv3);
    return AffinePoint(ax, ay);
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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.batch_size) { return; }

    var privk: U256;
    let in_base = idx * 8u;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        privk.l[k] = privkeys[in_base + k];
    }

    var r: JacobianPoint;
    r.is_inf = 1u;
    r.x.l[0] = 0u; r.y.l[0] = 0u; r.z.l[0] = 0u;
    for (var k: u32 = 1u; k < 8u; k = k + 1u) {
        r.x.l[k] = 0u; r.y.l[k] = 0u; r.z.l[k] = 0u;
    }

    let G = secp256k1_g();
    var started = false;

    for (var wi: i32 = 7; wi >= 0; wi = wi - 1) {
        let w = privk.l[u32(wi)];
        for (var bi: i32 = 31; bi >= 0; bi = bi - 1) {
            if (started) { r = point_double(r); }
            if (((w >> u32(bi)) & 1u) == 1u) {
                if (!started) {
                    r = G;
                    started = true;
                } else {
                    r = point_add_mixed(r, G.x, G.y);
                }
            }
        }
    }

    let aff = jacobian_to_affine(r);
    let ax  = aff.x;
    let ay  = aff.y;

    let parity: u32 = select(2u, 3u, (ay.l[0] & 1u) == 1u);

    let out_base = idx * 26u;
    pubkeys[out_base + 0u] = (parity << 24u) | (ax.l[7] >> 8u);
    pubkeys[out_base + 1u] = (ax.l[7] << 24u) | (ax.l[6] >> 8u);
    pubkeys[out_base + 2u] = (ax.l[6] << 24u) | (ax.l[5] >> 8u);
    pubkeys[out_base + 3u] = (ax.l[5] << 24u) | (ax.l[4] >> 8u);
    pubkeys[out_base + 4u] = (ax.l[4] << 24u) | (ax.l[3] >> 8u);
    pubkeys[out_base + 5u] = (ax.l[3] << 24u) | (ax.l[2] >> 8u);
    pubkeys[out_base + 6u] = (ax.l[2] << 24u) | (ax.l[1] >> 8u);
    pubkeys[out_base + 7u] = (ax.l[1] << 24u) | (ax.l[0] >> 8u);
    pubkeys[out_base + 8u] = ax.l[0] << 24u;

    pubkeys[out_base + 9u]  = (0x04u << 24u) | (ax.l[7] >> 8u);
    pubkeys[out_base + 10u] = (ax.l[7] << 24u) | (ax.l[6] >> 8u);
    pubkeys[out_base + 11u] = (ax.l[6] << 24u) | (ax.l[5] >> 8u);
    pubkeys[out_base + 12u] = (ax.l[5] << 24u) | (ax.l[4] >> 8u);
    pubkeys[out_base + 13u] = (ax.l[4] << 24u) | (ax.l[3] >> 8u);
    pubkeys[out_base + 14u] = (ax.l[3] << 24u) | (ax.l[2] >> 8u);
    pubkeys[out_base + 15u] = (ax.l[2] << 24u) | (ax.l[1] >> 8u);
    pubkeys[out_base + 16u] = (ax.l[1] << 24u) | (ax.l[0] >> 8u);
    pubkeys[out_base + 17u] = (ax.l[0] << 24u) | (ay.l[7] >> 8u);
    pubkeys[out_base + 18u] = (ay.l[7] << 24u) | (ay.l[6] >> 8u);
    pubkeys[out_base + 19u] = (ay.l[6] << 24u) | (ay.l[5] >> 8u);
    pubkeys[out_base + 20u] = (ay.l[5] << 24u) | (ay.l[4] >> 8u);
    pubkeys[out_base + 21u] = (ay.l[4] << 24u) | (ay.l[3] >> 8u);
    pubkeys[out_base + 22u] = (ay.l[3] << 24u) | (ay.l[2] >> 8u);
    pubkeys[out_base + 23u] = (ay.l[2] << 24u) | (ay.l[1] >> 8u);
    pubkeys[out_base + 24u] = (ay.l[1] << 24u) | (ay.l[0] >> 8u);
    pubkeys[out_base + 25u] = ay.l[0] << 24u;

    let prefix = parity;

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
                match_result[2u + wi] = privk.l[wi];
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
            match_result[16u + wi] = privk.l[wi];
        }
    }
}
`;

// =============================================================================
// WGSL SHADER 2: Pollard's Kangaroo Engine (%100 GPU Leaping + Negation Map)
// =============================================================================
const WGSL_KANGAROO_SHADER = `
struct U256 { l: array<u32, 8>, }
struct KangarooState { x: U256, y: U256, dist: U256, herd: u32, id: u32, _pad0: u32, _pad1: u32, }
struct JumpPoint { x: U256, y: U256, dist: U256, }
struct KangarooParams {
    num_kangaroos: u32, steps_per_call: u32, dp_mask_lo: u32, dp_mask_hi: u32,
    max_traps: u32, negation_mode: u32, _pad1: u32, _pad2: u32,
}
struct TrapRecord {
    x: U256, dist: U256, herd: u32, parity: u32, kangaroo_id: u32, negation_flag: u32,
}
struct TrapHeader { count: atomic<u32>, overflow: u32, _pad0: u32, _pad1: u32, }
struct AddResult { val: U256, carry: u32, }
struct AffinePoint { x: U256, y: U256, }

@group(0) @binding(0) var<storage, read_write> states: array<KangarooState>;
@group(0) @binding(1) var<storage, read>       jumps:  array<JumpPoint>;
@group(0) @binding(2) var<uniform>             params: KangarooParams;
@group(0) @binding(3) var<storage, read_write> trap_header:  TrapHeader;
@group(0) @binding(4) var<storage, read_write> trap_records: array<TrapRecord>;

fn secp256k1_p() -> U256 {
    var p: U256;
    p.l[0] = 0xFFFFFC2Fu; p.l[1] = 0xFFFFFFFEu; p.l[2] = 0xFFFFFFFFu; p.l[3] = 0xFFFFFFFFu;
    p.l[4] = 0xFFFFFFFFu; p.l[5] = 0xFFFFFFFFu; p.l[6] = 0xFFFFFFFFu; p.l[7] = 0xFFFFFFFFu;
    return p;
}

fn secp256k1_n() -> U256 {
    var n: U256;
    n.l[0] = 0xD0364141u; n.l[1] = 0xBFD25E8Cu; n.l[2] = 0xAF48A03Bu; n.l[3] = 0xEBAAEDCEu;
    n.l[4] = 0xFFFFFFFEu; n.l[5] = 0xFFFFFFFFu; n.l[6] = 0xFFFFFFFFu; n.l[7] = 0xFFFFFFFFu;
    return n;
}

fn mul32(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xFFFFu; let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu; let b_hi = b >> 16u;
    let ll = a_lo * b_lo; let lh = a_lo * b_hi;
    let hl = a_hi * b_lo; let hh = a_hi * b_hi;
    let mid = (ll >> 16u) + (lh & 0xFFFFu) + (hl & 0xFFFFu);
    let lo  = (ll & 0xFFFFu) | (mid << 16u);
    let hi  = hh + (lh >> 16u) + (hl >> 16u) + (mid >> 16u);
    return vec2<u32>(lo, hi);
}

fn add32c(a: u32, b: u32, carry_in: u32) -> vec2<u32> {
    let sum = a + b + carry_in;
    let carry_out = select(0u, 1u, (sum < a) || (carry_in != 0u && sum == a));
    return vec2<u32>(sum, carry_out);
}

fn u256_add(a: U256, b: U256) -> U256 {
    var out: U256; var carry: u32 = 0u;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let r = add32c(a.l[i], b.l[i], carry);
        out.l[i] = r.x;
        carry = r.y;
    }
    return out;
}

fn u256_add_c(a: U256, b: U256) -> AddResult {
    var out: U256; var carry: u32 = 0u;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let r = add32c(a.l[i], b.l[i], carry);
        out.l[i] = r.x;
        carry = r.y;
    }
    return AddResult(out, carry);
}

fn u256_sub(a: U256, b: U256) -> U256 {
    var out: U256; var borrow: u32 = 0u;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let diff = a.l[i] - b.l[i] - borrow;
        borrow = select(0u, 1u, (a.l[i] < b.l[i]) || (borrow == 1u && a.l[i] == b.l[i]));
        out.l[i] = diff;
    }
    return out;
}

fn u256_gte(a: U256, b: U256) -> bool {
    for (var i: i32 = 7; i >= 0; i = i - 1) {
        if (a.l[u32(i)] > b.l[u32(i)]) { return true; }
        if (a.l[u32(i)] < b.l[u32(i)]) { return false; }
    }
    return true;
}



// --- DÜZELTİLMİŞ alan aritmetiği (secp256k1 p = 2^256 - 2^32 - 977): gerçek gölgelendirici çalıştırmasıyla doğrulandı ---
fn fx_add32c(a: u32, b: u32, c: u32) -> vec2<u32> {
    let s1 = a + b;
    let c1 = select(0u, 1u, s1 < a);
    let s2 = s1 + c;
    let c2 = select(0u, 1u, s2 < s1);
    return vec2<u32>(s2, c1 + c2);
}

fn fx_add_raw(a: U256, b: U256) -> U256 {
    var r: U256;
    var c: u32 = 0u;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let t = fx_add32c(a.l[i], b.l[i], c);
        r.l[i] = t.x;
        c = t.y;
    }
    return r;
}

fn fx_mul16(a: U256, b: U256) -> array<u32, 16> {
    var t: array<u32, 16>;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) { t[k] = 0u; }
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        var carry: u32 = 0u;
        for (var j: u32 = 0u; j < 8u; j = j + 1u) {
            let pr = mul32(a.l[i], b.l[j]);
            let s1 = fx_add32c(t[i + j], pr.x, carry);
            t[i + j] = s1.x;
            carry = pr.y + s1.y;
        }
        t[i + 8u] = carry;
    }
    return t;
}

fn fp_add(a: U256, b: U256) -> U256 {
    let p = secp256k1_p();
    var r: U256;
    var c: u32 = 0u;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let t = fx_add32c(a.l[i], b.l[i], c);
        r.l[i] = t.x;
        c = t.y;
    }
    if (c != 0u || u256_gte(r, p)) { return u256_sub(r, p); }
    return r;
}

fn fp_mul(a: U256, b: U256) -> U256 {
    let p = secp256k1_p();
    let t = fx_mul16(a, b);
    var s: U256;
    var h: U256;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { s.l[k] = t[k]; h.l[k] = t[k + 8u]; }
    var cc: U256;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { cc.l[k] = 0u; }
    cc.l[0] = 977u;
    cc.l[1] = 1u;
    for (var it: u32 = 0u; it < 4u; it = it + 1u) {
        var hz = true;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { if (h.l[k] != 0u) { hz = false; } }
        if (hz) { break; }
        let pr = fx_mul16(h, cc);
        var pl: U256;
        var ph: U256;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { pl.l[k] = pr[k]; ph.l[k] = pr[k + 8u]; }
        let ns = fx_add_raw(s, pl);
        var carry: u32 = 0u;
        if (!u256_gte(ns, s)) { carry = 1u; }
        var cv: U256;
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { cv.l[k] = 0u; }
        cv.l[0] = carry;
        h = fx_add_raw(ph, cv);
        s = ns;
    }
    if (u256_gte(s, p)) { s = u256_sub(s, p); }
    if (u256_gte(s, p)) { s = u256_sub(s, p); }
    return s;
}

fn fp_sub(a: U256, b: U256) -> U256 {
    let p = secp256k1_p();
    if (u256_gte(a, b)) { return u256_sub(a, b); }
    return u256_sub(u256_add(a, p), b);
}

fn u256_mul_full(a: U256, b: U256) -> array<u32, 16> {
    var r: array<u32, 16>;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) { r[k] = 0u; }
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        var carry: u32 = 0u;
        for (var j: u32 = 0u; j < 8u; j = j + 1u) {
            let prod = mul32(a.l[i], b.l[j]);
            let s0 = r[i + j] + prod.x + carry;
            var c0: u32 = 0u;
            if (s0 < r[i + j] || (prod.x > 0u && s0 < prod.x)) { c0 = c0 + 1u; }
            if (carry > 0u && s0 < carry) { c0 = c0 + 1u; }
            let s1 = prod.y + c0;
            r[i + j] = s0;
            carry = s1;
        }
        r[i + 8u] = r[i + 8u] + carry;
    }
    return r;
}

fn u512_mod_p(r: array<u32, 16>) -> U256 {
    let p = secp256k1_p();
    var acc: U256;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) { acc.l[k] = r[k]; }

    var extra: u32 = 0u;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let hi_word = r[8u + i];
        if (hi_word == 0u) { continue; }
        let prod = mul32(hi_word, 977u);

        var c: u32 = 0u;
        let r0 = add32c(acc.l[i], prod.x, c);
        acc.l[i] = r0.x;
        c = r0.y;

        let r1 = add32c(acc.l[i + 1u], prod.y + hi_word, c);
        acc.l[i + 1u] = r1.x;
        c = r1.y;

        for (var j: u32 = i + 2u; j < 8u; j = j + 1u) {
            if (c == 0u) { break; }
            let rj = add32c(acc.l[j], 0u, c);
            acc.l[j] = rj.x;
            c = rj.y;
        }
        extra = extra + c;
    }

    if (extra > 0u) {
        let prod_ex = mul32(extra, 977u);
        var c2: u32 = 0u;
        let r0 = add32c(acc.l[0], prod_ex.x, c2);
        acc.l[0] = r0.x;
        c2 = r0.y;
        let r1 = add32c(acc.l[1], prod_ex.y + extra, c2);
        acc.l[1] = r1.x;
        c2 = r1.y;
        for (var j: u32 = 2u; j < 8u; j = j + 1u) {
            if (c2 == 0u) { break; }
            let rj = add32c(acc.l[j], 0u, c2);
            acc.l[j] = rj.x;
            c2 = rj.y;
        }
    }

    for (var iter: u32 = 0u; iter < 4u; iter = iter + 1u) {
        if (u256_gte(acc, p)) { acc = u256_sub(acc, p); }
    }
    return acc;
}


fn fp_sqr(a: U256) -> U256 { return fp_mul(a, a); }

fn fp_inv(a: U256) -> U256 {
    var out: U256;
    out.l[0] = 1u;
    for (var k: u32 = 1u; k < 8u; k = k + 1u) { out.l[k] = 0u; }
    let p = secp256k1_p();
    let exp = u256_sub(p, U256(array<u32, 8>(2u, 0u, 0u, 0u, 0u, 0u, 0u, 0u)));
    var base = a;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let w = exp.l[i];
        for (var b: u32 = 0u; b < 32u; b = b + 1u) {
            if (((w >> b) & 1u) == 1u) { out = fp_mul(out, base); }
            base = fp_sqr(base);
        }
    }
    return out;
}

fn point_add_affine(px: U256, py: U256, qx: U256, qy: U256) -> AffinePoint {
    let dx = fp_sub(qx, px);
    let dy = fp_sub(qy, py);
    let lambda = fp_mul(dy, fp_inv(dx));
    let x3 = fp_sub(fp_sub(fp_sqr(lambda), px), qx);
    let y3 = fp_sub(fp_mul(lambda, fp_sub(px, x3)), py);
    return AffinePoint(x3, y3);
}

fn dist_add(d: U256, j: U256) -> U256 {
    let n = secp256k1_n();
    let sum = u256_add(d, j);
    if (u256_gte(sum, n)) { return u256_sub(sum, n); }
    return sum;
}

fn is_distinguished(x: U256, mask_lo: u32, mask_hi: u32) -> bool {
    let lo_ok = ((x.l[0] >> 5u) & mask_lo) == 0u; // alt 5 bit sıçrama indeksi içindir
    let hi_ok = (mask_hi == 0u) || ((x.l[1] & mask_hi) == 0u);
    return lo_ok && hi_ok;
}

@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.num_kangaroos) { return; }

    var st = states[idx];
    let mask_lo = params.dp_mask_lo;
    let mask_hi = params.dp_mask_hi;
    let max_t   = params.max_traps;
    let steps   = params.steps_per_call;
    let use_neg = false; // negasyon: dist/işaret izlenmeden değişmezi bozduğu için kapalı

    for (var s: u32 = 0u; s < steps; s = s + 1u) {
        let jump_idx = st.x.l[0] & 31u;
        let jp = jumps[jump_idx];

        let pt = point_add_affine(st.x, st.y, jp.x, jp.y);
        st.x = pt.x;
        st.y = pt.y;
        st.dist = dist_add(st.dist, jp.dist);

        if (use_neg) {
            if ((st.y.l[0] & 1u) == 1u) {
                let p = secp256k1_p();
                st.y = fp_sub(p, st.y);
            }
        }

        if (is_distinguished(st.x, mask_lo, mask_hi)) {
            let slot = atomicAdd(&trap_header.count, 1u);
            if (slot < max_t) {
                var tr: TrapRecord;
                tr.x = st.x;
                tr.dist = st.dist;
                tr.herd = st.herd;
                tr.parity = st.y.l[0] & 1u;
                tr.kangaroo_id = st.id;
                tr.negation_flag = select(0u, 1u, use_neg);
                trap_records[slot] = tr;
            }
        }
    }
    states[idx] = st;
}
`;

// =============================================================================
// JS CLASS: WebGpuKangarooEngine
// =============================================================================
class WebGpuKangarooEngine {
    constructor() {
        this.device = null;
        this.pipeline = null;
        this.bindGroup = null;
        this.stateBuf = null;
        this.jumpsBuf = null;
        this.paramsBuf = null;
        this.trapHeaderBuf = null;
        this.trapRecordsBuf = null;
        this.trapReadbackBuf = null;

        this.numKangaroos = 2048;
        this.stepsPerCall = 64;
        this.maxTraps = 1024;
        this.span = 0n;
        this.dpMaskLo = 0x0000FFFF;
        this.dpMaskHi = 0;
        this.useNegationMap = false;
        this.tameTraps = new Map();
        this.wildTraps = new Map();
        this.isSolved = false;
        this.solvedKey = null;
        this.ec = null;
        this.targetPoint = null;
    }

    setStepsPerCall(steps) {
        const s = Math.max(8, Math.min(Number(steps) || 64, 256));
        this.stepsPerCall = s;
        if (this.paramsBuf && this.device) {
            const paramsData = new Uint32Array([
                this.numKangaroos, this.stepsPerCall, this.dpMaskLo, this.dpMaskHi,
                this.maxTraps, this.useNegationMap ? 1 : 0, 0, 0
            ]);
            try { this.device.queue.writeBuffer(this.paramsBuf, 0, paramsData); } catch(e) {}
        }
    }

    bigIntToU256Words(big) {
        const words = new Uint32Array(8);
        let temp = big;
        for (let i = 0; i < 8; i++) {
            words[i] = Number(temp & 0xFFFFFFFFn);
            temp >>= 32n;
        }
        return words;
    }

    u256WordsToBigInt(words, offset = 0) {
        let res = 0n;
        for (let i = 7; i >= 0; i--) {
            res = (res << 32n) | BigInt(words[offset + i] >>> 0);
        }
        return res;
    }

    bigIntSqrt(value) {
        if (value <= 0n) return 0n;
        let x0 = value / 2n;
        if (x0 === 0n) return 1n;
        let x1 = (x0 + value / x0) / 2n;
        while (x1 < x0) {
            x0 = x1;
            x1 = (x0 + value / x0) / 2n;
        }
        return x0;
    }

    async init(device, targetPubHex, rangeMin, rangeMax) {
        this.device = device;
        this.targetPubHex = targetPubHex.trim().replace(/^0x/i, '');
        this.rangeMin = typeof rangeMin === 'bigint' ? rangeMin : BigInt('0x' + rangeMin.toString().replace(/^0x/i, ''));
        this.rangeMax = typeof rangeMax === 'bigint' ? rangeMax : BigInt('0x' + rangeMax.toString().replace(/^0x/i, ''));
        this.span = this.rangeMax > this.rangeMin ? (this.rangeMax - this.rangeMin) : 1n;
        this.lastKey = this.rangeMax.toString(16);

        const el = (typeof self !== 'undefined' && self.elliptic) ? self.elliptic : (typeof elliptic !== 'undefined' ? elliptic : null);
        if (!el || !el.ec) throw new Error('[WebGPU Worker Kangaroo] elliptic kütüphanesi bulunamadı');
        this.ec = new el.ec('secp256k1');
        const keyObj = this.ec.keyFromPublic(this.targetPubHex, 'hex');
        this.targetPoint = keyObj.getPublic();

        const sqrtSpan = this.bigIntSqrt(this.span);
        const meanJump = (sqrtSpan / 2n) > 0n ? (sqrtSpan / 2n) : 1n;
        const jumpsData = new Uint32Array(32 * 24);
        let seed = 0x12345678n;

        for (let i = 0; i < 32; i++) {
            seed = (seed * 6364136223846793005n + 1442695040888963407n) & 0xFFFFFFFFFFFFFFFFn;
            const rnd = seed % 2000n;
            let jVal = (meanJump * (100n + rnd)) / 1000n;
            if (jVal <= 0n) jVal = 1n;

            const pt = this.ec.g.mul(jVal.toString(16));
            const xWords = this.bigIntToU256Words(BigInt('0x' + pt.getX().toString(16)));
            const yWords = this.bigIntToU256Words(BigInt('0x' + pt.getY().toString(16)));
            const dWords = this.bigIntToU256Words(jVal);

            const base = i * 24;
            jumpsData.set(xWords, base);
            jumpsData.set(yWords, base + 8);
            jumpsData.set(dWords, base + 16);
        }

        const sqrtBits = sqrtSpan.toString(2).length;
        const dpBits = Math.max(10, Math.min(22, Math.floor(sqrtBits / 2)));
        if (dpBits <= 32) {
            this.dpMaskLo = (1 << dpBits) - 1;
            this.dpMaskHi = 0;
        } else {
            this.dpMaskLo = 0xFFFFFFFF;
            this.dpMaskHi = (1 << (dpBits - 32)) - 1;
        }

        const STATE_WORDS = 28;
        const stateWords = new Uint32Array(this.numKangaroos * STATE_WORDS);
        const half = this.numKangaroos / 2;

        for (let i = 0; i < this.numKangaroos; i++) {
            const isWild = i >= half;
            const kIdx = BigInt(i);
            const shift = (meanJump * (kIdx + 1n)) % (this.span > 0n ? this.span : 1n);

            let pt = null;
            let dist = 0n;

            if (!isWild) {
                dist = (this.rangeMax > shift) ? (this.rangeMax - shift) : this.rangeMax;
                pt = this.ec.g.mul(dist.toString(16));
            } else {
                dist = shift;
                const shiftPt = this.ec.g.mul(shift.toString(16));
                pt = this.targetPoint.add(shiftPt);
            }

            const xWords = this.bigIntToU256Words(BigInt('0x' + pt.getX().toString(16)));
            const yWords = this.bigIntToU256Words(BigInt('0x' + pt.getY().toString(16)));
            const dWords = this.bigIntToU256Words(dist);

            const b = i * STATE_WORDS;
            stateWords.set(xWords, b);
            stateWords.set(yWords, b + 8);
            stateWords.set(dWords, b + 16);
            stateWords[b + 24] = isWild ? 1 : 0;
            stateWords[b + 25] = i;
            stateWords[b + 26] = 0;
            stateWords[b + 27] = 0;
        }

        this.stateBuf = this.device.createBuffer({
            size: stateWords.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.stateBuf, 0, stateWords);

        this.jumpsBuf = this.device.createBuffer({
            size: jumpsData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.jumpsBuf, 0, jumpsData);

        const paramsData = new Uint32Array([
            this.numKangaroos, this.stepsPerCall, this.dpMaskLo, this.dpMaskHi,
            this.maxTraps, this.useNegationMap ? 1 : 0, 0, 0
        ]);
        this.paramsBuf = this.device.createBuffer({
            size: paramsData.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.paramsBuf, 0, paramsData);

        const headerBytes = 16;
        const recordsBytes = this.maxTraps * 80;

        this.trapHeaderBuf = this.device.createBuffer({
            size: headerBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        this.trapRecordsBuf = this.device.createBuffer({
            size: recordsBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        this.trapReadbackBuf = this.device.createBuffer({
            size: headerBytes + recordsBytes,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        const module = this.device.createShaderModule({ code: WGSL_KANGAROO_SHADER });
        this.pipeline = await this.device.createComputePipelineAsync({
            layout: 'auto',
            compute: { module: module, entryPoint: 'main' }
        });

        this.bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.stateBuf } },
                { binding: 1, resource: { buffer: this.jumpsBuf } },
                { binding: 2, resource: { buffer: this.paramsBuf } },
                { binding: 3, resource: { buffer: this.trapHeaderBuf } },
                { binding: 4, resource: { buffer: this.trapRecordsBuf } }
            ]
        });

        return true;
    }

    async dispatchBatch() {
        if (!this.pipeline || !this.device || this.isSolved) return { steps: 0, lastKey: '' };

        const zeroHeader = new Uint32Array([0, 0, 0, 0]);
        this.device.queue.writeBuffer(this.trapHeaderBuf, 0, zeroHeader);

        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.numKangaroos / 32));
        pass.end();

        const headerBytes = 16;
        const recordsBytes = this.maxTraps * 80;
        encoder.copyBufferToBuffer(this.trapHeaderBuf, 0, this.trapReadbackBuf, 0, headerBytes);
        encoder.copyBufferToBuffer(this.trapRecordsBuf, 0, this.trapReadbackBuf, headerBytes, recordsBytes);

        this.device.queue.submit([encoder.finish()]);

        await this.trapReadbackBuf.mapAsync(GPUMapMode.READ);
        try {
            const arr = new Uint32Array(this.trapReadbackBuf.getMappedRange());
            const trapCount = Math.min(arr[0], this.maxTraps);

            if (trapCount > 0) {
                const headerOffset = 4;
                for (let i = 0; i < trapCount; i++) {
                    const base = headerOffset + i * 20;
                    const xBig = this.u256WordsToBigInt(arr, base);
                    const distBig = this.u256WordsToBigInt(arr, base + 8);
                    const herd = arr[base + 16];
                    const parity = arr[base + 17];
                    const kId = arr[base + 18];

                    const xHex = xBig.toString(16).padStart(64, '0');
                    // 🔍 Öz-denetim: ilk 3 tuzağın x'i CPU'da yeniden hesaplanır
                    if (this._selfCheck === undefined) this._selfCheck = 0;
                    if (this._selfCheck < 3 && this.ec) {
                        this._selfCheck++;
                        try {
                            const dHex = distBig.toString(16);
                            const chk = (herd === 0) ? this.ec.g.mul(dHex) : this.targetPoint.add(this.ec.g.mul(dHex));
                            const cx = chk.getX().toString(16).padStart(64, '0');
                            if (cx !== xHex) { console.error('[GPU Kangaroo] ❌ GPU aritmetiği CPU ile UYUŞMUYOR: çarpışma bulunamaz'); this.gpuMathBroken = true; }
                            else if (this._selfCheck === 3 && !this.gpuMathBroken) { console.log('[GPU Kangaroo] ✅ GPU aritmetiği CPU ile doğrulandı (3 tuzak)'); }
                        } catch (e) {}
                    }
                    const trapRecord = { dist: distBig, parity: parity, id: kId, herd: herd };

                    if (herd === 0) {
                        this.tameTraps.set(xHex, trapRecord);
                        if (this.wildTraps.has(xHex)) {
                            this.checkCollision(xHex, trapRecord, this.wildTraps.get(xHex));
                            if (this.isSolved) break;
                        }
                    } else {
                        this.wildTraps.set(xHex, trapRecord);
                        if (this.tameTraps.has(xHex)) {
                            this.checkCollision(xHex, this.tameTraps.get(xHex), trapRecord);
                            if (this.isSolved) break;
                        }
                    }
                }
            }
        } finally {
            this.trapReadbackBuf.unmap();
        }

        const totalSteps = this.numKangaroos * this.stepsPerCall;
        return { steps: totalSteps, lastKey: this.lastKey };
    }

    checkCollision(xHex, tameRecord, wildRecord) {
        const tDist = tameRecord.dist;
        const tParity = tameRecord.parity;
        const wDist = wildRecord.dist;
        const wParity = wildRecord.parity;

        const isDirect = (tParity === wParity);
        let candDirect1 = (tDist - wDist) % SECP256K1_ORDER;
        if (candDirect1 < 0n) candDirect1 += SECP256K1_ORDER;
        let candDirect2 = (SECP256K1_ORDER - candDirect1) % SECP256K1_ORDER;

        let sumDist = (tDist + wDist) % SECP256K1_ORDER;
        let candNeg1 = (SECP256K1_ORDER - sumDist) % SECP256K1_ORDER;
        let candNeg2 = sumDist;

        const candidates = isDirect
            ? [candDirect1, candDirect2, candNeg1, candNeg2]
            : [candNeg1, candNeg2, candDirect1, candDirect2];

        if (this.targetPoint) {
            const targetX = this.targetPoint.getX().toString(16).padStart(64, '0');
            const targetY = this.targetPoint.getY().toString(16).padStart(64, '0');

            for (const cand of candidates) {
                if (cand <= 0n || cand >= SECP256K1_ORDER) continue;
                try {
                    let checkPt = this.ec.g.mul(cand.toString(16));
                    let checkX = checkPt.getX().toString(16).padStart(64, '0');
                    let checkY = checkPt.getY().toString(16).padStart(64, '0');

                    if (checkX === targetX) {
                        let finalKey = cand;
                        if (checkY !== targetY) {
                            finalKey = (SECP256K1_ORDER - cand) % SECP256K1_ORDER;
                        }
                        this.isSolved = true;
                        this.solvedKey = finalKey;
                        return true;
                    }
                } catch(e) {}
            }
        }
        return false;
    }

    destroy() {
        try {
            if (this.stateBuf) this.stateBuf.destroy();
            if (this.jumpsBuf) this.jumpsBuf.destroy();
            if (this.paramsBuf) this.paramsBuf.destroy();
            if (this.trapHeaderBuf) this.trapHeaderBuf.destroy();
            if (this.trapRecordsBuf) this.trapRecordsBuf.destroy();
            if (this.trapReadbackBuf) this.trapReadbackBuf.destroy();
        } catch(e) {}
    }
}

// =============================================================================
// JS CLASS: GpuEngine (Standart WebGPU Nokta Çarpımı)
// =============================================================================
class GpuEngine {
    constructor() {
        this.device          = null;
        this.adapterInfo     = '';
        this.pipeline        = null;
        this.bindGroup       = null;
        this.privkeyBuf      = null;
        this.pubkeyBuf       = null;
        this.paramsBuf       = null;
        this.matchBuf        = null;
        this.readbackMatchBuf= null;
        this.readbackBuf     = null;
        this.isInitialized   = false;
        this.maxBatchSize    = 32768;
        this.batchSize       = 8192;
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

            const shaderModule = this.device.createShaderModule({ code: GPU_WGSL });
            const compileInfo  = await shaderModule.getCompilationInfo();
            const errors = compileInfo.messages.filter(m => m.type === 'error');
            if (errors.length > 0) {
                console.error('[GPU] WGSL derleme HATA:\n' + errors.map(e => `  Line ${e.lineNum}: ${e.message}`).join('\n'));
                return false;
            }

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

            try {
                await this.computePubkeysRaw([1n]);
            } catch(we) {
                console.warn('[GPU] Warmup uyarısı:', we);
            }

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

    async computePubkeysRaw(privkeysBig) {
        const n = privkeysBig.length;
        if (n === 0 || !this.isInitialized) return new Uint32Array(0);

        const inWords = new Uint32Array(n * 8);
        for (let i = 0; i < n; i++) {
            let k = privkeysBig[i];
            const base = i * 8;
            for (let w = 0; w < 8; w++) {
                inWords[base + w] = Number(k & 0xFFFFFFFFn);
                k >>= 32n;
            }
        }
        this.device.queue.writeBuffer(this.privkeyBuf, 0, inWords);

        const pData = new Uint32Array([n, 0, 0, 0, 0, 0, 0, 0]);
        this.device.queue.writeBuffer(this.paramsBuf, 0, pData);

        const enc = this.device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.dispatchWorkgroups(Math.ceil(n / 64));
        pass.end();

        const copyBytes = n * 26 * 4;
        enc.copyBufferToBuffer(this.pubkeyBuf, 0, this.readbackBuf, 0, copyBytes);
        this.device.queue.submit([enc.finish()]);

        await this.readbackBuf.mapAsync(GPUMapMode.READ, 0, copyBytes);
        const mapped = new Uint32Array(this.readbackBuf.getMappedRange(0, copyBytes));
        const result = new Uint32Array(mapped.length);
        result.set(mapped);
        this.readbackBuf.unmap();
        return result;
    }

    destroyBuffers() {
        try {
            if (this.privkeyBuf)       this.privkeyBuf.destroy();
            if (this.pubkeyBuf)        this.pubkeyBuf.destroy();
            if (this.readbackBuf)      this.readbackBuf.destroy();
            if (this.paramsBuf)        this.paramsBuf.destroy();
            if (this.matchBuf)         this.matchBuf.destroy();
            if (this.readbackMatchBuf) this.readbackMatchBuf.destroy();
        } catch(e) {}
        this.isInitialized = false;
    }
}

// =============================================================================
// PRNG & Key Generators (Worker İçin Bağımsız)
// =============================================================================
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

let gpuPrefixJumpCounter = 0;
let gpuActivePrefix = '';
let gpuAutoStepCounter = 0;
let gpuAutoIndex = 0;

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

// =============================================================================
// STATE & LOOP YÖNETİCİSİ
// =============================================================================
let workerDevice = null;
let workerAdapterInfo = '';
let workerGpuEngine = null;
let workerKangarooEngine = null;
let workerHuntRunning = false;

let workerIntensity = 'BALANCED';
let workerThrottleDelay = 2;
let workerStepsPerCall = 64;
let workerBatchSize = 8192;

const kChannel = new MessageChannel();
kChannel.port1.onmessage = () => { if (workerHuntRunning) workerKangarooLoop(); };
function workerNextKangaroo() { kChannel.port2.postMessage(null); }

const hChannel = new MessageChannel();
hChannel.port1.onmessage = () => { if (workerHuntRunning) workerHuntLoop(); };
function workerNextHunt() { hChannel.port2.postMessage(null); }

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

// =============================================================================
// KANGAROO LOOP
// =============================================================================
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
                rangeText: `⚡ WebGPU Worker Kangaroo — 2048 Sürü (${rate.toLocaleString()} key/s)`
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
            if (workerThrottleDelay <= 0) {
                workerNextKangaroo();
            } else {
                setTimeout(workerKangarooLoop, workerThrottleDelay);
            }
        }
    } catch(err) {
        console.error('[WebGPU Worker Kangaroo Error]:', err);
        self.postMessage({ type: 'error', error: err.message });
        workerHuntRunning = false;
    }
}

// =============================================================================
// STANDART GPU HUNT LOOP (Sobol, Kaos, vb.)
// =============================================================================
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
            if (workerThrottleDelay <= 0) {
                workerNextHunt();
            } else {
                setTimeout(workerHuntLoop, workerThrottleDelay);
            }
        }
    } catch(err) {
        console.error('[WebGPU Worker Standard Loop Error]:', err);
        self.postMessage({ type: 'error', error: err.message });
        workerHuntRunning = false;
    }
}

// =============================================================================
// MESAJ DİNLEYİCİSİ (COMMAND HANDLER)
// =============================================================================
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
        workerThrottleDelay = (data.throttleDelay !== undefined) ? data.throttleDelay : 2;
        workerStepsPerCall = data.stepsPerCall || 64;
        workerBatchSize = (workerIntensity === 'ECO') ? 2048 : ((workerIntensity === 'MAX') ? 16384 : 8192);

        workerTotalKeys = 0;
        workerLastTime = performance.now();
        workerAccumSteps = 0;
        workerHuntRunning = true;

        // Çoklu hedef haritası oluşturma
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
            // Standart GPU Arama Modu
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
        workerThrottleDelay = (data.throttleDelay !== undefined) ? data.throttleDelay : workerThrottleDelay;
        workerStepsPerCall = data.stepsPerCall || workerStepsPerCall;
        workerBatchSize = (workerIntensity === 'ECO') ? 2048 : ((workerIntensity === 'MAX') ? 16384 : 8192);

        if (workerKangarooEngine) {
            workerKangarooEngine.setStepsPerCall(workerStepsPerCall);
        }
        if (workerGpuEngine) {
            workerGpuEngine.batchSize = workerBatchSize;
        }
    }
};
