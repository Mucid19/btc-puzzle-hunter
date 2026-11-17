(function (global) {
    'use strict';

    const SECP_P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2Fn;
    const SECP_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141n;
    const SECP_LAMBDA = 0x5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72n;
    const GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798n;
    const GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8n;

    const STEPS_PER_THREAD = 64;
    const MAX_THREADS = 16384;
    const INTENSITY_MAP = { LOW: 2048, MEDIUM: 4096, HIGH: 8192, ULTRA: 16384 };
    const DEFAULT_THREADS = 4096;
    const MAX_BATCH_DISPATCHES = 16; // Adaptive up to 16 dispatches per readback targeting ~30-60ms per batch
    const BLOOM_WORDS = 524288; // 524288 * 32 = 2^24 bits (2 MB)
    const BLOOM_MASK = 0xFFFFFF;
    const KANG_PER_THREAD = 4;
    const KANG_STEPS_PER_DISPATCH = 16;
    const MAX_DP_OUT = 256;

    const WGSL_KERNEL = `
struct Params {
  target0: u32,
  target1: u32,
  target2: u32,
  target3: u32,
  target4: u32,
  stepsPerThread: u32,
  numThreads: u32,
  dispatchIdx: u32,
  bloomMask: u32,
  useGlvSym: u32,
  pad0: u32,
  pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> ptX: array<u32>;
@group(0) @binding(2) var<storage, read_write> ptY: array<u32>;
@group(0) @binding(3) var<storage, read> jumpX: array<u32>;
@group(0) @binding(4) var<storage, read> jumpY: array<u32>;
@group(0) @binding(5) var<storage, read_write> outHit: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> bloom: array<u32>;

fn getP(i: u32) -> u32 {
  if (i == 0u) { return 0xFFFFFC2Fu; }
  if (i == 1u) { return 0xFFFFFFFEu; }
  return 0xFFFFFFFFu;
}

fn fe_gte_p(a: ptr<function, array<u32, 8>>) -> bool {
  for (var idx: i32 = 7; idx >= 0; idx = idx - 1) {
    let i = u32(idx);
    let pi = getP(i);
    if ((*a)[i] > pi) { return true; }
    if ((*a)[i] < pi) { return false; }
  }
  return true;
}

fn fe_sub_p(a: ptr<function, array<u32, 8>>) {
  var borrow: u32 = 0u;
  for (var i: u32 = 0u; i < 8u; i = i + 1u) {
    let ai = (*a)[i];
    let pi = getP(i);
    let sub1 = ai - pi;
    let b1 = select(0u, 1u, ai < pi);
    let sub2 = sub1 - borrow;
    let b2 = select(0u, 1u, sub1 < borrow);
    (*a)[i] = sub2;
    borrow = b1 + b2;
  }
}

fn fe_sub(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
  var r: array<u32, 8>;
  var borrow: u32 = 0u;
  for (var i: u32 = 0u; i < 8u; i = i + 1u) {
    let s1 = a[i] - b[i];
    let b1 = select(0u, 1u, a[i] < b[i]);
    let s2 = s1 - borrow;
    let b2 = select(0u, 1u, s1 < borrow);
    r[i] = s2;
    borrow = b1 + b2;
  }
  if (borrow != 0u) {
    var carry: u32 = 0u;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
      let pi = getP(i);
      let s1 = r[i] + pi;
      let c1 = select(0u, 1u, s1 < r[i]);
      let s2 = s1 + carry;
      let c2 = select(0u, 1u, s2 < s1);
      r[i] = s2;
      carry = c1 + c2;
    }
  }
  return r;
}

fn mul32_64(a: u32, b: u32) -> vec2<u32> {
  let a0 = a & 0xFFFFu;
  let a1 = a >> 16u;
  let b0 = b & 0xFFFFu;
  let b1 = b >> 16u;
  let p00 = a0 * b0;
  let p01 = a0 * b1;
  let p10 = a1 * b0;
  let p11 = a1 * b1;
  let mid1 = p01 + (p00 >> 16u);
  let mid2 = mid1 + p10;
  let c_mid = select(0u, 1u, mid2 < mid1);
  let lo = (p00 & 0xFFFFu) | (mid2 << 16u);
  let hi = p11 + (mid2 >> 16u) + (c_mid << 16u);
  return vec2<u32>(lo, hi);
}

fn fe_reduce_512(t_in: array<u32, 16>) -> array<u32, 8> {
  var t = t_in;
  var r: array<u32, 8>;
  for (var i: u32 = 0u; i < 8u; i = i + 1u) { r[i] = t[i]; }
  var c977: u32 = 0u;
  for (var i: u32 = 0u; i < 8u; i = i + 1u) {
    let p = mul32_64(t[8u + i], 977u);
    let s1 = r[i] + p.x;
    let c1 = select(0u, 1u, s1 < r[i]);
    let s2 = s1 + c977;
    let c2 = select(0u, 1u, s2 < s1);
    r[i] = s2;
    c977 = p.y + c1 + c2;
  }
  var csh: u32 = 0u;
  for (var i: u32 = 1u; i < 8u; i = i + 1u) {
    let v = t[7u + i];
    let s1 = r[i] + v;
    let c1 = select(0u, 1u, s1 < r[i]);
    let s2 = s1 + csh;
    let c2 = select(0u, 1u, s2 < s1);
    r[i] = s2;
    csh = c1 + c2;
  }
  let top = c977 + csh;
  let ft = mul32_64(top, 977u);
  let f15 = mul32_64(t[15], 977u);

  var s0 = r[0] + ft.x;
  var cy0 = select(0u, 1u, s0 < r[0]);
  let s0b = s0 + f15.x;
  cy0 = cy0 + select(0u, 1u, s0b < s0);
  r[0] = s0b;

  var add1 = ft.y + cy0;
  var cy1 = select(0u, 1u, add1 < ft.y);
  let add1b = add1 + top;
  cy1 = cy1 + select(0u, 1u, add1b < add1);
  let add1c = add1b + f15.y;
  cy1 = cy1 + select(0u, 1u, add1c < add1b);
  let add1d = add1c + t[15];
  cy1 = cy1 + select(0u, 1u, add1d < add1c);
  let s1 = r[1] + add1d;
  cy1 = cy1 + select(0u, 1u, s1 < r[1]);
  r[1] = s1;

  var cy2 = cy1;
  for (var i: u32 = 2u; i < 8u; i = i + 1u) {
    if (cy2 == 0u) { break; }
    let sn = r[i] + cy2;
    cy2 = select(0u, 1u, sn < r[i]);
    r[i] = sn;
  }
  if (cy2 != 0u) {
    let f2 = mul32_64(cy2, 977u);
    let sn0 = r[0] + f2.x;
    var c2b = select(0u, 1u, sn0 < r[0]);
    r[0] = sn0;
    let a1 = r[1] + f2.y;
    var c1w = select(0u, 1u, a1 < r[1]);
    let b1 = a1 + cy2;
    c1w = c1w + select(0u, 1u, b1 < a1);
    let d1 = b1 + c2b;
    c1w = c1w + select(0u, 1u, d1 < b1);
    r[1] = d1;
    for (var i: u32 = 2u; i < 8u; i = i + 1u) {
      if (c1w == 0u) { break; }
      let sn = r[i] + c1w;
      c1w = select(0u, 1u, sn < r[i]);
      r[i] = sn;
    }
  }
  if (fe_gte_p(&r)) { fe_sub_p(&r); }
  if (fe_gte_p(&r)) { fe_sub_p(&r); }
  return r;
}

fn fe_mul(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
  var t: array<u32, 16>;
  for (var k: u32 = 0u; k < 16u; k = k + 1u) { t[k] = 0u; }
  for (var i: u32 = 0u; i < 8u; i = i + 1u) {
    var carry: u32 = 0u;
    let ai = a[i];
    for (var j: u32 = 0u; j < 8u; j = j + 1u) {
      let p = mul32_64(ai, b[j]);
      let s1 = t[i + j] + p.x;
      let c1 = select(0u, 1u, s1 < t[i + j]);
      let s2 = s1 + carry;
      let c2 = select(0u, 1u, s2 < s1);
      t[i + j] = s2;
      carry = p.y + c1 + c2;
    }
    t[i + 8u] = t[i + 8u] + carry;
  }
  return fe_reduce_512(t);
}

fn fe_sqr(a: array<u32, 8>) -> array<u32, 8> {
  var t: array<u32, 16>;
  for (var k: u32 = 0u; k < 16u; k = k + 1u) { t[k] = 0u; }
  for (var i: u32 = 0u; i < 7u; i = i + 1u) {
    var carry: u32 = 0u;
    let ai = a[i];
    for (var j: u32 = i + 1u; j < 8u; j = j + 1u) {
      let p = mul32_64(ai, a[j]);
      let s1 = t[i + j] + p.x;
      let c1 = select(0u, 1u, s1 < t[i + j]);
      let s2 = s1 + carry;
      let c2 = select(0u, 1u, s2 < s1);
      t[i + j] = s2;
      carry = p.y + c1 + c2;
    }
    var k: u32 = i + 8u;
    var hc: u32 = carry;
    while (hc != 0u && k < 16u) {
      let sk = t[k] + hc;
      hc = select(0u, 1u, sk < t[k]);
      t[k] = sk;
      k = k + 1u;
    }
  }
  var dbl_c: u32 = 0u;
  for (var k: u32 = 1u; k < 16u; k = k + 1u) {
    let tk = t[k];
    t[k] = (tk << 1u) | dbl_c;
    dbl_c = tk >> 31u;
  }
  var sq_c: u32 = 0u;
  for (var i: u32 = 0u; i < 8u; i = i + 1u) {
    let p = mul32_64(a[i], a[i]);
    let k0 = 2u * i;
    let s0 = t[k0] + p.x;
    let c0 = select(0u, 1u, s0 < t[k0]);
    let s0b = s0 + sq_c;
    let c0b = select(0u, 1u, s0b < s0);
    t[k0] = s0b;
    let s1 = t[k0 + 1u] + p.y;
    let c1 = select(0u, 1u, s1 < t[k0 + 1u]);
    let s1b = s1 + c0 + c0b;
    let c1b = select(0u, 1u, s1b < s1);
    t[k0 + 1u] = s1b;
    sq_c = c1 + c1b;
  }
  return fe_reduce_512(t);
}

fn fe_inv(x: array<u32, 8>) -> array<u32, 8> {
  var x2 = fe_mul(fe_sqr(x), x);
  var x3 = fe_mul(fe_sqr(x2), x);
  var x6 = x3;
  for (var i: u32 = 0u; i < 3u; i = i + 1u) { x6 = fe_sqr(x6); }
  x6 = fe_mul(x6, x3);
  var x9 = x6;
  for (var i: u32 = 0u; i < 3u; i = i + 1u) { x9 = fe_sqr(x9); }
  x9 = fe_mul(x9, x3);
  var x11 = x9;
  for (var i: u32 = 0u; i < 2u; i = i + 1u) { x11 = fe_sqr(x11); }
  x11 = fe_mul(x11, x2);
  var x22 = x11;
  for (var i: u32 = 0u; i < 11u; i = i + 1u) { x22 = fe_sqr(x22); }
  x22 = fe_mul(x22, x11);
  var x44 = x22;
  for (var i: u32 = 0u; i < 22u; i = i + 1u) { x44 = fe_sqr(x44); }
  x44 = fe_mul(x44, x22);
  var x88 = x44;
  for (var i: u32 = 0u; i < 44u; i = i + 1u) { x88 = fe_sqr(x88); }
  x88 = fe_mul(x88, x44);
  var x176 = x88;
  for (var i: u32 = 0u; i < 88u; i = i + 1u) { x176 = fe_sqr(x176); }
  x176 = fe_mul(x176, x88);
  var x220 = x176;
  for (var i: u32 = 0u; i < 44u; i = i + 1u) { x220 = fe_sqr(x220); }
  x220 = fe_mul(x220, x44);
  var x223 = x220;
  for (var i: u32 = 0u; i < 3u; i = i + 1u) { x223 = fe_sqr(x223); }
  x223 = fe_mul(x223, x3);

  var t = x223;
  for (var i: u32 = 0u; i < 23u; i = i + 1u) { t = fe_sqr(t); }
  t = fe_mul(t, x22);
  for (var i: u32 = 0u; i < 5u; i = i + 1u) { t = fe_sqr(t); }
  t = fe_mul(t, x);
  for (var i: u32 = 0u; i < 3u; i = i + 1u) { t = fe_sqr(t); }
  t = fe_mul(t, x2);
  for (var i: u32 = 0u; i < 2u; i = i + 1u) { t = fe_sqr(t); }
  t = fe_mul(t, x);
  return t;
}

fn rotr(x: u32, n: u32) -> u32 { return (x >> n) | (x << (32u - n)); }
fn rotl(x: u32, n: u32) -> u32 { return (x << n) | (x >> (32u - n)); }
fn bswap32(v: u32) -> u32 {
  return ((v & 0xFFu) << 24u) | ((v & 0xFF00u) << 8u) | ((v >> 8u) & 0xFF00u) | (v >> 24u);
}

fn sha256_k(i: u32) -> u32 {
  switch(i) {
    case 0u:{return 0x428a2f98u;} case 1u:{return 0x71374491u;} case 2u:{return 0xb5c0fbcfu;} case 3u:{return 0xe9b5dba5u;}
    case 4u:{return 0x3956c25bu;} case 5u:{return 0x59f111f1u;} case 6u:{return 0x923f82a4u;} case 7u:{return 0xab1c5ed5u;}
    case 8u:{return 0xd807aa98u;} case 9u:{return 0x12835b01u;} case 10u:{return 0x243185beu;} case 11u:{return 0x550c7dc3u;}
    case 12u:{return 0x72be5d74u;} case 13u:{return 0x80deb1feu;} case 14u:{return 0x9bdc06a7u;} case 15u:{return 0xc19bf174u;}
    case 16u:{return 0xe49b69c1u;} case 17u:{return 0xefbe4786u;} case 18u:{return 0x0fc19dc6u;} case 19u:{return 0x240ca1ccu;}
    case 20u:{return 0x2de92c6fu;} case 21u:{return 0x4a7484aau;} case 22u:{return 0x5cb0a9dcu;} case 23u:{return 0x76f988dau;}
    case 24u:{return 0x983e5152u;} case 25u:{return 0xa831c66du;} case 26u:{return 0xb00327c8u;} case 27u:{return 0xbf597fc7u;}
    case 28u:{return 0xc6e00bf3u;} case 29u:{return 0xd5a79147u;} case 30u:{return 0x06ca6351u;} case 31u:{return 0x14292967u;}
    case 32u:{return 0x27b70a85u;} case 33u:{return 0x2e1b2138u;} case 34u:{return 0x4d2c6dfcu;} case 35u:{return 0x53380d13u;}
    case 36u:{return 0x650a7354u;} case 37u:{return 0x766a0abbu;} case 38u:{return 0x81c2c92eu;} case 39u:{return 0x92722c85u;}
    case 40u:{return 0xa2bfe8a1u;} case 41u:{return 0xa81a664bu;} case 42u:{return 0xc24b8b70u;} case 43u:{return 0xc76c51a3u;}
    case 44u:{return 0xd192e819u;} case 45u:{return 0xd6990624u;} case 46u:{return 0xf40e3585u;} case 47u:{return 0x106aa070u;}
    case 48u:{return 0x19a4c116u;} case 49u:{return 0x1e376c08u;} case 50u:{return 0x2748774cu;} case 51u:{return 0x34b0bcb5u;}
    case 52u:{return 0x391c0cb3u;} case 53u:{return 0x4ed8aa4au;} case 54u:{return 0x5b9cca4fu;} case 55u:{return 0x682e6ff3u;}
    case 56u:{return 0x748f82eeu;} case 57u:{return 0x78a5636fu;} case 58u:{return 0x84c87814u;} case 59u:{return 0x8cc70208u;}
    case 60u:{return 0x90befffau;} case 61u:{return 0xa4506cebu;} case 62u:{return 0xbef9a3f7u;} default:{return 0xc67178f2u;}
  }
}

fn sha256_33(prefixByte: u32, x: array<u32, 8>) -> array<u32, 8> {
  var w: array<u32, 64>;
  w[0] = (prefixByte << 24u) | (x[7] >> 8u);
  w[1] = ((x[7] & 0xFFu) << 24u) | (x[6] >> 8u);
  w[2] = ((x[6] & 0xFFu) << 24u) | (x[5] >> 8u);
  w[3] = ((x[5] & 0xFFu) << 24u) | (x[4] >> 8u);
  w[4] = ((x[4] & 0xFFu) << 24u) | (x[3] >> 8u);
  w[5] = ((x[3] & 0xFFu) << 24u) | (x[2] >> 8u);
  w[6] = ((x[2] & 0xFFu) << 24u) | (x[1] >> 8u);
  w[7] = ((x[1] & 0xFFu) << 24u) | (x[0] >> 8u);
  w[8] = ((x[0] & 0xFFu) << 24u) | 0x00800000u;
  for (var i: u32 = 9u; i < 15u; i = i + 1u) { w[i] = 0u; }
  w[15] = 264u;
  for (var i: u32 = 16u; i < 64u; i = i + 1u) {
    let s0 = rotr(w[i - 15u], 7u) ^ rotr(w[i - 15u], 18u) ^ (w[i - 15u] >> 3u);
    let s1 = rotr(w[i - 2u], 17u) ^ rotr(w[i - 2u], 19u) ^ (w[i - 2u] >> 10u);
    w[i] = w[i - 16u] + s0 + w[i - 7u] + s1;
  }
  var a: u32 = 0x6a09e667u; var b: u32 = 0xbb67ae85u; var c: u32 = 0x3c6ef372u; var d: u32 = 0xa54ff53au;
  var e: u32 = 0x510e527fu; var f: u32 = 0x9b05688cu; var g: u32 = 0x1f83d9abu; var h: u32 = 0x5be0cd19u;
  for (var i: u32 = 0u; i < 64u; i = i + 1u) {
    let s1 = rotr(e, 6u) ^ rotr(e, 11u) ^ rotr(e, 25u);
    let ch = (e & f) ^ ((~e) & g);
    let t1 = h + s1 + ch + sha256_k(i) + w[i];
    let s0 = rotr(a, 2u) ^ rotr(a, 13u) ^ rotr(a, 22u);
    let maj = (a & b) ^ (a & c) ^ (b & c);
    let t2 = s0 + maj;
    h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
  }
  var out: array<u32, 8>;
  out[0] = 0x6a09e667u + a; out[1] = 0xbb67ae85u + b; out[2] = 0x3c6ef372u + c; out[3] = 0xa54ff53au + d;
  out[4] = 0x510e527fu + e; out[5] = 0x9b05688cu + f; out[6] = 0x1f83d9abu + g; out[7] = 0x5be0cd19u + h;
  return out;
}

fn rmd_f(j: u32, x: u32, y: u32, z: u32) -> u32 {
  if (j < 16u) { return x ^ y ^ z; }
  if (j < 32u) { return (x & y) | ((~x) & z); }
  if (j < 48u) { return (x | (~y)) ^ z; }
  if (j < 64u) { return (x & z) | (y & (~z)); }
  return x ^ (y | (~z));
}
fn rmd_k(j: u32) -> u32 {
  if (j < 16u) { return 0x00000000u; }
  if (j < 32u) { return 0x5A827999u; }
  if (j < 48u) { return 0x6ED9EBA1u; }
  if (j < 64u) { return 0x8F1BBCDCu; }
  return 0xA953FD4Eu;
}
fn rmd_kp(j: u32) -> u32 {
  if (j < 16u) { return 0x50A28BE6u; }
  if (j < 32u) { return 0x5C4DD124u; }
  if (j < 48u) { return 0x6D703EF3u; }
  if (j < 64u) { return 0x7A6D76E9u; }
  return 0x00000000u;
}
fn rmd_r(j: u32) -> u32 {
  switch(j) {
    case 0u:{return 0u;} case 1u:{return 1u;} case 2u:{return 2u;} case 3u:{return 3u;}
    case 4u:{return 4u;} case 5u:{return 5u;} case 6u:{return 6u;} case 7u:{return 7u;}
    case 8u:{return 8u;} case 9u:{return 9u;} case 10u:{return 10u;} case 11u:{return 11u;}
    case 12u:{return 12u;} case 13u:{return 13u;} case 14u:{return 14u;} case 15u:{return 15u;}
    case 16u:{return 7u;} case 17u:{return 4u;} case 18u:{return 13u;} case 19u:{return 1u;}
    case 20u:{return 10u;} case 21u:{return 6u;} case 22u:{return 15u;} case 23u:{return 3u;}
    case 24u:{return 12u;} case 25u:{return 0u;} case 26u:{return 9u;} case 27u:{return 5u;}
    case 28u:{return 2u;} case 29u:{return 14u;} case 30u:{return 11u;} case 31u:{return 8u;}
    case 32u:{return 3u;} case 33u:{return 10u;} case 34u:{return 14u;} case 35u:{return 4u;}
    case 36u:{return 9u;} case 37u:{return 15u;} case 38u:{return 8u;} case 39u:{return 1u;}
    case 40u:{return 2u;} case 41u:{return 7u;} case 42u:{return 0u;} case 43u:{return 6u;}
    case 44u:{return 13u;} case 45u:{return 11u;} case 46u:{return 5u;} case 47u:{return 12u;}
    case 48u:{return 1u;} case 49u:{return 9u;} case 50u:{return 11u;} case 51u:{return 10u;}
    case 52u:{return 0u;} case 53u:{return 8u;} case 54u:{return 12u;} case 55u:{return 4u;}
    case 56u:{return 13u;} case 57u:{return 3u;} case 58u:{return 7u;} case 59u:{return 15u;}
    case 60u:{return 14u;} case 61u:{return 5u;} case 62u:{return 6u;} case 63u:{return 2u;}
    case 64u:{return 4u;} case 65u:{return 0u;} case 66u:{return 5u;} case 67u:{return 9u;}
    case 68u:{return 7u;} case 69u:{return 12u;} case 70u:{return 2u;} case 71u:{return 10u;}
    case 72u:{return 14u;} case 73u:{return 1u;} case 74u:{return 3u;} case 75u:{return 8u;}
    case 76u:{return 11u;} case 77u:{return 6u;} case 78u:{return 15u;} default:{return 13u;}
  }
}
fn rmd_rp(j: u32) -> u32 {
  switch(j) {
    case 0u:{return 5u;} case 1u:{return 14u;} case 2u:{return 7u;} case 3u:{return 0u;}
    case 4u:{return 9u;} case 5u:{return 2u;} case 6u:{return 11u;} case 7u:{return 4u;}
    case 8u:{return 13u;} case 9u:{return 6u;} case 10u:{return 15u;} case 11u:{return 8u;}
    case 12u:{return 1u;} case 13u:{return 10u;} case 14u:{return 3u;} case 15u:{return 12u;}
    case 16u:{return 6u;} case 17u:{return 11u;} case 18u:{return 3u;} case 19u:{return 7u;}
    case 20u:{return 0u;} case 21u:{return 13u;} case 22u:{return 5u;} case 23u:{return 10u;}
    case 24u:{return 14u;} case 25u:{return 15u;} case 26u:{return 8u;} case 27u:{return 12u;}
    case 28u:{return 4u;} case 29u:{return 9u;} case 30u:{return 1u;} case 31u:{return 2u;}
    case 32u:{return 15u;} case 33u:{return 5u;} case 34u:{return 1u;} case 35u:{return 3u;}
    case 36u:{return 7u;} case 37u:{return 14u;} case 38u:{return 6u;} case 39u:{return 9u;}
    case 40u:{return 11u;} case 41u:{return 8u;} case 42u:{return 12u;} case 43u:{return 2u;}
    case 44u:{return 10u;} case 45u:{return 0u;} case 46u:{return 4u;} case 47u:{return 13u;}
    case 48u:{return 8u;} case 49u:{return 6u;} case 50u:{return 4u;} case 51u:{return 1u;}
    case 52u:{return 3u;} case 53u:{return 11u;} case 54u:{return 15u;} case 55u:{return 0u;}
    case 56u:{return 5u;} case 57u:{return 12u;} case 58u:{return 2u;} case 59u:{return 13u;}
    case 60u:{return 9u;} case 61u:{return 7u;} case 62u:{return 10u;} case 63u:{return 14u;}
    case 64u:{return 12u;} case 65u:{return 15u;} case 66u:{return 10u;} case 67u:{return 4u;}
    case 68u:{return 1u;} case 69u:{return 5u;} case 70u:{return 8u;} case 71u:{return 7u;}
    case 72u:{return 6u;} case 73u:{return 2u;} case 74u:{return 13u;} case 75u:{return 14u;}
    case 76u:{return 0u;} case 77u:{return 3u;} case 78u:{return 9u;} default:{return 11u;}
  }
}
fn rmd_s(j: u32) -> u32 {
  switch(j) {
    case 0u:{return 11u;} case 1u:{return 14u;} case 2u:{return 15u;} case 3u:{return 12u;}
    case 4u:{return 5u;} case 5u:{return 8u;} case 6u:{return 7u;} case 7u:{return 9u;}
    case 8u:{return 11u;} case 9u:{return 13u;} case 10u:{return 14u;} case 11u:{return 15u;}
    case 12u:{return 6u;} case 13u:{return 7u;} case 14u:{return 9u;} case 15u:{return 8u;}
    case 16u:{return 7u;} case 17u:{return 6u;} case 18u:{return 8u;} case 19u:{return 13u;}
    case 20u:{return 11u;} case 21u:{return 9u;} case 22u:{return 7u;} case 23u:{return 15u;}
    case 24u:{return 7u;} case 25u:{return 12u;} case 26u:{return 15u;} case 27u:{return 9u;}
    case 28u:{return 11u;} case 29u:{return 7u;} case 30u:{return 13u;} case 31u:{return 12u;}
    case 32u:{return 11u;} case 33u:{return 13u;} case 34u:{return 6u;} case 35u:{return 7u;}
    case 36u:{return 14u;} case 37u:{return 9u;} case 38u:{return 13u;} case 39u:{return 15u;}
    case 40u:{return 14u;} case 41u:{return 8u;} case 42u:{return 13u;} case 43u:{return 6u;}
    case 44u:{return 5u;} case 45u:{return 12u;} case 46u:{return 7u;} case 47u:{return 5u;}
    case 48u:{return 11u;} case 49u:{return 12u;} case 50u:{return 14u;} case 51u:{return 15u;}
    case 52u:{return 14u;} case 53u:{return 15u;} case 54u:{return 9u;} case 55u:{return 8u;}
    case 56u:{return 9u;} case 57u:{return 14u;} case 58u:{return 5u;} case 59u:{return 6u;}
    case 60u:{return 8u;} case 61u:{return 6u;} case 62u:{return 5u;} case 63u:{return 12u;}
    case 64u:{return 9u;} case 65u:{return 15u;} case 66u:{return 5u;} case 67u:{return 11u;}
    case 68u:{return 6u;} case 69u:{return 8u;} case 70u:{return 13u;} case 71u:{return 12u;}
    case 72u:{return 5u;} case 73u:{return 12u;} case 74u:{return 13u;} case 75u:{return 14u;}
    case 76u:{return 11u;} case 77u:{return 8u;} case 78u:{return 5u;} default:{return 6u;}
  }
}
fn rmd_sp(j: u32) -> u32 {
  switch(j) {
    case 0u:{return 8u;} case 1u:{return 9u;} case 2u:{return 9u;} case 3u:{return 11u;}
    case 4u:{return 13u;} case 5u:{return 15u;} case 6u:{return 15u;} case 7u:{return 5u;}
    case 8u:{return 7u;} case 9u:{return 7u;} case 10u:{return 8u;} case 11u:{return 11u;}
    case 12u:{return 14u;} case 13u:{return 14u;} case 14u:{return 12u;} case 15u:{return 6u;}
    case 16u:{return 9u;} case 17u:{return 13u;} case 18u:{return 15u;} case 19u:{return 7u;}
    case 20u:{return 12u;} case 21u:{return 8u;} case 22u:{return 9u;} case 23u:{return 11u;}
    case 24u:{return 7u;} case 25u:{return 7u;} case 26u:{return 12u;} case 27u:{return 7u;}
    case 28u:{return 6u;} case 29u:{return 15u;} case 30u:{return 13u;} case 31u:{return 11u;}
    case 32u:{return 9u;} case 33u:{return 7u;} case 34u:{return 15u;} case 35u:{return 11u;}
    case 36u:{return 8u;} case 37u:{return 6u;} case 38u:{return 6u;} case 39u:{return 14u;}
    case 40u:{return 12u;} case 41u:{return 13u;} case 42u:{return 5u;} case 43u:{return 14u;}
    case 44u:{return 13u;} case 45u:{return 13u;} case 46u:{return 7u;} case 47u:{return 5u;}
    case 48u:{return 15u;} case 49u:{return 5u;} case 50u:{return 8u;} case 51u:{return 11u;}
    case 52u:{return 14u;} case 53u:{return 14u;} case 54u:{return 6u;} case 55u:{return 14u;}
    case 56u:{return 6u;} case 57u:{return 9u;} case 58u:{return 12u;} case 59u:{return 9u;}
    case 60u:{return 12u;} case 61u:{return 5u;} case 62u:{return 15u;} case 63u:{return 8u;}
    case 64u:{return 8u;} case 65u:{return 5u;} case 66u:{return 12u;} case 67u:{return 9u;}
    case 68u:{return 12u;} case 69u:{return 5u;} case 70u:{return 14u;} case 71u:{return 6u;}
    case 72u:{return 8u;} case 73u:{return 13u;} case 74u:{return 6u;} case 75u:{return 5u;}
    case 76u:{return 15u;} case 77u:{return 13u;} case 78u:{return 11u;} default:{return 11u;}
  }
}

fn ripemd160_32(sha: array<u32, 8>) -> array<u32, 5> {
  var X: array<u32, 16>;
  for (var i: u32 = 0u; i < 8u; i = i + 1u) { X[i] = bswap32(sha[i]); }
  X[8] = 0x00000080u;
  for (var i: u32 = 9u; i < 14u; i = i + 1u) { X[i] = 0u; }
  X[14] = 256u;
  X[15] = 0u;

  var al: u32 = 0x67452301u; var bl: u32 = 0xEFCDAB89u; var cl: u32 = 0x98BADCFEu; var dl: u32 = 0x10325476u; var el: u32 = 0xC3D2E1F0u;
  var ar: u32 = al; var br: u32 = bl; var cr: u32 = cl; var dr: u32 = dl; var er: u32 = el;
  for (var j: u32 = 0u; j < 80u; j = j + 1u) {
    var t = al + rmd_f(j, bl, cl, dl) + X[rmd_r(j)] + rmd_k(j);
    t = rotl(t, rmd_s(j)) + el;
    al = el; el = dl; dl = rotl(cl, 10u); cl = bl; bl = t;

    var tp = ar + rmd_f(79u - j, br, cr, dr) + X[rmd_rp(j)] + rmd_kp(j);
    tp = rotl(tp, rmd_sp(j)) + er;
    ar = er; er = dr; dr = rotl(cr, 10u); cr = br; br = tp;
  }
  var out: array<u32, 5>;
  out[0] = bswap32(0xEFCDAB89u + cl + dr);
  out[1] = bswap32(0x98BADCFEu + dl + er);
  out[2] = bswap32(0x10325476u + el + ar);
  out[3] = bswap32(0xC3D2E1F0u + al + br);
  out[4] = bswap32(0x67452301u + bl + cr);
  return out;
}

fn getBeta() -> array<u32, 8> {
  var b: array<u32, 8>;
  b[0] = 0x719501EEu; b[1] = 0xC1396C28u; b[2] = 0x12F58995u; b[3] = 0x9CF04975u;
  b[4] = 0xAC3434E9u; b[5] = 0x6E64479Eu; b[6] = 0x657C0710u; b[7] = 0x7AE96A2Bu;
  return b;
}

fn check_h160(h160: array<u32, 5>, tid: u32, stepVal: u32, subKind: u32) {
  if (h160[0] == params.target0 && h160[1] == params.target1 && h160[2] == params.target2 && h160[3] == params.target3 && h160[4] == params.target4) {
    let res = atomicCompareExchangeWeak(&outHit[0], 0u, 1u);
    if (res.exchanged) {
      atomicStore(&outHit[1], tid);
      atomicStore(&outHit[2], (subKind << 16u) | stepVal);
      atomicStore(&outHit[3], params.dispatchIdx);
    }
    return;
  }
  if (params.bloomMask != 0u) {
    let b0 = (h160[0] ^ (h160[3] * 0x9E3779B1u)) & params.bloomMask;
    if ((bloom[b0 >> 5u] & (1u << (b0 & 31u))) != 0u) {
      let b1 = (h160[1] ^ (h160[4] * 0x85EBCA77u)) & params.bloomMask;
      if ((bloom[b1 >> 5u] & (1u << (b1 & 31u))) != 0u) {
        let b2 = ((h160[2] ^ h160[0]) * 0xC2B2AE3Du) & params.bloomMask;
        if ((bloom[b2 >> 5u] & (1u << (b2 & 31u))) != 0u) {
          let b3 = ((h160[3] ^ h160[1] ^ h160[4]) * 0x27D4EB2Fu) & params.bloomMask;
          if ((bloom[b3 >> 5u] & (1u << (b3 & 31u))) != 0u) {
            let res = atomicCompareExchangeWeak(&outHit[4], 0u, 2u);
            if (res.exchanged) {
              atomicStore(&outHit[5], tid);
              atomicStore(&outHit[6], (subKind << 16u) | stepVal);
              atomicStore(&outHit[7], params.dispatchIdx);
            }
          }
        }
      }
    }
  }
}

fn check_point(pfx: u32, px: array<u32, 8>, tid: u32, stepVal: u32) {
  check_h160(ripemd160_32(sha256_33(pfx, px)), tid, stepVal, 0u);
  if (params.useGlvSym != 0u) {
    check_h160(ripemd160_32(sha256_33(pfx ^ 1u, px)), tid, stepVal, 1u);
    let bx = fe_mul(px, getBeta());
    check_h160(ripemd160_32(sha256_33(pfx, bx)), tid, stepVal, 2u);
    check_h160(ripemd160_32(sha256_33(pfx ^ 1u, bx)), tid, stepVal, 3u);
  }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tid = gid.x;
  if (tid >= params.numThreads) { return; }

  let base = tid * 8u;
  var cx: array<u32, 8>;
  var cy: array<u32, 8>;
  for (var i: u32 = 0u; i < 8u; i = i + 1u) {
    cx[i] = ptX[base + i];
    cy[i] = ptY[base + i];
  }

  check_point(select(2u, 3u, (cy[0] & 1u) == 1u), cx, tid, 0u);

  let S = params.stepsPerThread;
  var cArr: array<array<u32, 8>, 64>;

  for (var s: u32 = 0u; s < S; s = s + 1u) {
    let jb = s * 8u;
    var jx: array<u32, 8>;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) { jx[i] = jumpX[jb + i]; }
    let dx = fe_sub(jx, cx);
    if (s == 0u) {
      cArr[0] = dx;
    } else {
      cArr[s] = fe_mul(cArr[s - 1u], dx);
    }
  }

  var invAll = fe_inv(cArr[S - 1u]);
  var nextX: array<u32, 8>;
  var nextY: array<u32, 8>;

  for (var idx: i32 = i32(S) - 1; idx >= 0; idx = idx - 1) {
    let s = u32(idx);
    let jb = s * 8u;
    var jx: array<u32, 8>;
    var jy: array<u32, 8>;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
      jx[i] = jumpX[jb + i];
      jy[i] = jumpY[jb + i];
    }
    var invDx: array<u32, 8>;
    if (s == 0u) {
      invDx = invAll;
    } else {
      invDx = fe_mul(invAll, cArr[s - 1u]);
      invAll = fe_mul(invAll, fe_sub(jx, cx));
    }
    let dy = fe_sub(jy, cy);
    let lam = fe_mul(dy, invDx);
    let lam2 = fe_sqr(lam);
    let nx = fe_sub(fe_sub(lam2, cx), jx);
    let ny = fe_sub(fe_mul(lam, fe_sub(cx, nx)), cy);

    if (s == S - 1u) {
      nextX = nx; nextY = ny;
    } else {
      check_point(select(2u, 3u, (ny[0] & 1u) == 1u), nx, tid, s + 1u);
    }
  }

  for (var i: u32 = 0u; i < 8u; i = i + 1u) {
    ptX[base + i] = nextX[i];
    ptY[base + i] = nextY[i];
  }
}
`;

    const WGSL_KANGAROO_KERNEL = `
struct KangParams {
  numThreads: u32,
  stepsPerDispatch: u32,
  dpMaskLo: u32,
  dpMaskHi: u32,
};

@group(0) @binding(0) var<uniform> params: KangParams;
@group(0) @binding(1) var<storage, read_write> kangX: array<u32>;
@group(0) @binding(2) var<storage, read_write> kangY: array<u32>;
@group(0) @binding(3) var<storage, read_write> kangDist: array<u32>;
@group(0) @binding(4) var<storage, read> kangHerd: array<u32>;
@group(0) @binding(5) var<storage, read> kJumpX: array<u32>;
@group(0) @binding(6) var<storage, read> kJumpY: array<u32>;
@group(0) @binding(7) var<storage, read> kJumpDist: array<u32>;
@group(0) @binding(8) var<storage, read_write> dpOut: array<atomic<u32>>;

fn getP(i: u32) -> u32 {
  if (i == 0u) { return 0xFFFFFC2Fu; }
  if (i == 1u) { return 0xFFFFFFFEu; }
  return 0xFFFFFFFFu;
}
fn fe_gte_p(a: ptr<function, array<u32, 8>>) -> bool {
  for (var idx: i32 = 7; idx >= 0; idx = idx - 1) {
    let i = u32(idx); let pi = getP(i);
    if ((*a)[i] > pi) { return true; }
    if ((*a)[i] < pi) { return false; }
  }
  return true;
}
fn fe_sub_p(a: ptr<function, array<u32, 8>>) {
  var borrow: u32 = 0u;
  for (var i: u32 = 0u; i < 8u; i = i + 1u) {
    let ai = (*a)[i]; let pi = getP(i);
    let sub1 = ai - pi; let b1 = select(0u, 1u, ai < pi);
    let sub2 = sub1 - borrow; let b2 = select(0u, 1u, sub1 < borrow);
    (*a)[i] = sub2; borrow = b1 + b2;
  }
}
fn fe_sub(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
  var r: array<u32, 8>; var borrow: u32 = 0u;
  for (var i: u32 = 0u; i < 8u; i = i + 1u) {
    let s1 = a[i] - b[i]; let b1 = select(0u, 1u, a[i] < b[i]);
    let s2 = s1 - borrow; let b2 = select(0u, 1u, s1 < borrow);
    r[i] = s2; borrow = b1 + b2;
  }
  if (borrow != 0u) {
    var carry: u32 = 0u;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
      let pi = getP(i);
      let s1 = r[i] + pi; let c1 = select(0u, 1u, s1 < r[i]);
      let s2 = s1 + carry; let c2 = select(0u, 1u, s2 < s1);
      r[i] = s2; carry = c1 + c2;
    }
  }
  return r;
}
fn mul32_64(a: u32, b: u32) -> vec2<u32> {
  let a0 = a & 0xFFFFu; let a1 = a >> 16u; let b0 = b & 0xFFFFu; let b1 = b >> 16u;
  let p00 = a0 * b0; let p01 = a0 * b1; let p10 = a1 * b0; let p11 = a1 * b1;
  let mid1 = p01 + (p00 >> 16u); let mid2 = mid1 + p10;
  let c_mid = select(0u, 1u, mid2 < mid1);
  return vec2<u32>((p00 & 0xFFFFu) | (mid2 << 16u), p11 + (mid2 >> 16u) + (c_mid << 16u));
}
fn fe_reduce_512(t_in: array<u32, 16>) -> array<u32, 8> {
  var t = t_in; var r: array<u32, 8>;
  for (var i: u32 = 0u; i < 8u; i = i + 1u) { r[i] = t[i]; }
  var c977: u32 = 0u;
  for (var i: u32 = 0u; i < 8u; i = i + 1u) {
    let p = mul32_64(t[8u + i], 977u);
    let s1 = r[i] + p.x; let c1 = select(0u, 1u, s1 < r[i]);
    let s2 = s1 + c977; let c2 = select(0u, 1u, s2 < s1);
    r[i] = s2; c977 = p.y + c1 + c2;
  }
  var csh: u32 = 0u;
  for (var i: u32 = 1u; i < 8u; i = i + 1u) {
    let v = t[7u + i];
    let s1 = r[i] + v; let c1 = select(0u, 1u, s1 < r[i]);
    let s2 = s1 + csh; let c2 = select(0u, 1u, s2 < s1);
    r[i] = s2; csh = c1 + c2;
  }
  let top = c977 + csh; let ft = mul32_64(top, 977u); let f15 = mul32_64(t[15], 977u);
  let s0 = r[0] + ft.x; var cy0 = select(0u, 1u, s0 < r[0]);
  let s0b = s0 + f15.x; cy0 = cy0 + select(0u, 1u, s0b < s0); r[0] = s0b;
  let add1 = ft.y + cy0; var cy1 = select(0u, 1u, add1 < ft.y);
  let add1b = add1 + top; cy1 = cy1 + select(0u, 1u, add1b < add1);
  let add1c = add1b + f15.y; cy1 = cy1 + select(0u, 1u, add1c < add1b);
  let add1d = add1c + t[15]; cy1 = cy1 + select(0u, 1u, add1d < add1c);
  let s1 = r[1] + add1d; cy1 = cy1 + select(0u, 1u, s1 < r[1]); r[1] = s1;
  var cy2 = cy1;
  for (var i: u32 = 2u; i < 8u; i = i + 1u) {
    if (cy2 == 0u) { break; }
    let sn = r[i] + cy2; cy2 = select(0u, 1u, sn < r[i]); r[i] = sn;
  }
  if (cy2 != 0u) {
    let f2 = mul32_64(cy2, 977u); let sn0 = r[0] + f2.x; let c2b = select(0u, 1u, sn0 < r[0]); r[0] = sn0;
    let a1 = r[1] + f2.y; var c1w = select(0u, 1u, a1 < r[1]);
    let b1 = a1 + cy2; c1w = c1w + select(0u, 1u, b1 < a1);
    let d1 = b1 + c2b; c1w = c1w + select(0u, 1u, d1 < b1); r[1] = d1;
    for (var i: u32 = 2u; i < 8u; i = i + 1u) {
      if (c1w == 0u) { break; }
      let sn = r[i] + c1w; c1w = select(0u, 1u, sn < r[i]); r[i] = sn;
    }
  }
  if (fe_gte_p(&r)) { fe_sub_p(&r); }
  if (fe_gte_p(&r)) { fe_sub_p(&r); }
  return r;
}
fn fe_mul(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
  var t: array<u32, 16>;
  for (var k: u32 = 0u; k < 16u; k = k + 1u) { t[k] = 0u; }
  for (var i: u32 = 0u; i < 8u; i = i + 1u) {
    var carry: u32 = 0u; let ai = a[i];
    for (var j: u32 = 0u; j < 8u; j = j + 1u) {
      let p = mul32_64(ai, b[j]);
      let s1 = t[i + j] + p.x; let c1 = select(0u, 1u, s1 < t[i + j]);
      let s2 = s1 + carry; let c2 = select(0u, 1u, s2 < s1);
      t[i + j] = s2; carry = p.y + c1 + c2;
    }
    t[i + 8u] = t[i + 8u] + carry;
  }
  return fe_reduce_512(t);
}
fn fe_sqr(a: array<u32, 8>) -> array<u32, 8> { return fe_mul(a, a); }
fn fe_inv(x: array<u32, 8>) -> array<u32, 8> {
  var x2 = fe_mul(fe_sqr(x), x); var x3 = fe_mul(fe_sqr(x2), x);
  var x6 = x3; for (var i: u32 = 0u; i < 3u; i = i + 1u) { x6 = fe_sqr(x6); } x6 = fe_mul(x6, x3);
  var x9 = x6; for (var i: u32 = 0u; i < 3u; i = i + 1u) { x9 = fe_sqr(x9); } x9 = fe_mul(x9, x3);
  var x11 = x9; for (var i: u32 = 0u; i < 2u; i = i + 1u) { x11 = fe_sqr(x11); } x11 = fe_mul(x11, x2);
  var x22 = x11; for (var i: u32 = 0u; i < 11u; i = i + 1u) { x22 = fe_sqr(x22); } x22 = fe_mul(x22, x11);
  var x44 = x22; for (var i: u32 = 0u; i < 22u; i = i + 1u) { x44 = fe_sqr(x44); } x44 = fe_mul(x44, x22);
  var x88 = x44; for (var i: u32 = 0u; i < 44u; i = i + 1u) { x88 = fe_sqr(x88); } x88 = fe_mul(x88, x44);
  var x176 = x88; for (var i: u32 = 0u; i < 88u; i = i + 1u) { x176 = fe_sqr(x176); } x176 = fe_mul(x176, x88);
  var x220 = x176; for (var i: u32 = 0u; i < 44u; i = i + 1u) { x220 = fe_sqr(x220); } x220 = fe_mul(x220, x44);
  var x223 = x220; for (var i: u32 = 0u; i < 3u; i = i + 1u) { x223 = fe_sqr(x223); } x223 = fe_mul(x223, x3);
  var t = x223;
  for (var i: u32 = 0u; i < 23u; i = i + 1u) { t = fe_sqr(t); } t = fe_mul(t, x22);
  for (var i: u32 = 0u; i < 5u; i = i + 1u) { t = fe_sqr(t); } t = fe_mul(t, x);
  for (var i: u32 = 0u; i < 3u; i = i + 1u) { t = fe_sqr(t); } t = fe_mul(t, x2);
  for (var i: u32 = 0u; i < 2u; i = i + 1u) { t = fe_sqr(t); } t = fe_mul(t, x);
  return t;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tid = gid.x;
  if (tid >= params.numThreads) { return; }
  var kx: array<array<u32, 8>, 4>;
  var ky: array<array<u32, 8>, 4>;
  var kd: array<array<u32, 8>, 4>;
  var kh: array<u32, 4>;
  for (var k: u32 = 0u; k < 4u; k = k + 1u) {
    let kid = tid * 4u + k;
    let base = kid * 8u;
    kh[k] = kangHerd[kid];
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
      kx[k][i] = kangX[base + i];
      ky[k][i] = kangY[base + i];
      kd[k][i] = kangDist[base + i];
    }
  }
  for (var step: u32 = 0u; step < params.stepsPerDispatch; step = step + 1u) {
    var jIdx: array<u32, 4>;
    var dxArr: array<array<u32, 8>, 4>;
    var cArr: array<array<u32, 8>, 4>;
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
      let ji = kx[k][0] & 63u;
      jIdx[k] = ji;
      let jb = ji * 8u;
      var jx: array<u32, 8>;
      for (var i: u32 = 0u; i < 8u; i = i + 1u) { jx[i] = kJumpX[jb + i]; }
      dxArr[k] = fe_sub(jx, kx[k]);
      if (k == 0u) { cArr[0] = dxArr[0]; } else { cArr[k] = fe_mul(cArr[k - 1u], dxArr[k]); }
    }
    var invAll = fe_inv(cArr[3]);
    for (var kk: i32 = 3; kk >= 0; kk = kk - 1) {
      let k = u32(kk);
      var invDx: array<u32, 8>;
      if (k == 0u) {
        invDx = invAll;
      } else {
        invDx = fe_mul(invAll, cArr[k - 1u]);
        invAll = fe_mul(invAll, dxArr[k]);
      }
      let jb = jIdx[k] * 8u;
      var jx: array<u32, 8>; var jy: array<u32, 8>;
      for (var i: u32 = 0u; i < 8u; i = i + 1u) { jx[i] = kJumpX[jb + i]; jy[i] = kJumpY[jb + i]; }
      let dy = fe_sub(jy, ky[k]);
      let lam = fe_mul(dy, invDx);
      let nx = fe_sub(fe_sub(fe_sqr(lam), kx[k]), jx);
      let ny = fe_sub(fe_mul(lam, fe_sub(kx[k], nx)), ky[k]);
      kx[k] = nx; ky[k] = ny;
      var cy: u32 = 0u;
      for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let jdi = kJumpDist[jb + i];
        let s1 = kd[k][i] + jdi; let c1 = select(0u, 1u, s1 < kd[k][i]);
        let s2 = s1 + cy; let c2 = select(0u, 1u, s2 < s1);
        kd[k][i] = s2; cy = c1 + c2;
      }
      if ((nx[0] & params.dpMaskLo) == 0u && (nx[1] & params.dpMaskHi) == 0u) {
        let slot = atomicAdd(&dpOut[0], 1u);
        if (slot < 256u) {
          let ob = 1u + slot * 17u;
          atomicStore(&dpOut[ob], kh[k]);
          for (var i: u32 = 0u; i < 8u; i = i + 1u) {
            atomicStore(&dpOut[ob + 1u + i], nx[i]);
            atomicStore(&dpOut[ob + 9u + i], kd[k][i]);
          }
        }
      }
    }
  }
  for (var k: u32 = 0u; k < 4u; k = k + 1u) {
    let base = (tid * 4u + k) * 8u;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
      kangX[base + i] = kx[k][i];
      kangY[base + i] = ky[k][i];
      kangDist[base + i] = kd[k][i];
    }
  }
}
`;

    function modInv(a, p) {
        let t = 0n, newT = 1n;
        let r = p, newR = ((a % p) + p) % p;
        let tt = t, nt = newT, rr = r, nr = newR;
        while (nr !== 0n) {
            const q = rr / nr;
            const tmpT = tt - q * nt; tt = nt; nt = tmpT;
            const tmpR = rr - q * nr; rr = nr; nr = tmpR;
        }
        return tt < 0n ? tt + p : tt;
    }

    function pointAddAffine(x1, y1, x2, y2) {
        if (typeof RealPollardKangaroo !== 'undefined' && typeof RealPollardKangaroo.pointAddAffine === 'function') {
            return RealPollardKangaroo.pointAddAffine(x1, y1, x2, y2);
        }
        if (x1 === x2 && y1 === y2) {
            const lam = (3n * x1 * x1 * modInv(2n * y1, SECP_P)) % SECP_P;
            const x3 = (lam * lam - 2n * x1 + 2n * SECP_P) % SECP_P;
            const y3 = (lam * (x1 - x3 + SECP_P) - y1 + SECP_P) % SECP_P;
            return [x3, y3];
        }
        const lam = ((y2 - y1 + SECP_P) * modInv((x2 - x1 + SECP_P) % SECP_P, SECP_P)) % SECP_P;
        const x3 = (lam * lam - x1 - x2 + 2n * SECP_P) % SECP_P;
        const y3 = (lam * (x1 - x3 + SECP_P) - y1 + SECP_P) % SECP_P;
        return [x3, y3];
    }

    function scalarMul(gx, gy, k) {
        let rx = null, ry = null, ax = gx, ay = gy, n = k;
        while (n > 0n) {
            if (n & 1n) {
                if (rx === null) { rx = ax; ry = ay; }
                else { const r = pointAddAffine(rx, ry, ax, ay); rx = r[0]; ry = r[1]; }
            }
            n >>= 1n;
            if (n > 0n) {
                const d = pointAddAffine(ax, ay, ax, ay);
                ax = d[0]; ay = d[1];
            }
        }
        return [rx, ry];
    }

    function bigIntToU32x8(val, arr, offset) {
        let v = val;
        for (let i = 0; i < 8; i++) {
            arr[offset + i] = Number(v & 0xFFFFFFFFn);
            v >>= 32n;
        }
    }

    function hash160HexToU32x5(hexStr) {
        const clean = (hexStr || '').replace(/^0x/i, '').padStart(40, '0');
        const out = new Uint32Array(5);
        for (let i = 0; i < 5; i++) {
            out[i] = parseInt(clean.slice(i * 8, i * 8 + 8), 16) >>> 0;
        }
        return out;
    }

    function randomBigIntInRange(minVal, maxVal) {
        if (maxVal <= minVal) return minVal;
        const span = maxVal - minVal;
        const bits = span.toString(2).length;
        const bytes = Math.ceil(bits / 8);
        const buf = new Uint8Array(bytes);
        while (true) {
            crypto.getRandomValues(buf);
            let r = 0n;
            for (let i = 0; i < bytes; i++) r = (r << 8n) | BigInt(buf[i]);
            const mask = (1n << BigInt(bits)) - 1n;
            r &= mask;
            if (r <= span) return minVal + r;
        }
    }

    class SiriusWebGpuScanner {
        constructor() {
            this.device = null;
            this.pipe = null;
            this.bg = null;
            this.pBuf = null;
            this.ptXBuf = null;
            this.ptYBuf = null;
            this.jXBuf = null;
            this.jYBuf = null;
            this.outBuf = null;
            this.readBuf = null;
            this.ptXArr = new Uint32Array(MAX_THREADS * 8);
            this.ptYArr = new Uint32Array(MAX_THREADS * 8);
            this.activeThreads = DEFAULT_THREADS;
            this.intensity = 'MEDIUM';
            this._lastKeysPerDisp = 0;
            this.strideTable = null; // 128 precomputed multiples of (64 * G)
            this.kangPipe = null;
            this.kangBg = null;
            this.kangPBuf = null;
            this.kangXBuf = null;
            this.kangYBuf = null;
            this.kangDistBuf = null;
            this.kangHerdBuf = null;
            this.kJumpXBuf = null;
            this.kJumpYBuf = null;
            this.kJumpDistBuf = null;
            this.dpOutBuf = null;
            this.dpReadBuf = null;
            this.ready = false;
            this.unavailable = false;
            this._initPromise = null;
            this._runPromise = null;
            this.running = false;
            this._sessionId = 0;
            this._adaptiveBatch = 2;
            this.statusText = 'Hazır';
            this.gpuSpeed = 0;
        }

        setIntensity(level) {
            const key = String(level || 'MEDIUM').toUpperCase();
            if (INTENSITY_MAP[key]) {
                this.intensity = key;
                this.activeThreads = INTENSITY_MAP[key];
            }
            return this.activeThreads;
        }

        _syncDispAdv() {
            const kpd = this.activeThreads * STEPS_PER_THREAD;
            if (!this.device || !this.jXBuf || !this.jYBuf || this._lastKeysPerDisp === kpd) return;
            const dispAdv = scalarMul(GX, GY, BigInt(kpd));
            const x8 = new Uint32Array(8);
            const y8 = new Uint32Array(8);
            bigIntToU32x8(dispAdv[0], x8, 0);
            bigIntToU32x8(dispAdv[1], y8, 0);
            this.device.queue.writeBuffer(this.jXBuf, (STEPS_PER_THREAD - 1) * 32, x8);
            this.device.queue.writeBuffer(this.jYBuf, (STEPS_PER_THREAD - 1) * 32, y8);
            this._lastKeysPerDisp = kpd;
        }

        static isSupported() {
            return typeof navigator !== 'undefined' && Boolean(navigator.gpu);
        }

        async init() {
            if (this.ready) return true;
            if (this.unavailable) return false;
            if (this._initPromise) return this._initPromise;

            this._initPromise = (async () => {
                try {
                    if (!SiriusWebGpuScanner.isSupported()) {
                        this.unavailable = true;
                        this.statusText = 'Desteklenmiyor';
                        return false;
                    }
                    this.statusText = 'Başlatılıyor...';
                    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
                    if (!adapter) {
                        this.unavailable = true;
                        this.statusText = 'GPU Adaptörü Yok';
                        return false;
                    }
                    const device = await adapter.requestDevice();
                    this.device = device;

                    this.statusText = 'Shader Derleniyor...';
                    const mod = device.createShaderModule({ code: WGSL_KERNEL });
                    const info = await mod.getCompilationInfo();
                    const errs = info.messages.filter(m => m.type === 'error');
                    if (errs.length > 0) {
                        console.error('[WebGPU] WGSL Compilation Error:', errs.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join(' | '));
                        this.unavailable = true;
                        this.statusText = 'Shader Hatası';
                        return false;
                    }

                    this.pipe = await device.createComputePipelineAsync({
                        layout: 'auto',
                        compute: { module: mod, entryPoint: 'main' }
                    });

                    // Precompute 1G .. 31G, and slot 31 = (NUM_THREADS * STEPS_PER_THREAD) * G to advance to next dispatch without overlap
                    const jumpXArr = new Uint32Array(STEPS_PER_THREAD * 8);
                    const jumpYArr = new Uint32Array(STEPS_PER_THREAD * 8);
                    let cur = [GX, GY];
                    for (let s = 0; s < STEPS_PER_THREAD - 1; s++) {
                        bigIntToU32x8(cur[0], jumpXArr, s * 8);
                        bigIntToU32x8(cur[1], jumpYArr, s * 8);
                        cur = pointAddAffine(cur[0], cur[1], GX, GY);
                    }
                    const dispAdv = scalarMul(GX, GY, BigInt(this.activeThreads * STEPS_PER_THREAD));
                    bigIntToU32x8(dispAdv[0], jumpXArr, (STEPS_PER_THREAD - 1) * 8);
                    bigIntToU32x8(dispAdv[1], jumpYArr, (STEPS_PER_THREAD - 1) * 8);
                    this._lastKeysPerDisp = this.activeThreads * STEPS_PER_THREAD;

                    // Precompute strideTable[0..127] = (j+1) * (64 * G) for fast Montgomery batch anchor init
                    const stride1 = scalarMul(GX, GY, BigInt(STEPS_PER_THREAD));
                    this.strideTable = new Array(128);
                    let sCur = stride1;
                    for (let j = 0; j < 128; j++) {
                        this.strideTable[j] = sCur;
                        if (j < 127) sCur = pointAddAffine(sCur[0], sCur[1], stride1[0], stride1[1]);
                    }

                    this.pBuf = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
                    this.ptXBuf = device.createBuffer({ size: MAX_THREADS * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
                    this.ptYBuf = device.createBuffer({ size: MAX_THREADS * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
                    this.jXBuf = device.createBuffer({ size: jumpXArr.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
                    this.jYBuf = device.createBuffer({ size: jumpYArr.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
                    this.outBuf = device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
                    this.readBuf = device.createBuffer({ size: 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
                    this.bloomBuf = device.createBuffer({ size: BLOOM_WORDS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

                    // Kangaroo GPU Pipeline & Buffers
                    const kMod = device.createShaderModule({ code: WGSL_KANGAROO_KERNEL });
                    const kInfo = await kMod.getCompilationInfo();
                    if (!kInfo.messages.some(m => m.type === 'error')) {
                        this.kangPipe = await device.createComputePipelineAsync({
                            layout: 'auto',
                            compute: { module: kMod, entryPoint: 'main' }
                        });
                        const maxKangs = MAX_THREADS; // 4096 threads * 4 = 16384 max kangaroos
                        this.kangPBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
                        this.kangXBuf = device.createBuffer({ size: maxKangs * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
                        this.kangYBuf = device.createBuffer({ size: maxKangs * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
                        this.kangDistBuf = device.createBuffer({ size: maxKangs * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
                        this.kangHerdBuf = device.createBuffer({ size: maxKangs * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
                        this.kJumpXBuf = device.createBuffer({ size: 64 * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
                        this.kJumpYBuf = device.createBuffer({ size: 64 * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
                        this.kJumpDistBuf = device.createBuffer({ size: 64 * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
                        const dpBytes = (1 + MAX_DP_OUT * 17) * 4;
                        this.dpOutBuf = device.createBuffer({ size: dpBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
                        this.dpReadBuf = device.createBuffer({ size: dpBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
                        this.kangBg = device.createBindGroup({
                            layout: this.kangPipe.getBindGroupLayout(0),
                            entries: [
                                { binding: 0, resource: { buffer: this.kangPBuf } },
                                { binding: 1, resource: { buffer: this.kangXBuf } },
                                { binding: 2, resource: { buffer: this.kangYBuf } },
                                { binding: 3, resource: { buffer: this.kangDistBuf } },
                                { binding: 4, resource: { buffer: this.kangHerdBuf } },
                                { binding: 5, resource: { buffer: this.kJumpXBuf } },
                                { binding: 6, resource: { buffer: this.kJumpYBuf } },
                                { binding: 7, resource: { buffer: this.kJumpDistBuf } },
                                { binding: 8, resource: { buffer: this.dpOutBuf } }
                            ]
                        });
                    }

                    device.queue.writeBuffer(this.jXBuf, 0, jumpXArr);
                    device.queue.writeBuffer(this.jYBuf, 0, jumpYArr);

                    this.bg = device.createBindGroup({
                        layout: this.pipe.getBindGroupLayout(0),
                        entries: [
                            { binding: 0, resource: { buffer: this.pBuf } },
                            { binding: 1, resource: { buffer: this.ptXBuf } },
                            { binding: 2, resource: { buffer: this.ptYBuf } },
                            { binding: 3, resource: { buffer: this.jXBuf } },
                            { binding: 4, resource: { buffer: this.jYBuf } },
                            { binding: 5, resource: { buffer: this.outBuf } },
                            { binding: 6, resource: { buffer: this.bloomBuf } }
                        ]
                    });

                    this.ready = true;
                    this.statusText = 'Hazır';
                    return true;
                } catch (e) {
                    console.warn('[WebGPU] Init failed, falling back to CPU:', e);
                    this.unavailable = true;
                    this.statusText = 'Hata';
                    return false;
                }
            })();

            return this._initPromise;
        }

        _uploadBloom(hash160List) {
            if (!Array.isArray(hash160List) || hash160List.length === 0) return false;
            const sig = hash160List.length + ':' + hash160List[0] + ':' + hash160List[hash160List.length - 1];
            if (this._lastBloomSig === sig) return true;
            const arr = new Uint32Array(BLOOM_WORDS);
            for (let i = 0; i < hash160List.length; i++) {
                const w = hash160HexToU32x5(hash160List[i]);
                const b0 = (w[0] ^ Math.imul(w[3], 0x9E3779B1)) & BLOOM_MASK;
                const b1 = (w[1] ^ Math.imul(w[4], 0x85EBCA77)) & BLOOM_MASK;
                const b2 = Math.imul((w[2] ^ w[0]), 0xC2B2AE3D) & BLOOM_MASK;
                const b3 = Math.imul((w[3] ^ w[1] ^ w[4]), 0x27D4EB2F) & BLOOM_MASK;
                arr[b0 >>> 5] |= (1 << (b0 & 31));
                arr[b1 >>> 5] |= (1 << (b1 & 31));
                arr[b2 >>> 5] |= (1 << (b2 & 31));
                arr[b3 >>> 5] |= (1 << (b3 & 31));
            }
            this.device.queue.writeBuffer(this.bloomBuf, 0, arr);
            this._lastBloomSig = sig;
            return true;
        }

        _fillAnchors(startKey, numThreads) {
            const CHUNK = 128;
            const dx = new Array(CHUNK - 1);
            const c = new Array(CHUNK - 1);
            let basePt = scalarMul(GX, GY, startKey);

            for (let ch = 0; ch < numThreads / CHUNK; ch++) {
                const tBase = ch * CHUNK;
                bigIntToU32x8(basePt[0], this.ptXArr, tBase * 8);
                bigIntToU32x8(basePt[1], this.ptYArr, tBase * 8);

                // Batch-invert (strideTable[j].x - basePt.x) for j = 0 .. CHUNK-2 (points 1..127 of chunk)
                let fallback = false;
                for (let j = 0; j < CHUNK - 1; j++) {
                    const diff = (this.strideTable[j][0] - basePt[0] + SECP_P) % SECP_P;
                    if (diff === 0n) { fallback = true; break; }
                    dx[j] = diff;
                    c[j] = (j === 0) ? diff : (c[j - 1] * diff) % SECP_P;
                }

                if (fallback) {
                    for (let j = 0; j < CHUNK - 1; j++) {
                        const pt = pointAddAffine(basePt[0], basePt[1], this.strideTable[j][0], this.strideTable[j][1]);
                        bigIntToU32x8(pt[0], this.ptXArr, (tBase + 1 + j) * 8);
                        bigIntToU32x8(pt[1], this.ptYArr, (tBase + 1 + j) * 8);
                    }
                    basePt = pointAddAffine(basePt[0], basePt[1], this.strideTable[CHUNK - 1][0], this.strideTable[CHUNK - 1][1]);
                } else {
                    let invAll = modInv(c[CHUNK - 2], SECP_P);
                    const bx = basePt[0], by = basePt[1];
                    for (let j = CHUNK - 2; j >= 0; j--) {
                        const invDx = (j === 0) ? invAll : (invAll * c[j - 1]) % SECP_P;
                        if (j > 0) invAll = (invAll * dx[j]) % SECP_P;

                        const sx = this.strideTable[j][0], sy = this.strideTable[j][1];
                        const lam = ((sy - by + SECP_P) * invDx) % SECP_P;
                        const nx = (lam * lam - bx - sx + 2n * SECP_P) % SECP_P;
                        const ny = (lam * (bx - nx + SECP_P) - by + SECP_P) % SECP_P;
                        bigIntToU32x8(nx, this.ptXArr, (tBase + 1 + j) * 8);
                        bigIntToU32x8(ny, this.ptYArr, (tBase + 1 + j) * 8);
                    }
                    basePt = pointAddAffine(bx, by, this.strideTable[CHUNK - 1][0], this.strideTable[CHUNK - 1][1]);
                }
            }

            this.device.queue.writeBuffer(this.ptXBuf, 0, this.ptXArr, 0, numThreads * 8);
            this.device.queue.writeBuffer(this.ptYBuf, 0, this.ptYArr, 0, numThreads * 8);
        }

        stop() {
            this.running = false;
            this._sessionId++;
            this.gpuSpeed = 0;
            if (this.ready) this.statusText = 'Hazır';
        }

        async start(cfg, callbacks) {
            this.stop();
            const mySession = ++this._sessionId;
            this.running = true;
            this.statusText = 'Aktif';

            // Wait for any in-flight mapAsync/unmap from the previous session to settle
            const prevRun = this._runPromise;
            if (prevRun) {
                try { await prevRun; } catch (_) {}
            }
            if (!this.running || mySession !== this._sessionId) return;

            const runTask = (async () => {
                try {
                    const ok = await this.init();
                    if (!ok || !this.running || mySession !== this._sessionId) return;

                    // Build target list (filter compressed targets with valid hash160)
                    let targets = [];
                    if (cfg && cfg.isCircularMode && Array.isArray(cfg.unsolvedList) && cfg.unsolvedList.length > 0) {
                        targets = cfg.unsolvedList.filter(p => !p.isUncompressed && p.targetHash160);
                    } else if (cfg && cfg.activeTarget && !cfg.activeTarget.isUncompressed && cfg.activeTarget.targetHash160) {
                        targets = [cfg.activeTarget];
                    }
                    if (targets.length === 0) {
                        return;
                    }

                    const hasBloom = Boolean(cfg && Array.isArray(cfg.customHash160List) && cfg.customHash160List.length > 0 && this._uploadBloom(cfg.customHash160List));
                    const bloomMaskVal = hasBloom ? BLOOM_MASK : 0;
                    const useGlvSymVal = hasBloom ? 1 : 0;

                    const numThreads = this.activeThreads || DEFAULT_THREADS;
                    const keysPerDisp = numThreads * STEPS_PER_THREAD;
                    this._syncDispAdv();

                    let targetIdx = 0;
                    const paramArr = new Uint32Array(12);
                    const zeroHit = new Uint32Array(8);
                    const wgCount = Math.ceil(numThreads / 64);

                    this.statusText = 'Aktif';

                    while (this.running && mySession === this._sessionId) {
                        const tgt = targets[targetIdx % targets.length];
                        let tStart = BigInt('0x' + (tgt.start || '1').replace(/^0x/i, ''));
                        let tEnd = BigInt('0x' + (tgt.end || '1').replace(/^0x/i, ''));
                        if (!cfg.isCircularMode && targets.length === 1) {
                            if (cfg.isDualSliderActive && cfg.sliderMinHex && cfg.sliderMaxHex) {
                                const sMin = BigInt('0x' + String(cfg.sliderMinHex).replace(/^0x/i, ''));
                                const sMax = BigInt('0x' + String(cfg.sliderMaxHex).replace(/^0x/i, ''));
                                if (sMin > tStart) tStart = sMin;
                                if (sMax < tEnd) tEnd = sMax;
                            }
                            if (cfg.activeRangeMode === 'HEX' && cfg.userHexStart && cfg.userHexEnd) {
                                const hMin = BigInt('0x' + String(cfg.userHexStart).replace(/^0x/i, ''));
                                const hMax = BigInt('0x' + String(cfg.userHexEnd).replace(/^0x/i, ''));
                                if (hMin > tStart) tStart = hMin;
                                if (hMax < tEnd) tEnd = hMax;
                            }
                        }
                        const minK = tStart < 1n ? 1n : tStart;
                        const maxK = tEnd >= SECP_N ? (SECP_N - 1n) : (tEnd < minK ? minK : tEnd);
                        const rangeLen = maxK >= minK ? (maxK - minK + 1n) : 1n;
                        if (maxK <= BigInt(STEPS_PER_THREAD)) {
                            // Tiny keys <= 64 (e.g. Puzzles #1-#6) are solved instantaneously on CPU and would hit dx==0 in batch jump table
                            if (targets.length > 1) { targetIdx++; continue; }
                            return;
                        }
                        const hWords = hash160HexToU32x5(tgt.targetHash160);

                        // For tiny ranges (<= 1 dispatch), start at minK; for small/medium ranges (<= 2^28), start at midpoint and wrap to minK so GPU never overlaps CPU worker #0 at minK
                        const isSmallRange = rangeLen <= 268435456n;
                        const startAtMid = isSmallRange && rangeLen > BigInt(keysPerDisp);
                        let anchorBase = !isSmallRange
                            ? randomBigIntInRange(minK, maxK > BigInt(keysPerDisp) ? (maxK - BigInt(keysPerDisp)) : minK)
                            : (startAtMid ? (minK + (rangeLen >> 1n)) : minK);
                        if (anchorBase <= BigInt(STEPS_PER_THREAD)) {
                            anchorBase = BigInt(STEPS_PER_THREAD) + 1n;
                        } else if (anchorBase === BigInt(keysPerDisp)) {
                            anchorBase = BigInt(keysPerDisp) + 1n;
                        }
                        let wrappedToStart = false;

                        this._fillAnchors(anchorBase, numThreads);
                        let dispatchesSinceReanchor = 0;
                        const MAX_DISPATCHES_PER_ANCHOR = isSmallRange ? (Math.ceil(Number(rangeLen) / keysPerDisp) + 2) : 1600;

                        while (this.running && mySession === this._sessionId) {
                            const targetBatch = Math.min(MAX_BATCH_DISPATCHES, Math.max(1, this._adaptiveBatch || 2));
                            const batchDispatches = isSmallRange
                                ? Math.min(targetBatch, Math.max(1, Math.ceil(Number(maxK - anchorBase + 1n) / keysPerDisp)))
                                : targetBatch;

                            const t0 = performance.now();
                            this.device.queue.writeBuffer(this.outBuf, 0, zeroHit);

                            for (let d = 0; d < batchDispatches; d++) {
                                paramArr[0] = hWords[0];
                                paramArr[1] = hWords[1];
                                paramArr[2] = hWords[2];
                                paramArr[3] = hWords[3];
                                paramArr[4] = hWords[4];
                                paramArr[5] = STEPS_PER_THREAD;
                                paramArr[6] = numThreads;
                                paramArr[7] = d;
                                paramArr[8] = bloomMaskVal;
                                paramArr[9] = useGlvSymVal;
                                paramArr[10] = 0;
                                paramArr[11] = 0;
                                this.device.queue.writeBuffer(this.pBuf, 0, paramArr);

                                const enc = this.device.createCommandEncoder();
                                const pass = enc.beginComputePass();
                                pass.setPipeline(this.pipe);
                                pass.setBindGroup(0, this.bg);
                                pass.dispatchWorkgroups(wgCount);
                                pass.end();
                                if (d === batchDispatches - 1) {
                                    enc.copyBufferToBuffer(this.outBuf, 0, this.readBuf, 0, 32);
                                }
                                this.device.queue.submit([enc.finish()]);
                            }

                            await this.readBuf.mapAsync(GPUMapMode.READ);
                            if (!this.running || mySession !== this._sessionId) {
                                try { this.readBuf.unmap(); } catch (_) {}
                                return;
                            }
                            const hit = new Uint32Array(this.readBuf.getMappedRange().slice(0));
                            this.readBuf.unmap();
                            const dt = performance.now() - t0;
                            const batchKeys = batchDispatches * keysPerDisp;
                            const effCheckedKeys = useGlvSymVal ? (batchKeys * 4) : batchKeys;
                            if (dt > 0) {
                                this.gpuSpeed = Math.round(effCheckedKeys / (dt / 1000));
                                if (!isSmallRange) {
                                    if (dt < 28 && this._adaptiveBatch < MAX_BATCH_DISPATCHES) {
                                        this._adaptiveBatch = Math.min(MAX_BATCH_DISPATCHES, this._adaptiveBatch + 1);
                                    } else if (dt > 85 && this._adaptiveBatch > 1) {
                                        this._adaptiveBatch = Math.max(1, this._adaptiveBatch >> 1);
                                    }
                                }
                            }

                            const decodeHitKey = (hTid, hPacked, hDisp) => {
                                const stepVal = hPacked & 0xFFFF;
                                const subKind = (hPacked >>> 16) & 0xFFFF;
                                const baseKey = anchorBase + BigInt(hDisp) * BigInt(keysPerDisp) + BigInt(hTid) * BigInt(STEPS_PER_THREAD) + BigInt(stepVal);
                                let foundKey = baseKey;
                                if (subKind === 1) foundKey = SECP_N - baseKey;
                                else if (subKind === 2) foundKey = (SECP_LAMBDA * baseKey) % SECP_N;
                                else if (subKind === 3) foundKey = SECP_N - ((SECP_LAMBDA * baseKey) % SECP_N);
                                return { foundKey, subKind };
                            };

                            if (hit[0] === 1) {
                                const { foundKey, subKind } = decodeHitKey(hit[1], hit[2], hit[3]);
                                if (foundKey >= 1n && foundKey < SECP_N && (subKind !== 0 || (foundKey >= minK && foundKey <= maxK))) {
                                    this.stop();
                                    if (callbacks && typeof callbacks.onWin === 'function') {
                                        callbacks.onWin(foundKey, tgt);
                                    }
                                    return;
                                }
                            } else if (hit[4] === 2 && callbacks && typeof callbacks.onBloomHit === 'function') {
                                const { foundKey } = decodeHitKey(hit[5], hit[6], hit[7]);
                                if (callbacks.onBloomHit(foundKey)) {
                                    this.stop();
                                    return;
                                }
                            }

                            const lastKey = anchorBase + BigInt(batchKeys - 1);
                            if (callbacks && typeof callbacks.onProgress === 'function') {
                                callbacks.onProgress(effCheckedKeys, lastKey > maxK ? maxK : lastKey, this.gpuSpeed);
                            }

                            anchorBase += BigInt(batchKeys);
                            dispatchesSinceReanchor += batchDispatches;

                            if (isSmallRange && anchorBase > maxK && startAtMid && !wrappedToStart) {
                                wrappedToStart = true;
                                anchorBase = minK;
                                this._fillAnchors(anchorBase, numThreads);
                            } else if (anchorBase > maxK || dispatchesSinceReanchor >= MAX_DISPATCHES_PER_ANCHOR) {
                                if (targets.length > 1) targetIdx++;
                                break; // Re-anchor (or switch puzzle in circular mode)
                            }
                            // mapAsync already yields during GPU work; only yield extra tick if batch took >= 60ms
                            if (dt >= 60) {
                                await new Promise(r => setTimeout(r, 0));
                            }
                        }
                    }
                } catch (err) {
                    console.warn('[WebGPU] Run loop error:', err);
                } finally {
                    try {
                        if (this.readBuf && this.readBuf.mapState === 'mapped') this.readBuf.unmap();
                    } catch (_) {}
                    if (mySession === this._sessionId) {
                        this.running = false;
                        this.gpuSpeed = 0;
                        if (this.ready) this.statusText = 'Hazır';
                    }
                }
            })();
            this._runPromise = runTask;
            await runTask;
        }

        async startKangaroo(kangEngine, callbacks) {
            if (!kangEngine || !kangEngine.targetPoint || !kangEngine.jumps || kangEngine.jumps.length < 64) return;
            this.stop();
            const mySession = ++this._sessionId;
            this.running = true;
            this.statusText = 'Kangaroo Aktif';

            const prevRun = this._runPromise;
            if (prevRun) {
                try { await prevRun; } catch (_) {}
            }
            if (!this.running || mySession !== this._sessionId) return;

            const runTask = (async () => {
                try {
                    const ok = await this.init();
                    if (!ok || !this.kangPipe || !this.running || mySession !== this._sessionId) return;

                    const kThreads = Math.min(4096, Math.max(256, (this.activeThreads || DEFAULT_THREADS) >> 2));
                    const totalKangs = kThreads * KANG_PER_THREAD;

                    // Upload 64 jump points & distances from RealPollardKangaroo
                    const jxArr = new Uint32Array(64 * 8);
                    const jyArr = new Uint32Array(64 * 8);
                    const jdArr = new Uint32Array(64 * 8);
                    for (let i = 0; i < 64; i++) {
                        bigIntToU32x8(kangEngine.jumpX[i], jxArr, i * 8);
                        bigIntToU32x8(kangEngine.jumpY[i], jyArr, i * 8);
                        bigIntToU32x8(kangEngine.jumps[i], jdArr, i * 8);
                    }
                    this.device.queue.writeBuffer(this.kJumpXBuf, 0, jxArr);
                    this.device.queue.writeBuffer(this.kJumpYBuf, 0, jyArr);
                    this.device.queue.writeBuffer(this.kJumpDistBuf, 0, jdArr);

                    // Initialize Tame & Wild GPU kangaroos
                    const kxArr = new Uint32Array(totalKangs * 8);
                    const kyArr = new Uint32Array(totalKangs * 8);
                    const kdArr = new Uint32Array(totalKangs * 8);
                    const khArr = new Uint32Array(totalKangs);
                    const tX = BigInt('0x' + kangEngine.targetPoint.getX().toString(16));
                    const tY = BigInt('0x' + kangEngine.targetPoint.getY().toString(16));
                    const span = kangEngine.span > 0n ? kangEngine.span : 1n;
                    const halfSpan = span >> 1n;
                    const midPt = kangEngine.rangeMin + halfSpan;

                    for (let k = 0; k < totalKangs; k++) {
                        const isWild = (k & 1) === 1;
                        khArr[k] = isWild ? 1 : 0;
                        const jitter = (BigInt(k + 1) * (span / BigInt(totalKangs + 2)));
                        if (!isWild) {
                            const d = (midPt + jitter) % SECP_N;
                            const pt = scalarMul(GX, GY, d);
                            bigIntToU32x8(pt[0], kxArr, k * 8);
                            bigIntToU32x8(pt[1], kyArr, k * 8);
                            bigIntToU32x8(d, kdArr, k * 8);
                        } else {
                            const off = jitter;
                            const offPt = scalarMul(GX, GY, off > 0n ? off : 1n);
                            const pt = pointAddAffine(tX, tY, offPt[0], offPt[1]);
                            bigIntToU32x8(pt[0], kxArr, k * 8);
                            bigIntToU32x8(pt[1], kyArr, k * 8);
                            bigIntToU32x8(off > 0n ? off : 1n, kdArr, k * 8);
                        }
                    }
                    this.device.queue.writeBuffer(this.kangXBuf, 0, kxArr);
                    this.device.queue.writeBuffer(this.kangYBuf, 0, kyArr);
                    this.device.queue.writeBuffer(this.kangDistBuf, 0, kdArr);
                    this.device.queue.writeBuffer(this.kangHerdBuf, 0, khArr);

                    const dpMask = kangEngine.dpMask || 0xFFn;
                    const dpMaskLo = Number(dpMask & 0xFFFFFFFFn) >>> 0;
                    const dpMaskHi = Number((dpMask >> 32n) & 0xFFFFFFFFn) >>> 0;
                    const kParam = new Uint32Array([kThreads, KANG_STEPS_PER_DISPATCH, dpMaskLo, dpMaskHi]);
                    this.device.queue.writeBuffer(this.kangPBuf, 0, kParam);

                    const zeroCount = new Uint32Array([0]);
                    const wgCount = Math.ceil(kThreads / 64);
                    const dpBytes = (1 + MAX_DP_OUT * 17) * 4;
                    const stepsPerDisp = totalKangs * KANG_STEPS_PER_DISPATCH;

                    this.statusText = 'Kangaroo Aktif';

                    while (this.running && mySession === this._sessionId && !kangEngine.isSolved) {
                        const t0 = performance.now();
                        this.device.queue.writeBuffer(this.dpOutBuf, 0, zeroCount);
                        const enc = this.device.createCommandEncoder();
                        const pass = enc.beginComputePass();
                        pass.setPipeline(this.kangPipe);
                        pass.setBindGroup(0, this.kangBg);
                        pass.dispatchWorkgroups(wgCount);
                        pass.end();
                        enc.copyBufferToBuffer(this.dpOutBuf, 0, this.dpReadBuf, 0, dpBytes);
                        this.device.queue.submit([enc.finish()]);

                        await this.dpReadBuf.mapAsync(GPUMapMode.READ);
                        if (!this.running || mySession !== this._sessionId) {
                            try { this.dpReadBuf.unmap(); } catch (_) {}
                            return;
                        }
                        const out = new Uint32Array(this.dpReadBuf.getMappedRange().slice(0));
                        this.dpReadBuf.unmap();

                        const dt = performance.now() - t0;
                        if (dt > 0) this.gpuSpeed = Math.round(stepsPerDisp / (dt / 1000));

                        const count = Math.min(MAX_DP_OUT, out[0]);
                        for (let idx = 0; idx < count; idx++) {
                            const ob = 1 + idx * 17;
                            const herdStr = out[ob] === 1 ? 'WILD' : 'TAME';
                            let xBig = 0n, dBig = 0n;
                            for (let w = 7; w >= 0; w--) {
                                xBig = (xBig << 32n) | BigInt(out[ob + 1 + w]);
                                dBig = (dBig << 32n) | BigInt(out[ob + 9 + w]);
                            }
                            if (callbacks && typeof callbacks.onTrap === 'function') {
                                callbacks.onTrap(xBig, herdStr, dBig);
                            }
                            if (kangEngine.isSolved) break;
                        }

                        if (callbacks && typeof callbacks.onProgress === 'function') {
                            callbacks.onProgress(stepsPerDisp, this.gpuSpeed);
                        }
                        if (kangEngine.isSolved) {
                            this.stop();
                            return;
                        }
                        if (dt >= 50) await new Promise(r => setTimeout(r, 0));
                    }
                } catch (err) {
                    console.warn('[WebGPU Kangaroo] Run loop error:', err);
                } finally {
                    try {
                        if (this.dpReadBuf && this.dpReadBuf.mapState === 'mapped') this.dpReadBuf.unmap();
                    } catch (_) {}
                    if (mySession === this._sessionId) {
                        this.running = false;
                        this.gpuSpeed = 0;
                        if (this.ready) this.statusText = 'Hazır';
                    }
                }
            })();
            this._runPromise = runTask;
            await runTask;
        }
    }

    global.SiriusWebGpuScanner = SiriusWebGpuScanner;
    global.siriusWebGpu = new SiriusWebGpuScanner();
})(typeof window !== 'undefined' ? window : globalThis);
