/**
 * webgpu_kangaroo.js — Yüksek Performanslı WebGPU Pollard's Kangaroo (Lambda) Motoru
 *
 * Mimari:
 *   - %100 WebGPU Compute Shader (WGSL 1.0)
 *   - Montgomery Batch Inversion ile 32 iş parçacıklı saf Affine Nokta Toplaması
 *   - Her iş parçacığı bağımsız bir Kanguru (Tame veya Wild) yürütür
 *   - GPU içinde otomatik Ayırt Edici Nokta (Distinguished Point / Tuzak) tespiti
 *   - Sıfır CPU yükü, saniyede on milyonlarca eliptik eğri sıçraması
 *   - Çarpışma anında anında anahtar türetimi: d = (d_Tame - d_Wild) mod n
 */

const SECP256K1_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141n;

const WGSL_KANGAROO_SHADER = `
struct U256 {
    l: array<u32, 8>,
}

struct KangarooState {
    x: U256,
    y: U256,
    dist: U256,
    herd: u32,       // 0 = TAME, 1 = WILD
    id: u32,
    _pad0: u32,
    _pad1: u32,
}

struct JumpPoint {
    x: U256,
    y: U256,
    dist: U256,
}

struct KangarooParams {
    num_kangaroos: u32,
    steps_per_call: u32,
    dp_mask_lo: u32,
    dp_mask_hi: u32,
    max_traps: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct TrapRecord {
    x: U256,
    dist: U256,
    herd: u32,
    parity: u32,
    kangaroo_id: u32,
    _pad: u32,
}

struct TrapHeader {
    trap_count: atomic<u32>,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> states: array<KangarooState>;
@group(0) @binding(1) var<storage, read>       jumps: array<JumpPoint>;
@group(0) @binding(2) var<uniform>             params: KangarooParams;
@group(0) @binding(3) var<storage, read_write> trap_header: TrapHeader;
@group(0) @binding(4) var<storage, read_write> trap_records: array<TrapRecord>;

var<workgroup> shared_dx: array<U256, 32>;
var<workgroup> shared_c: array<U256, 32>;
var<workgroup> shared_inv: array<U256, 32>;

// ----------------------------------------------------------------------------
// 256-bit Temel Matematik Fonksiyonları
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

    let mid = (ll >> 16u) + (lh & 0xFFFFu) + (hl & 0xFFFFu);
    let lo  = (ll & 0xFFFFu) | (mid << 16u);
    let hi  = hh + (lh >> 16u) + (hl >> 16u) + (mid >> 16u);
    return vec2<u32>(lo, hi);
}

fn add32c(a: u32, b: u32, c: u32) -> vec2<u32> {
    let s1 = a + b;
    let c1 = select(0u, 1u, s1 < a);
    let s2 = s1 + c;
    let c2 = select(0u, 1u, s2 < s1);
    return vec2<u32>(s2, c1 + c2);
}

fn u256_zero() -> U256 {
    var r: U256;
    for (var i = 0u; i < 8u; i++) { r.l[i] = 0u; }
    return r;
}

fn u256_one() -> U256 {
    var r: U256;
    r.l[0] = 1u;
    for (var i = 1u; i < 8u; i++) { r.l[i] = 0u; }
    return r;
}

fn u256_is_zero(a: U256) -> bool {
    for (var i = 0u; i < 8u; i++) {
        if (a.l[i] != 0u) { return false; }
    }
    return true;
}

fn u256_lt(a: U256, b: U256) -> bool {
    for (var i = 7u; i >= 1u; i--) {
        if (a.l[i] < b.l[i]) { return true; }
        if (a.l[i] > b.l[i]) { return false; }
    }
    return a.l[0] < b.l[0];
}

fn u256_add_raw(a: U256, b: U256) -> U256 {
    var r: U256;
    var carry = 0u;
    for (var i = 0u; i < 8u; i++) {
        let res = add32c(a.l[i], b.l[i], carry);
        r.l[i] = res.x;
        carry = res.y;
    }
    return r;
}

fn u256_sub_raw(a: U256, b: U256) -> U256 {
    var r: U256;
    var borrow = 0u;
    for (var i = 0u; i < 8u; i++) {
        let bi = b.l[i];
        let diff = a.l[i] - bi - borrow;
        borrow = select(0u, 1u, a.l[i] < (bi + borrow) || (borrow == 1u && bi == 0xFFFFFFFFu));
        r.l[i] = diff;
    }
    return r;
}

fn p_val() -> U256 {
    var p: U256;
    p.l[0] = 0xFFFFFC2Fu; p.l[1] = 0xFFFFFFFEu; p.l[2] = 0xFFFFFFFFu; p.l[3] = 0xFFFFFFFFu;
    p.l[4] = 0xFFFFFFFFu; p.l[5] = 0xFFFFFFFFu; p.l[6] = 0xFFFFFFFFu; p.l[7] = 0xFFFFFFFFu;
    return p;
}

fn fp_add(a: U256, b: U256) -> U256 {
    let p = p_val();
    let sum = u256_add_raw(a, b);
    let overflow = u256_lt(sum, a);
    if (overflow || !u256_lt(sum, p)) {
        return u256_sub_raw(sum, p);
    }
    return sum;
}

fn fp_sub(a: U256, b: U256) -> U256 {
    let p = p_val();
    if (u256_lt(a, b)) {
        return u256_sub_raw(u256_add_raw(a, p), b);
    }
    return u256_sub_raw(a, b);
}

struct U512 { lo: U256, hi: U256 }

fn u256_mul_full(a: U256, b: U256) -> U512 {
    var w: array<u32, 16>;
    for (var i = 0u; i < 16u; i++) { w[i] = 0u; }
    for (var i = 0u; i < 8u; i++) {
        var carry = 0u;
        for (var j = 0u; j < 8u; j++) {
            let prod = mul32(a.l[i], b.l[j]);
            let s1 = add32c(w[i + j], prod.x, carry);
            let s2 = add32c(s1.x, 0u, 0u);
            w[i + j] = s2.x;
            carry = prod.y + s1.y;
        }
        w[i + 8u] = carry;
    }
    var res: U512;
    for (var i = 0u; i < 8u; i++) {
        res.lo.l[i] = w[i];
        res.hi.l[i] = w[i + 8u];
    }
    return res;
}

fn u512_mod_p(t: U512) -> U256 {
    let p = p_val();
    var s_lo = t.lo;
    for (var iter = 0u; iter < 4u; iter++) {
        let hi_prod = u256_mul_full(t.hi, U256(array<u32, 8>(977u, 1u, 0u, 0u, 0u, 0u, 0u, 0u)));
        let sum_lo = u256_add_raw(s_lo, hi_prod.lo);
        let carry = select(0u, 1u, u256_lt(sum_lo, s_lo));
        let rem_hi = u256_add_raw(hi_prod.hi, U256(array<u32, 8>(carry, 0u, 0u, 0u, 0u, 0u, 0u, 0u)));
        s_lo = sum_lo;
        if (u256_is_zero(rem_hi)) { break; }
    }
    while (!u256_lt(s_lo, p)) {
        s_lo = u256_sub_raw(s_lo, p);
    }
    return s_lo;
}

fn fp_mul(a: U256, b: U256) -> U256 {
    return u512_mod_p(u256_mul_full(a, b));
}

fn fp_sq(a: U256) -> U256 {
    return fp_mul(a, a);
}

fn fp_inv(a: U256) -> U256 {
    // Fermat's Little Theorem: a^(p-2) mod p
    let exp = U256(array<u32, 8>(0xFFFFFC2Du, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
                                 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu));
    var result = u256_one();
    var base = a;
    for (var i = 0u; i < 256u; i++) {
        let word = i / 32u;
        let bit  = i % 32u;
        if (((exp.l[word] >> bit) & 1u) != 0u) {
            result = fp_mul(result, base);
        }
        base = fp_sq(base);
    }
    return result;
}

// ----------------------------------------------------------------------------
// Ana Kanguru Compute Kernel (Montgomery Batch Affine)
// ----------------------------------------------------------------------------

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>
) {
    let tid = gid.x;
    let wid = lid.x; // 0..31 workgroup içi sıra

    var state = states[tid];
    var cur_x = state.x;
    var cur_y = state.y;
    var cur_dist = state.dist;
    let herd = state.herd;
    let k_id = state.id;

    for (var s = 0u; s < params.steps_per_call; s++) {
        // 1. Sıçrama indeksini X koordinatının alt 5 bitinden seç (0..31)
        let j_idx = cur_x.l[0] & 31u;
        let j_pt = jumps[j_idx];

        // 2. Delta X hesapla: J.x - cur_x
        var dx = fp_sub(j_pt.x, cur_x);
        if (u256_is_zero(dx)) { dx = u256_one(); } // Sayısal güvenlik koruması
        shared_dx[wid] = dx;
        workgroupBarrier();

        // 3. Montgomery Toplu Ters Alma (32 thread için Thread 0 çalıştırır)
        if (wid == 0u) {
            shared_c[0] = shared_dx[0];
            for (var i = 1u; i < 32u; i++) {
                shared_c[i] = fp_mul(shared_c[i - 1u], shared_dx[i]);
            }
            let total_inv = fp_inv(shared_c[31u]);
            var cur_inv = total_inv;
            for (var i = 31u; i >= 1u; i--) {
                shared_inv[i] = fp_mul(cur_inv, shared_c[i - 1u]);
                cur_inv = fp_mul(cur_inv, shared_dx[i]);
            }
            shared_inv[0] = cur_inv;
        }
        workgroupBarrier();

        let inv_dx = shared_inv[wid];

        // 4. Saf Affine Eliptik Eğri Nokta Toplaması
        let dy = fp_sub(j_pt.y, cur_y);
        let lambda = fp_mul(dy, inv_dx);
        let x_new = fp_sub(fp_sq(lambda), fp_add(cur_x, j_pt.x));
        let y_new = fp_sub(fp_mul(lambda, fp_sub(cur_x, x_new)), cur_y);

        cur_x = x_new;
        cur_y = y_new;
        cur_dist = u256_add_raw(cur_dist, j_pt.dist);

        // 5. Ayırt Edici Nokta (Distinguished Point / Tuzak) Kontrolü
        if ((cur_x.l[0] & params.dp_mask_lo) == 0u && (cur_x.l[1] & params.dp_mask_hi) == 0u) {
            let slot = atomicAdd(&trap_header.trap_count, 1u);
            if (slot < params.max_traps) {
                var rec: TrapRecord;
                rec.x = cur_x;
                rec.dist = cur_dist;
                rec.herd = herd;
                rec.parity = cur_y.l[0] & 1u;
                rec.kangaroo_id = k_id;
                rec._pad = 0u;
                trap_records[slot] = rec;
            }
        }
    }

    // Durumu kaydet
    state.x = cur_x;
    state.y = cur_y;
    state.dist = cur_dist;
    states[tid] = state;
}
`;

class WebGpuKangarooEngine {
    constructor() {
        this.device = null;
        this.pipeline = null;
        this.stateBuf = null;
        this.jumpsBuf = null;
        this.paramsBuf = null;
        this.trapHeaderBuf = null;
        this.trapRecordsBuf = null;
        this.trapReadbackBuf = null;
        this.bindGroup = null;

        this.numKangaroos = 2048; // 64 workgroup x 32 threads
        this.stepsPerCall = 128;
        this.maxTraps = 512;

        this.targetPubHex = '';
        this.rangeMin = 0n;
        this.rangeMax = 0n;
        this.span = 0n;
        this.dpMaskLo = 0x0000FFFF;
        this.dpMaskHi = 0;

        this.tameTraps = new Map(); // pointKey -> distBigInt
        this.wildTraps = new Map(); // pointKey -> distBigInt
        this.totalSteps = 0n;
        this.isSolved = false;
        this.solvedKey = null;
        this.isRunning = false;

        this.ec = null;
        this.targetPoint = null;
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

    async init(device, targetPubHex, rangeMin, rangeMax) {
        this.device = device;
        this.targetPubHex = targetPubHex.trim().replace(/^0x/i, '');
        this.rangeMin = typeof rangeMin === 'bigint' ? rangeMin : BigInt('0x' + rangeMin.toString().replace(/^0x/i, ''));
        this.rangeMax = typeof rangeMax === 'bigint' ? rangeMax : BigInt('0x' + rangeMax.toString().replace(/^0x/i, ''));
        this.span = this.rangeMax > this.rangeMin ? (this.rangeMax - this.rangeMin) : 1n;

        const el = (typeof window !== 'undefined' && window.elliptic) ? window.elliptic : null;
        if (!el || !el.ec) throw new Error('[WebGPU Kangaroo] elliptic kütüphanesi bulunamadı');
        this.ec = new el.ec('secp256k1');
        const keyObj = this.ec.keyFromPublic(this.targetPubHex, 'hex');
        this.targetPoint = keyObj.getPublic();

        // 1. Sıçrama Tablosunu Oluştur (Van Oorschot - Wiener uyumlu)
        const sqrtSpan = this.bigIntSqrt(this.span);
        const meanJump = (sqrtSpan / 2n) > 0n ? (sqrtSpan / 2n) : 1n;
        const jumpsData = new Uint32Array(32 * 24); // 32 * (8 x + 8 y + 8 dist)
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

        // 2. DP Maskesi (Hedef büyüklüğe göre dinamik)
        const sqrtBits = sqrtSpan.toString(2).length;
        const dpBits = Math.max(10, Math.min(22, Math.floor(sqrtBits / 2)));
        if (dpBits <= 32) {
            this.dpMaskLo = (1 << dpBits) - 1;
            this.dpMaskHi = 0;
        } else {
            this.dpMaskLo = 0xFFFFFFFF;
            this.dpMaskHi = (1 << (dpBits - 32)) - 1;
        }

        // 3. Başlangıç Kanguru Durumları
        const stateWords = new Uint32Array(this.numKangaroos * 36);
        const half = this.numKangaroos / 2;

        for (let i = 0; i < this.numKangaroos; i++) {
            const isWild = i >= half;
            const kIdx = BigInt(i);
            const shift = (meanJump * (kIdx + 1n)) % (this.span > 0n ? this.span : 1n);

            let pt = null;
            let dist = 0n;

            if (!isWild) {
                // TAME: rangeMax - shift
                dist = (this.rangeMax > shift) ? (this.rangeMax - shift) : this.rangeMax;
                pt = this.ec.g.mul(dist.toString(16));
            } else {
                // WILD: Target + shift
                dist = shift;
                const shiftPt = this.ec.g.mul(shift.toString(16));
                pt = this.targetPoint.add(shiftPt);
            }

            const xWords = this.bigIntToU256Words(BigInt('0x' + pt.getX().toString(16)));
            const yWords = this.bigIntToU256Words(BigInt('0x' + pt.getY().toString(16)));
            const distWords = this.bigIntToU256Words(dist);

            const base = i * 36;
            stateWords.set(xWords, base);
            stateWords.set(yWords, base + 8);
            stateWords.set(distWords, base + 16);
            stateWords[base + 24] = isWild ? 1 : 0; // herd
            stateWords[base + 25] = i;              // id
        }

        // 4. GPU Buffer Tahsisleri
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
            this.numKangaroos,
            this.stepsPerCall,
            this.dpMaskLo,
            this.dpMaskHi,
            this.maxTraps,
            0, 0, 0
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

        // 5. Shader ve Pipeline Derleme
        const module = this.device.createShaderModule({ code: WGSL_KANGAROO_SHADER });
        const compileInfo = await module.getCompilationInfo();
        const errs = compileInfo.messages.filter(m => m.type === 'error');
        if (errs.length > 0) {
            console.error('[WebGPU Kangaroo] Shader HATA:\n' + errs.map(e => e.message).join('\n'));
            return false;
        }

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

        console.log(`[WebGPU Kangaroo] ✅ Motor Hazır! Kangurular: ${this.numKangaroos} (${half} Tame / ${half} Wild), DP Maskesi: 0x${dpBits.toString(16)} bits`);
        return true;
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

    async dispatchBatch() {
        if (!this.pipeline || !this.device || this.isSolved) return 0;

        // Tuzak sayacını sıfırla
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
                const headerOffset = 4; // 16 bytes / 4
                for (let i = 0; i < trapCount; i++) {
                    const base = headerOffset + i * 20; // 80 bytes / 4 = 20 words
                    const xBig = this.u256WordsToBigInt(arr, base);
                    const distBig = this.u256WordsToBigInt(arr, base + 8);
                    const herd = arr[base + 16]; // 0 = TAME, 1 = WILD
                    const parity = arr[base + 17];
                    const pointKey = (parity === 0 ? '02' : '03') + xBig.toString(16).padStart(64, '0');

                    if (herd === 0) {
                        // TAME
                        this.tameTraps.set(pointKey, distBig);
                        if (this.wildTraps.has(pointKey)) {
                            this.checkCollision(pointKey, distBig, this.wildTraps.get(pointKey));
                            if (this.isSolved) break;
                        }
                    } else {
                        // WILD
                        this.wildTraps.set(pointKey, distBig);
                        if (this.tameTraps.has(pointKey)) {
                            this.checkCollision(pointKey, this.tameTraps.get(pointKey), distBig);
                            if (this.isSolved) break;
                        }
                    }
                    this.lastKey = distBig.toString(16).padStart(64, '0');
                }
            }
        } finally {
            this.trapReadbackBuf.unmap();
        }

        const batchSteps = BigInt(this.numKangaroos) * BigInt(this.stepsPerCall);
        this.totalSteps += batchSteps;
        return { steps: Number(batchSteps), lastKey: this.lastKey || '' };
    }

    checkCollision(pointKey, tDist, wDist) {
        let diff = (tDist - wDist) % SECP256K1_ORDER;
        if (diff < 0n) diff += SECP256K1_ORDER;

        if (this.targetPoint) {
            let checkPt = this.ec.g.mul(diff.toString(16));
            if (checkPt.getX().toString(16) === this.targetPoint.getX().toString(16)) {
                if (checkPt.getY().toString(16) !== this.targetPoint.getY().toString(16)) {
                    diff = (SECP256K1_ORDER - diff) % SECP256K1_ORDER;
                }
                this.isSolved = true;
                this.solvedKey = diff;
                console.log('🎉🎉🎉 [WEBGPU KANGAROO BINGO!] BULMACA ÇÖZÜLDÜ! Özel Anahtar: 0x' + diff.toString(16));
                return true;
            }
        }
        return false;
    }
}

if (typeof window !== 'undefined') {
    window.WebGpuKangarooEngine = WebGpuKangarooEngine;
}
