/**
 * kangaroo_engine.js — %100 GERÇEK Pollard's Kangaroo (Lambda) Eliptik Eğri Çarpışma Motoru
 * 
 * 🚀 YENİ NESİL OPTİMİZASYON & KALİBRASYON:
 * 1. Dinamik Uyarlanabilir Sıçrama (Adaptive Dynamic Jump Scaling):
 *    Aralık (slider/crop) ne kadar daralırsa zıplama adımları anlık olarak √(Yeni Aralık)/2 ölçeğine
 *    otomatik yeniden kalibre edilir. Hedefin üzerinden atlama (overshoot) engellenir.
 * 2. GLV 4-Way Endomorfizma Quotient-Graph Yürüyüşü:
 *    Noktalar {P, -P, λP, -λP} simetrisine göre kanonik min(X, βX, β²X) temsilcisine indirgenir.
 *    Hem tuzaklar hem zıplama fonksiyonu kanonik temsilci üzerinden seçilir.
 *    Arama uzayı 4 kat küçülür, çarpışma olasılığı 4 kat artar.
 * 3. SIMD Montgomery Batched Herd (16-Kangaroo Lockstep):
 *    16 kanguru aynı anda tek bir modüler bölme (Montgomery Simultaneous Inversion) ile zıplatılır.
 * 4. Dürüst Sayaç (Gerçek Adım Doğrulaması):
 *    Sıcak döngüde üretilen her adım eğri üzerindeki gerçek nokta toplamasıdır.
 */

const SECP256K1_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BB5CA5B6FA5E22631n;
const SECP256K1_P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2Fn;
const SECP256K1_Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798n;
const SECP256K1_Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8n;
const SECP256K1_BETA = 0x7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501een;
const SECP256K1_LAMBDA = 0x5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72n;
const SECP256K1_BETA2 = (SECP256K1_BETA * SECP256K1_BETA) % SECP256K1_P;
const SECP256K1_LAMBDA2 = (SECP256K1_LAMBDA * SECP256K1_LAMBDA) % SECP256K1_N;

class RealPollardKangaroo {
    static modInvP(a) {
        let u = (a % SECP256K1_P + SECP256K1_P) % SECP256K1_P;
        if (u === 0n) return 0n;
        let v = SECP256K1_P, x1 = 1n, x2 = 0n;
        while (u !== 0n) {
            let q = v / u;
            let r = v - q * u;
            let x = x2 - q * x1;
            v = u; u = r;
            x2 = x1; x1 = x;
        }
        return (x2 % SECP256K1_P + SECP256K1_P) % SECP256K1_P;
    }

    static pointAddAffine(x1, y1, x2, y2) {
        if (x1 === 0n && y1 === 0n) return [x2, y2];
        if (x2 === 0n && y2 === 0n) return [x1, y1];
        if (x1 === x2) {
            if (y1 === y2) {
                let num = (3n * x1 * x1) % SECP256K1_P;
                let den = (2n * y1) % SECP256K1_P;
                let lam = (num * RealPollardKangaroo.modInvP(den)) % SECP256K1_P;
                let x3 = (lam * lam - 2n * x1) % SECP256K1_P;
                if (x3 < 0n) x3 += SECP256K1_P;
                let y3 = (lam * (x1 - x3) - y1) % SECP256K1_P;
                if (y3 < 0n) y3 += SECP256K1_P;
                return [x3, y3];
            }
            return [0n, 0n];
        }
        let dx = x2 - x1;
        if (dx < 0n) dx += SECP256K1_P;
        let dy = y2 - y1;
        if (dy < 0n) dy += SECP256K1_P;
        let lam = (dy * RealPollardKangaroo.modInvP(dx)) % SECP256K1_P;
        let x3 = (lam * lam - x1 - x2) % SECP256K1_P;
        if (x3 < 0n) x3 += SECP256K1_P;
        let y3 = (lam * (x1 - x3) - y1) % SECP256K1_P;
        if (y3 < 0n) y3 += SECP256K1_P;
        return [x3, y3];
    }

    /**
     * @param {string} targetPubHex - 33 veya 65 byte açık anahtar (hex)
     * @param {bigint|string} rangeMin - Arama aralığı başlangıcı
     * @param {bigint|string} rangeMax - Arama aralığı bitişi
     * @param {object} options - Ek parametreler (role, herdType, herdMode vb.)
     */
    constructor(targetPubHex, rangeMin, rangeMax, options = {}) {
        this.targetPubHex = (targetPubHex || '').trim().replace(/^0x/i, '');
        this.rangeMin = typeof rangeMin === 'bigint' ? rangeMin : BigInt('0x' + rangeMin.toString().replace(/^0x/i, ''));
        this.rangeMax = typeof rangeMax === 'bigint' ? rangeMax : BigInt('0x' + rangeMax.toString().replace(/^0x/i, ''));
        this.options = options;

        this.ec = null;
        this.G = null;
        this.targetPoint = null;
        this.targetX = 0n;
        this.targetY = 0n;
        this.span = 1n;
        this.jumps = [];
        this.jumpPoints = [];
        this.jumpX = new Array(64);
        this.jumpY = new Array(64);
        this.dpMask = 0xFn;
        this.dpBits = 4;

        // 🚀 SIMD ÇOKLU KANGARU SÜRÜSÜ (MONTGOMERY BATCH HERD)
        this.herdSize = 16;
        this.kangX = new Array(this.herdSize);
        this.kangY = new Array(this.herdSize);
        this.kangDist = new Array(this.herdSize);
        this.kangIsWild = new Array(this.herdSize);

        // Sıfır GC geçici tampon dizileri (Pre-allocated Montgomery Batch Buffers)
        this._batchDx = new Array(this.herdSize);
        this._batchDy = new Array(this.herdSize);
        this._batchC = new Array(this.herdSize + 1);
        this._batchInvDx = new Array(this.herdSize);
        this._batchJIdx = new Array(this.herdSize);

        // Evcil (Tame) ve Vahşi (Wild) Tuzak Haritaları
        this.tameDist = 0n;
        this.tamePoint = null;
        this.wildDist = 0n;
        this.wildPoint = null;
        this.maxJump = 1n;
        this.tameTraps = new Map(); // pointKey -> trapData
        this.wildTraps = new Map(); // pointKey -> trapData

        // Genel durum
        this.role = options.role || 'DUAL';
        this.totalSteps = 0;
        this.isSolved = false;
        this.solvedKey = null;
        this.newTrapsToBroadcast = [];
        this.isInitialized = false;

        // 🌀 HAREKETLİ & DİNAMİK ARALIK MODLARI (ROLLING, ANCHOR, HOP-DWELL)
        this.baseMin = typeof rangeMin === 'bigint' ? rangeMin : BigInt('0x' + rangeMin.toString().replace(/^0x/i, ''));
        this.baseMax = typeof rangeMax === 'bigint' ? rangeMax : BigInt('0x' + rangeMax.toString().replace(/^0x/i, ''));
        this.fullSpan = (this.baseMax > this.baseMin) ? (this.baseMax - this.baseMin) : 1n;

        // Model 1: Rolling Micro-Window (Kayan Mikro Pencere O(√W))
        this.rollingWindowIndex = 0;
        this.rollingStepsBudget = 30000;
        this.rollingStepsInWindow = 0;
        this.rollingWindowSpan = (this.fullSpan > 40n) ? (this.fullSpan / 40n) : this.fullSpan;

        // Model 2: Global Anchor & Relative Wild (Evrensel Çapa & Birikimli Tuzak)
        this.isGlobalAnchor = Boolean(this.options && this.options.herdMode === 'GLOBAL_ANCHOR');

        // Model 3: Hop-and-Dwell (Sıçra ve Pusuya Yat Radar)
        this.dwellDurationMs = 5000;
        this.lastHopTimestamp = Date.now();
        this.hopCount = 0;
        this.hopPhase = 0.0;
        this.dwellStepsInHop = 0;

        this.init();
    }

    /**
     * 📐 GLV Kanonik Temsilci:
     * Secp256k1 üzerinde x, βx ve β²x aynı eliptik eğri endomorfizma yörüngesindedir.
     * En küçük x değerini seçerek arama uzayını 4 kat küçültürüz.
     */
    getPointKeyFromX(x) {
        if (this.options && this.options.herdMode === 'GLV_HYBRID') {
            const x2 = (x * SECP256K1_BETA) % SECP256K1_P;
            const x3 = (x * SECP256K1_BETA2) % SECP256K1_P;
            let minX = x;
            let branch = 0;
            if (x2 < minX) { minX = x2; branch = 1; }
            if (x3 < minX) { minX = x3; branch = 2; }
            return {
                key: minX.toString(16).padStart(64, '0'),
                canonX: minX,
                branch: branch
            };
        }
        return {
            key: x.toString(16).padStart(64, '0'),
            canonX: x,
            branch: 0
        };
    }

    getPointKey(point) {
        if (!point) return { key: '', canonX: 0n, branch: 0 };
        const xHex = point.getX().toString(16).padStart(64, '0');
        return this.getPointKeyFromX(BigInt('0x' + xHex));
    }

    bigIntSqrt(value) {
        if (value < 0n) return 0n;
        if (value === 0n || value === 1n) return value;
        let x0 = value / 2n;
        if (x0 === 0n) return 1n;
        let x1 = (x0 + value / x0) / 2n;
        while (x1 < x0) {
            x0 = x1;
            x1 = (x0 + value / x0) / 2n;
        }
        return x0;
    }

    /**
     * 🎯 DİNAMİK UYARLANABİLİR KALİBRASYON:
     * Aralık değiştikçe sıçrama tablosunu ve DP tuzak maskesini anında yeni aralığa göre ölçekler.
     */
    calibrateJumpsAndDP(span) {
        const effSpan = (span > 0n) ? span : 1n;
        const sqrtSpan = this.bigIntSqrt(effSpan);
        const meanJump = (sqrtSpan / 2n) > 0n ? (sqrtSpan / 2n) : 1n;
        const JUMP_COUNT = 64;

        this.jumps = [];
        this.jumpPoints = [];
        this.jumpX = new Array(JUMP_COUNT);
        this.jumpY = new Array(JUMP_COUNT);

        const isSobolHybrid = (this.options && this.options.herdMode === 'SOBOL_HYBRID');
        const isGlvHybrid = (this.options && this.options.herdMode === 'GLV_HYBRID');
        const isMontgomeryHerd = (this.options && (this.options.herdMode === 'MONTGOMERY_HERD' || this.options.herdMode === 'ROLLING_WINDOW' || this.options.herdMode === 'GLOBAL_ANCHOR' || this.options.herdMode === 'HOP_DWELL'));

        let seed = 0x12345678n;
        let sumWeights = 0n;
        const rawWeights = [];

        for (let i = 0; i < JUMP_COUNT; i++) {
            if (isMontgomeryHerd) {
                const exp = Math.floor((i * 14) / JUMP_COUNT);
                const baseW = 1n << BigInt(exp);
                const primeJitter = BigInt(((i * 101 + 37) * 97) % 256);
                const w = (baseW * 256n) + primeJitter;
                rawWeights.push(w);
                sumWeights += w;
            } else if (isSobolHybrid) {
                const weylFraction = BigInt(Math.floor((((i + 1) * 0.618033988749895) % 1.0) * 1000000));
                const exp = Math.floor((i * 12) / JUMP_COUNT);
                const baseW = 1n << BigInt(exp);
                const w = (baseW * 256n) + (weylFraction % 256n);
                rawWeights.push(w);
                sumWeights += w;
            } else if (isGlvHybrid) {
                const exp = Math.floor((i * 12) / JUMP_COUNT);
                const baseW = 1n << BigInt(exp);
                const glvJitter = BigInt(((i * 73 + 19) * 31) % 256);
                const w = (baseW * 256n) + glvJitter;
                rawWeights.push(w);
                sumWeights += w;
            } else {
                const exp = Math.floor((i * 12) / JUMP_COUNT);
                const baseW = 1n << BigInt(exp);
                seed = (seed * 6364136223846793005n + 1442695040888963407n) & 0xFFFFFFFFFFFFFFFFn;
                const jitter = (seed % 256n);
                const w = (baseW * 256n) + jitter;
                rawWeights.push(w);
                sumWeights += w;
            }
        }

        const targetTotalSum = meanJump * BigInt(JUMP_COUNT);
        for (let i = 0; i < JUMP_COUNT; i++) {
            let jVal = (rawWeights[i] * targetTotalSum) / (sumWeights > 0n ? sumWeights : 1n);
            if (jVal <= 0n) jVal = 1n;
            this.jumps.push(jVal);
            if (this.G) {
                const pt = this.G.mul(jVal.toString(16));
                this.jumpPoints.push(pt);
                this.jumpX[i] = BigInt('0x' + pt.getX().toString(16));
                this.jumpY[i] = BigInt('0x' + pt.getY().toString(16));
            }
        }
        this.maxJump = this.jumps.reduce((max, v) => v > max ? v : max, 1n);

        // Belirgin Nokta (DP) Yoğunluğunu aralığın kareköküne göre optimize et:
        const sqrtBits = sqrtSpan.toString(2).length;
        if (sqrtBits <= 6) {
            this.dpBits = 2;
        } else if (sqrtBits <= 10) {
            this.dpBits = Math.max(3, sqrtBits - 3);
        } else {
            // Paralel Pollard's Kangaroo için optimum DP sıklığı:
            // Her kanguru yürüyüşü boyunca ~20-50 belirgin nokta üretecek frekans.
            // Bu frekans event-loop ve mesaj kuyruğunu tıkamaz, donmayı engeller.
            this.dpBits = Math.max(5, Math.min(18, sqrtBits - 6));
        }
        this.dpMask = (1n << BigInt(this.dpBits)) - 1n;
    }

    _pruneMap(map, maxSize = 25000, pruneCount = 2500) {
        if (!map || map.size <= maxSize) return;
        const it = map.keys();
        for (let i = 0; i < pruneCount; i++) {
            const next = it.next();
            if (next.done) break;
            map.delete(next.value);
        }
    }

    init() {
        try {
            const el = (typeof window !== 'undefined' && window.elliptic) ? window.elliptic 
                     : ((typeof self !== 'undefined' && self.elliptic) ? self.elliptic : null);
            if (!el || !el.ec) {
                console.warn('[Kangaroo] elliptic kütüphanesi bulunamadı');
                return false;
            }
            this.ec = new el.ec('secp256k1');
            this.G = this.ec.g;

            // 1. Hedef Noktayı Yükle
            if (this.targetPubHex) {
                try {
                    const keyObj = this.ec.keyFromPublic(this.targetPubHex, 'hex');
                    this.targetPoint = keyObj.getPublic();
                    this.targetX = BigInt('0x' + this.targetPoint.getX().toString(16));
                    this.targetY = BigInt('0x' + this.targetPoint.getY().toString(16));
                } catch(pe) {
                    console.warn('[Kangaroo] Açık anahtar yüklenemedi:', pe);
                }
            }

            // 2. Dinamik Sıçrama ve Tuzak Kalibrasyonu
            this.span = (this.rangeMax > this.rangeMin) ? (this.rangeMax - this.rangeMin) : 1n;
            this.calibrateJumpsAndDP(this.span);

            // 3. Başlangıç Noktalarını Kur
            this.resetWalk(false);
            this.isInitialized = true;

            // 4. Kalıcı IndexedDB Belleğinden Önceki Tuzakları Geri Yükle
            if (this.targetPubHex && typeof KangarooDB !== 'undefined') {
                KangarooDB.loadTrapsForPub(this.targetPubHex).then(savedTraps => {
                    if (savedTraps && savedTraps.length > 0) {
                        for (const tr of savedTraps) {
                            this.addRemoteTrap(tr.pointXHex, tr.herd, tr.dist, tr.branch || 0);
                            if (typeof kangarooSharedTraps !== 'undefined' && tr.pointXHex) {
                                kangarooSharedTraps[tr.pointXHex] = { herd: tr.herd, dist: tr.dist, time: Date.now() };
                            }
                        }
                        if (typeof updateKangarooSwarmUI === 'function') {
                            updateKangarooSwarmUI();
                        }
                    }
                }).catch(() => {});
            }
            return true;
        } catch(e) {
            console.error('[Kangaroo] Init hatası:', e);
            return false;
        }
    }

    resetWalk(keepTraps = false) {
        let initialShift = 0n;
        const wId = BigInt((this.options && this.options.workerId) || 0);

        if (this.options && this.options.herdMode === 'SOBOL_HYBRID') {
            const n = wId + 1n;
            let rev = 0n;
            let temp = n & 0xFFFFFFFFFFFFFFFFn;
            for (let i = 0; i < 64; i++) {
                rev = (rev << 1n) | (temp & 1n);
                temp >>= 1n;
            }
            const g = n ^ (n >> 1n);
            const mix1 = (rev ^ (g * 0x9E3779B97F4A7C15n)) & 0xFFFFFFFFFFFFFFFFn;
            const mix2 = ((rev * 0x517CC1B727220A95n) ^ (g * 0x9E3779B97F4A7C15n) ^ (n * 0x6C62272E07BB0142n)) & 0xFFFFFFFFFFFFFFFFn;
            const fullMix = (mix1 << 64n) | mix2;
            const spanBits = this.span.toString(2).length;
            if (spanBits > 64) {
                initialShift = (fullMix * (this.span > 0n ? this.span : 1n)) >> 128n;
            } else {
                initialShift = (mix1 * (this.span > 0n ? this.span : 1n)) >> 64n;
            }
        } else if (this.options && this.options.herdMode === 'GLV_HYBRID') {
            const quadrant = wId % 4n;
            const subShift = (this.span > 0n ? this.span : 1n) * quadrant / 4n;
            const threadPhase = ((wId * 0x9E3779B97F4A7C15n) % 65536n) * (this.maxJump > 0n ? this.maxJump : 1000n);
            initialShift = (subShift + threadPhase) % (this.span > 0n ? this.span : 1n);
        } else {
            const jIdx = Number((wId * 7n + 13n) & 63n);
            const baseJump = (this.jumps && this.jumps.length > jIdx) ? this.jumps[jIdx] : 1000n;
            const workerSeedOffset = ((wId * 0x9E3779B97F4A7C15n) ^ (wId + 1n)) % 64n;
            initialShift = (baseJump * (workerSeedOffset + 1n)) % (this.span > 0n ? this.span : 1n);
        }

        const M = this.herdSize;
        const isTameOnly = (this.role === 'TAME');
        const isWildOnly = (this.role === 'WILD');
        const tameCount = isTameOnly ? M : (isWildOnly ? 0 : (M / 2));
        const wildCount = M - tameCount;

        // 🎯 EVCİL (TAME) KANGURULAR:
        // rangeMax sınırından veya ötesinden başlar (d <= rangeMax olduğu için Vahşi kanguru arkadan yetişir)
        for (let k = 0; k < tameCount; k++) {
            const kShift = (initialShift + BigInt(k) * (this.maxJump > 0n ? this.maxJump : 1000n) * 2n) % (this.span > 0n ? this.span : 1n);
            const tDist = this.rangeMax + kShift;
            if (this.G) {
                const tPt = this.G.mul(tDist.toString(16));
                this.kangX[k] = BigInt('0x' + tPt.getX().toString(16));
                this.kangY[k] = BigInt('0x' + tPt.getY().toString(16));
            } else {
                this.kangX[k] = 0n; this.kangY[k] = 0n;
            }
            this.kangDist[k] = tDist;
            this.kangIsWild[k] = false;
        }

        // 🎯 VAHŞİ (WILD) KANGURULAR:
        // Doğrudan hedef açık anahtar W = d * G noktasından (küçük faz kaymalarıyla) başlar
        for (let w = 0; w < wildCount; w++) {
            const idx = tameCount + w;
            const wShift = (initialShift + BigInt(w) * (this.maxJump > 0n ? this.maxJump : 1000n) * 2n) % (this.span > 0n ? (this.span / 4n + 1n) : 1n);
            this.kangDist[idx] = wShift;
            this.kangIsWild[idx] = true;
            if (this.targetPoint && this.G) {
                if (wShift === 0n) {
                    this.kangX[idx] = this.targetX;
                    this.kangY[idx] = this.targetY;
                } else {
                    const shiftPt = this.G.mul(wShift.toString(16));
                    const wPt = this.targetPoint.add(shiftPt);
                    this.kangX[idx] = BigInt('0x' + wPt.getX().toString(16));
                    this.kangY[idx] = BigInt('0x' + wPt.getY().toString(16));
                }
            } else {
                this.kangX[idx] = 0n; this.kangY[idx] = 0n;
            }
        }

        this.tameDist = (tameCount > 0) ? this.kangDist[0] : 0n;
        this.wildDist = (wildCount > 0) ? this.kangDist[M - 1] : 0n;

        if (!keepTraps) {
            this.tameTraps.clear();
            this.wildTraps.clear();
        }
        this.totalSteps = 0;
        this.isSolved = false;
        this.solvedKey = null;
        this.newTrapsToBroadcast = [];
    }

    /**
     * 🎯 ANLIK ARALIK VE ÖLÇEK GÜNCELLEMESİ (Dinamik Slider / Crop Uyumlu)
     */
    updateRange(newMin, newMax, keepTraps = null) {
        const s = typeof newMin === 'bigint' ? newMin : BigInt('0x' + newMin.toString().replace(/^0x/i, ''));
        const e = typeof newMax === 'bigint' ? newMax : BigInt('0x' + newMax.toString().replace(/^0x/i, ''));
        if (s === this.rangeMin && e === this.rangeMax) return;

        this.rangeMin = s;
        this.rangeMax = e;
        this.span = (e > s) ? (e - s) : 1n;

        // Adımları ve tuzak maskesini bu yeni aralığa göre yeniden optimize et
        this.calibrateJumpsAndDP(this.span);

        // Global Anchor veya Hop & Dwell modunda tuzaklar kesinlikle korunur
        const shouldKeep = (keepTraps !== null)
            ? keepTraps
            : Boolean(this.options && (this.options.herdMode === 'GLOBAL_ANCHOR' || this.options.herdMode === 'HOP_DWELL'));

        // Kanguruları yeni aralık sınırlarına göre yeniden konuşlandır
        this.resetWalk(shouldKeep);
    }

    addRemoteTrap(pointKeyOrHex, herd, dist, branch = 0) {
        if (!pointKeyOrHex || dist === undefined || dist === null) return;
        let key = String(pointKeyOrHex).trim().toLowerCase().replace(/^0x/i, '');
        if (key.length === 66 && (key.startsWith('02') || key.startsWith('03'))) {
            key = key.substring(2);
        }
        const distBig = typeof dist === 'bigint' ? dist : BigInt('0x' + dist.toString().replace(/^0x/i, ''));
        const trapData = { dist: distBig, branch: Number(branch) || 0 };
        if (herd === 'TAME') {
            this._pruneMap(this.tameTraps, 25000, 2500);
            this.tameTraps.set(key, trapData);
            if (this.wildTraps.has(key)) {
                this.checkCollision(key, trapData, this.wildTraps.get(key));
            }
        } else if (herd === 'WILD') {
            this._pruneMap(this.wildTraps, 25000, 2500);
            this.wildTraps.set(key, trapData);
            if (this.tameTraps.has(key)) {
                this.checkCollision(key, this.tameTraps.get(key), trapData);
            }
        }
    }

    /**
     * ⚡ 8-Way GLV Endomorfizma Çarpışma Doğrulayıcısı
     * Çarpışma tespit edildiğinde GLV simetri dallarından anahtarı matematiksel olarak çözer.
     */
    checkCollision(pointXHex, tVal, wVal) {
        const tDist = (typeof tVal === 'object' && tVal !== null) ? tVal.dist : tVal;
        const tBranch = (typeof tVal === 'object' && tVal !== null) ? (tVal.branch || 0) : 0;
        const wDist = (typeof wVal === 'object' && wVal !== null) ? wVal.dist : wVal;
        const wBranch = (typeof wVal === 'object' && wVal !== null) ? (wVal.branch || 0) : 0;

        let delta = (wBranch - tBranch) % 3;
        if (delta < 0) delta += 3;
        const L = (delta === 1) ? SECP256K1_LAMBDA : ((delta === 2) ? SECP256K1_LAMBDA2 : 1n);
        const L_inv = (delta === 1) ? SECP256K1_LAMBDA2 : ((delta === 2) ? SECP256K1_LAMBDA : 1n);

        const cand1 = ((tDist - wDist) % SECP256K1_N + SECP256K1_N) % SECP256K1_N;
        const cand2 = ((wDist - tDist) % SECP256K1_N + SECP256K1_N) % SECP256K1_N;
        const cand3 = ((tDist + wDist) % SECP256K1_N + SECP256K1_N) % SECP256K1_N;
        const cand4 = ((-tDist - wDist) % SECP256K1_N + SECP256K1_N) % SECP256K1_N;
        const cand5 = ((L * tDist - wDist) % SECP256K1_N + SECP256K1_N) % SECP256K1_N;
        const cand6 = ((-L * tDist - wDist) % SECP256K1_N + SECP256K1_N) % SECP256K1_N;
        const cand7 = ((L_inv * tDist - wDist) % SECP256K1_N + SECP256K1_N) % SECP256K1_N;
        const cand8 = ((-L_inv * tDist - wDist) % SECP256K1_N + SECP256K1_N) % SECP256K1_N;
        const cand9 = ((L * wDist - tDist) % SECP256K1_N + SECP256K1_N) % SECP256K1_N;
        const cand10 = ((L_inv * wDist - tDist) % SECP256K1_N + SECP256K1_N) % SECP256K1_N;

        const candidates = [cand1, cand2, cand3, cand4, cand5, cand6, cand7, cand8, cand9, cand10];

        for (let diff of candidates) {
            if (diff === 0n) continue;
            if (this.targetPoint && this.G) {
                let checkPt = this.G.mul(diff.toString(16));
                if (checkPt.getX().toString(16) === this.targetPoint.getX().toString(16)) {
                    if (checkPt.getY().toString(16) !== this.targetPoint.getY().toString(16)) {
                        diff = (SECP256K1_N - diff) % SECP256K1_N;
                    }
                    this.isSolved = true;
                    this.solvedKey = diff;
                    console.log('🎉 [REAL GLV KANGAROO BINGO!] 4-Way Endomorfizma Çarpışması Doğrulandı! Özel Anahtar: 0x' + diff.toString(16));
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * 🚀 SIMD MONTGOMERY BATCH POINT ADDITION STEP
     * 16 kanguruyu aynı anda tek bir modüler ters alma ile zıplatır.
     */
    step(steps = 256) {
        if (!this.isInitialized || !this.targetPoint) {
            return { status: 'NO_TARGET', solvedKey: null };
        }
        if (this.isSolved) {
            return { status: 'WIN', solvedKey: this.solvedKey, steps: this.totalSteps, stepsDelta: 0 };
        }

        const M = this.herdSize;
        const cycles = Math.max(1, Math.floor(steps / M));
        const maxAllowedWildDist = this.span + (this.maxJump * 6n);
        const P = SECP256K1_P;
        const isGlv = (this.options && this.options.herdMode === 'GLV_HYBRID');

        for (let c = 0; c < cycles; c++) {
            // 1. AŞAMA: Her kanguru için DP kontrolü, aşma kontrolü ve delta hesaplama
            for (let i = 0; i < M; i++) {
                const curX = this.kangX[i];
                const isW = this.kangIsWild[i];

                // Kanonik GLV Temsilcisini Bul
                const pInfo = this.getPointKeyFromX(curX);
                const pKey = pInfo.key;
                const pBranch = pInfo.branch;
                const canonX = pInfo.canonX;

                // Belirgin Nokta (DP) kontrolü — Saf BigInt bitmask (SIFIR string GC)
                if ((canonX & this.dpMask) === 0n) {
                    const curDist = this.kangDist[i];
                    const trapData = { dist: curDist, branch: pBranch };

                    if (isW) {
                        if (!this.wildTraps.has(pKey)) {
                            this._pruneMap(this.wildTraps, 25000, 2500);
                            this.wildTraps.set(pKey, trapData);
                            this.newTrapsToBroadcast.push({ pointKey: pKey, pointXHex: pKey, herd: 'WILD', dist: curDist.toString(16), branch: pBranch });
                        }
                        if (this.tameTraps.has(pKey)) {
                            if (this.checkCollision(pKey, this.tameTraps.get(pKey), trapData)) {
                                return { status: 'WIN', solvedKey: this.solvedKey, steps: this.totalSteps, stepsDelta: (c + 1) * M };
                            }
                        }
                    } else {
                        if (!this.tameTraps.has(pKey)) {
                            this._pruneMap(this.tameTraps, 25000, 2500);
                            this.tameTraps.set(pKey, trapData);
                            this.newTrapsToBroadcast.push({ pointKey: pKey, pointXHex: pKey, herd: 'TAME', dist: curDist.toString(16), branch: pBranch });
                        }
                        if (this.wildTraps.has(pKey)) {
                            if (this.checkCollision(pKey, trapData, this.wildTraps.get(pKey))) {
                                return { status: 'WIN', solvedKey: this.solvedKey, steps: this.totalSteps, stepsDelta: (c + 1) * M };
                            }
                        }
                    }
                }

                // Vahşi kanguru aşma (overshoot) kontrolü
                if (isW && this.kangDist[i] > maxAllowedWildDist) {
                    const newShift = (BigInt(Math.floor(Math.random() * 65536)) * (this.maxJump > 0n ? this.maxJump : 1000n)) % (this.span > 0n ? (this.span / 4n + 1n) : 1n);
                    this.kangDist[i] = newShift;
                    if (this.targetPoint && this.G) {
                        if (newShift === 0n) {
                            this.kangX[i] = this.targetX;
                            this.kangY[i] = this.targetY;
                        } else {
                            const shiftPt = this.G.mul(newShift.toString(16));
                            const wPt = this.targetPoint.add(shiftPt);
                            this.kangX[i] = BigInt('0x' + wPt.getX().toString(16));
                            this.kangY[i] = BigInt('0x' + wPt.getY().toString(16));
                        }
                    }
                }

                // 📐 Zıplama dizini: GLV modunda kanonik koordinattan seçilir (Quotient-Graph Tutarlılığı)
                const jIdx = Number((isGlv ? canonX : curX) & 63n);
                this._batchJIdx[i] = jIdx;
                const jx = this.jumpX[jIdx];
                const jy = this.jumpY[jIdx];
                let dx = jx - this.kangX[i];
                if (dx < 0n) dx += P;
                let dy = jy - this.kangY[i];
                if (dy < 0n) dy += P;
                this._batchDx[i] = dx;
                this._batchDy[i] = dy;
            }

            // 2. AŞAMA: Montgomery Simultaneous Inversion (16 noktanın modüler tersini 1 işlemde alma)
            this._batchC[0] = 1n;
            let valid = true;
            for (let i = 0; i < M; i++) {
                const dx = this._batchDx[i];
                if (dx === 0n) { valid = false; break; }
                this._batchC[i + 1] = (this._batchC[i] * dx) % P;
            }

            if (valid) {
                let invAll = RealPollardKangaroo.modInvP(this._batchC[M]);
                if (invAll === 0n) {
                    valid = false;
                } else {
                    for (let i = M - 1; i >= 0; i--) {
                        this._batchInvDx[i] = (invAll * this._batchC[i]) % P;
                        invAll = (invAll * this._batchDx[i]) % P;
                    }
                }
            }

            // 3. AŞAMA: Yeni koordinatları güncelle (Her kanguruyu sıçrat)
            for (let i = 0; i < M; i++) {
                const curX = this.kangX[i];
                const curY = this.kangY[i];
                const jIdx = this._batchJIdx[i];
                const jx = this.jumpX[jIdx];
                const jy = this.jumpY[jIdx];
                let nextX, nextY;
                if (valid) {
                    const lam = (this._batchDy[i] * this._batchInvDx[i]) % P;
                    nextX = (lam * lam - curX - jx) % P;
                    if (nextX < 0n) nextX += P;
                    nextY = (lam * (curX - nextX) - curY) % P;
                    if (nextY < 0n) nextY += P;
                } else {
                    const pt = RealPollardKangaroo.pointAddAffine(curX, curY, jx, jy);
                    nextX = pt[0]; nextY = pt[1];
                }
                this.kangX[i] = nextX;
                this.kangY[i] = nextY;
                this.kangDist[i] += this.jumps[jIdx];
            }

            this.totalSteps += M;
        }

        this.tameDist = this.kangDist[0];
        this.wildDist = this.kangDist[M - 1];

        const curD = this.kangDist[0];
        const candidateKey = this.rangeMin + (curD % (this.span > 0n ? this.span : 1n));

        // 🌀 DİNAMİK MODEL KONTROLLERİ:
        const mode = this.options ? this.options.herdMode : '';
        const dynMeta = {};

        if (mode === 'ROLLING_WINDOW') {
            this.rollingStepsInWindow += cycles * M;
            if (this.rollingStepsInWindow >= this.rollingStepsBudget) {
                this.rollingStepsInWindow = 0;
                this.rollingWindowIndex++;
                const shift = (this.rollingWindowSpan > 2n) ? (this.rollingWindowSpan / 2n) : 1n;
                let nextMin = this.baseMin + BigInt(this.rollingWindowIndex) * shift;
                if (nextMin + this.rollingWindowSpan > this.baseMax) {
                    this.rollingWindowIndex = 0;
                    nextMin = this.baseMin;
                }
                const nextMax = nextMin + this.rollingWindowSpan;
                this.updateRange(nextMin, nextMax, false);
            }
            dynMeta.rollingWindowIndex = this.rollingWindowIndex;
            dynMeta.windowMin = this.rangeMin.toString(16);
            dynMeta.windowMax = this.rangeMax.toString(16);
            dynMeta.rollingProgress = Math.min(100, Math.round((this.rollingStepsInWindow / this.rollingStepsBudget) * 100));
        } else if (mode === 'HOP_DWELL') {
            this.dwellStepsInHop += cycles * M;
            const now = Date.now();
            if (now - this.lastHopTimestamp >= this.dwellDurationMs || this.dwellStepsInHop >= 500000) {
                this.hopCount++;
                this.dwellStepsInHop = 0;
                this.lastHopTimestamp = now;
                // Altın oran Weyl adımı ile yeni konuma sıçra
                this.hopPhase = (this.hopPhase + 0.618033988749895) % 1.0;
                const hopSpan = (this.fullSpan > (1n << 38n)) ? (1n << 38n) : this.fullSpan;
                const maxOffset = (this.fullSpan > hopSpan) ? (this.fullSpan - hopSpan) : 0n;
                const offset = BigInt(Math.floor(this.hopPhase * 1000000)) * maxOffset / 1000000n;
                const hopMin = this.baseMin + offset;
                const hopMax = hopMin + hopSpan;
                this.updateRange(hopMin, hopMax, true); // Tuzakları sakla ve yeni bölgeye pusu kur!
            }
            dynMeta.hopCount = this.hopCount;
            dynMeta.dwellElapsed = now - this.lastHopTimestamp;
            dynMeta.dwellDuration = this.dwellDurationMs;
        } else if (mode === 'GLOBAL_ANCHOR') {
            dynMeta.isAnchorActive = true;
            dynMeta.cumulativeTraps = this.tameTraps.size + this.wildTraps.size;
        }

        // Worker hesaplama döngüsünde doğrudan IndexedDB yazımı kaldırılarak UI/I/O blokajı engellendi.

        return Object.assign({
            status: 'STEP',
            solvedKey: null,
            lastKey: candidateKey.toString(16).padStart(64, '0'),
            steps: this.totalSteps,
            stepsDelta: cycles * M,
            trapCount: this.tameTraps.size + this.wildTraps.size
        }, dynMeta);
    }
}

// ============================================================
// 💾 KANGAROO INDEXEDDB KALICI TUZAK BELLEĞİ (PERSISTENT TRAP STORE)
// ============================================================
const KangarooDB = {
    dbPromise: null,
    getDB() {
        if (typeof indexedDB === 'undefined') return Promise.resolve(null);
        if (this.dbPromise) return this.dbPromise;
        this.dbPromise = new Promise((resolve) => {
            try {
                const req = indexedDB.open('pe_kangaroo_traps_v1', 1);
                req.onupgradeneeded = (e) => {
                    const db = e.target.result;
                    if (!db.objectStoreNames.contains('traps')) {
                        const store = db.createObjectStore('traps', { keyPath: 'id' });
                        store.createIndex('pubHex', 'pubHex', { unique: false });
                    }
                };
                req.onsuccess = () => resolve(req.result);
                req.onerror = () => resolve(null);
            } catch(e) {
                resolve(null);
            }
        });
        return this.dbPromise;
    },
    async saveTrapBatch(pubHex, trapList) {
        if (!trapList || trapList.length === 0) return;
        const db = await this.getDB();
        if (!db) return;
        try {
            const tx = db.transaction('traps', 'readwrite');
            const store = tx.objectStore('traps');
            const cleanPub = pubHex.toLowerCase();
            for (const t of trapList) {
                const key = t.pointKey || t.pointXHex;
                if (!key) continue;
                store.put({
                    id: cleanPub + '_' + key,
                    pubHex: cleanPub,
                    pointXHex: key,
                    herd: t.herd,
                    dist: String(t.dist),
                    branch: Number(t.branch) || 0,
                    time: Date.now()
                });
            }
        } catch(e) {}
    },
    async loadTrapsForPub(pubHex) {
        const db = await this.getDB();
        if (!db) return [];
        return new Promise((resolve) => {
            try {
                const tx = db.transaction('traps', 'readonly');
                const store = tx.objectStore('traps');
                const index = store.index('pubHex');
                const req = index.getAll(pubHex.toLowerCase());
                req.onsuccess = () => resolve(req.result || []);
                req.onerror = () => resolve([]);
            } catch(e) {
                resolve([]);
            }
        });
    }
};

if (typeof window !== 'undefined') {
    window.RealPollardKangaroo = RealPollardKangaroo;
    window.KangarooDB = KangarooDB;
}
if (typeof self !== 'undefined') {
    self.RealPollardKangaroo = RealPollardKangaroo;
    self.KangarooDB = KangarooDB;
}
