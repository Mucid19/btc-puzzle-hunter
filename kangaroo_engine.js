/**
 * kangaroo_engine.js — %100 GERÇEK Pollard's Kangaroo (Lambda) Eliptik Eğri Çarpışma Motoru
 *
 * Matematiksel İlke:
 *   Hedef: W = d * G (Açık Anahtar / Public Key), d in [rangeMin, rangeMax]
 *   Evcil (Tame): T_0 = rangeMax * G, mesafe d_T = rangeMax
 *                 T_{i+1} = T_i + J_k (EC Nokta Toplaması), d_T += j_k
 *   Vahşi (Wild): W_0 = W, mesafe d_W = 0
 *                 W_{i+1} = W_i + J_k (EC Nokta Toplaması), d_W += j_k
 *   Çarpışma (Collision): T_i == W_j (aynı eliptik eğri noktası!)
 *                 => (rangeMax + sum j_T) * G == (d + sum j_W) * G
 *                 => d_T * G == (d + d_W) * G
 *                 => d = (d_T - d_W) mod n
 *
 * Karmaşıklık: O(sqrt(N)) — Brute force'a göre milyonlarca kat daha hızlı!
 */

const SECP256K1_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141n;

class RealPollardKangaroo {
    /**
     * @param {string} targetPubHex - 33 veya 65 byte açık anahtar (hex)
     * @param {bigint|string} rangeMin - Arama aralığı başlangıcı
     * @param {bigint|string} rangeMax - Arama aralığı bitişi
     * @param {object} options - Ek parametreler (role, herdType, jumpCount vb.)
     */
    constructor(targetPubHex, rangeMin, rangeMax, options = {}) {
        this.targetPubHex = (targetPubHex || '').trim().replace(/^0x/i, '');
        this.rangeMin = typeof rangeMin === 'bigint' ? rangeMin : BigInt('0x' + rangeMin.toString().replace(/^0x/i, ''));
        this.rangeMax = typeof rangeMax === 'bigint' ? rangeMax : BigInt('0x' + rangeMax.toString().replace(/^0x/i, ''));
        this.options = options;

        this.ec = null;
        this.G = null;
        this.targetPoint = null;
        this.span = 1n;
        this.jumps = [];
        this.jumpPoints = [];
        this.dpMask = 0xFn;
        this.dpBits = 4;

        // Evcil (Tame) Durumu
        this.tameDist = 0n;
        this.tamePoint = null;
        this.maxJump = 1n;
        this.tameTraps = new Map(); // pointKey -> tameDist

        // Vahşi (Wild) Durumu
        this.wildDist = 0n;
        this.wildPoint = null;
        this.wildTraps = new Map(); // pointKey -> wildDist

        // Genel durum
        this.role = options.role || 'DUAL'; // 'TAME', 'WILD', veya 'DUAL' (ikisini de yürütür)
        this.totalSteps = 0;
        this.isSolved = false;
        this.solvedKey = null;
        this.newTrapsToBroadcast = [];
        this.isInitialized = false;

        this.init();
    }

    getPointKey(point) {
        if (!point) return '';
        const isEven = point.getY().isEven();
        const prefix = isEven ? '02' : '03';
        return prefix + point.getX().toString(16).padStart(64, '0');
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
                } catch(pe) {
                    console.warn('[Kangaroo] Açık anahtar yüklenemedi:', pe);
                }
            }

            // 2. Aralık ve Sıçrama Parametreleri
            this.span = (this.rangeMax > this.rangeMin) ? (this.rangeMax - this.rangeMin) : 1n;
            const sqrtSpan = this.bigIntSqrt(this.span);
            const meanJump = (sqrtSpan / 2n) > 0n ? (sqrtSpan / 2n) : 1n;
            
            // 32 adet deterministik sıçrama ve EC noktaları (Van Oorschot-Wiener standardı)
            this.jumps = [];
            this.jumpPoints = [];
            let seed = 0x12345678n;
            for (let i = 0; i < 32; i++) {
                seed = (seed * 6364136223846793005n + 1442695040888963407n) & 0xFFFFFFFFFFFFFFFFn;
                const rnd = seed % 2000n;
                let jVal = (meanJump * (100n + rnd)) / 1000n;
                if (jVal <= 0n) jVal = 1n;
                this.jumps.push(jVal);
                this.jumpPoints.push(this.G.mul(jVal.toString(16)));
            }
            this.maxJump = this.jumps.reduce((max, v) => v > max ? v : max, 1n);

            // 3. Belirgin Nokta (Distinguished Point) Maskesi
            // Ortalama her 256..16384 adımda 1 tuzak üretir (IPC ve bellek taşmasını önler)
            const sqrtBits = sqrtSpan.toString(2).length;
            this.dpBits = Math.max(8, Math.min(16, Math.floor(sqrtBits / 2)));
            this.dpMask = (1n << BigInt(this.dpBits)) - 1n;

            // 4. Başlangıç Noktalarını Kur
            this.resetWalk();
            this.isInitialized = true;
            return true;
        } catch(e) {
            console.error('[Kangaroo] Init hatası:', e);
            return false;
        }
    }

    resetWalk() {
        // Bağımsız Çekirdek Ofseti (Van Oorschot & Wiener Paralel Kangaroo Standardı)
        // Her çekirdek farklı bir başlangıç sıçramasından başlar, böylece aynı ayak izlerini kopyalamazlar
        const wId = BigInt(this.options.workerId || 0);
        const jIdx = Number(wId % 32n);
        const baseJump = (this.jumps && this.jumps.length > jIdx) ? this.jumps[jIdx] : 1000n;
        const initialShift = (baseJump * (wId + 1n)) % (this.span > 0n ? this.span : 1n);

        // Evcil Kanguru: Aralığın sonundan ofsetli başlar
        this.tameDist = (this.rangeMax > initialShift) ? (this.rangeMax - initialShift) : this.rangeMax;
        this.tamePoint = this.G.mul(this.tameDist.toString(16));

        // Vahşi Kanguru: Hedef Açık Anahtardan (W) ofsetli başlar
        this.wildDist = initialShift;
        if (this.targetPoint) {
            const shiftPoint = this.G.mul(initialShift.toString(16));
            this.wildPoint = this.targetPoint.add(shiftPoint);
        } else {
            this.wildPoint = null;
        }

        this.tameTraps.clear();
        this.wildTraps.clear();
        this.totalSteps = 0;
        this.isSolved = false;
        this.solvedKey = null;
        this.newTrapsToBroadcast = [];

        // Pollard's Kangaroo İlkesi (DUAL Modu):
        // Hafif ön hazırlık adımları (başlangıç takılmasını önlemek için 64..256 adım)
        if (this.role === 'DUAL' && this.tamePoint) {
            const sqrtSpan = this.bigIntSqrt(this.span);
            const meanJump = (sqrtSpan / 2n) > 0n ? (sqrtSpan / 2n) : 1n;
            const preSteps = Math.min(256, Math.max(32, Number((this.span / meanJump) + 10n)));
            this.runTameSteps(preSteps);
        }
    }

    runTameSteps(count) {
        if (!this.tamePoint) return false;
        for (let s = 0; s < count; s++) {
            const txBig = BigInt('0x' + this.tamePoint.getX().toString(16));
            const pKey = this.getPointKey(this.tamePoint);

            // Belirgin Nokta (DP) kontrolü
            if ((txBig & this.dpMask) === 0n) {
                if (!this.tameTraps.has(pKey)) {
                    this.tameTraps.set(pKey, this.tameDist);
                    this.newTrapsToBroadcast.push({ pointKey: pKey, pointXHex: pKey, herd: 'TAME', dist: this.tameDist.toString(16) });
                }
                if (this.wildTraps.has(pKey)) {
                    if (this.checkCollision(pKey, this.tameDist, this.wildTraps.get(pKey))) {
                        return true;
                    }
                }
            }

            const tIdx = Number(txBig % 32n);
            this.tamePoint = this.tamePoint.add(this.jumpPoints[tIdx]);
            this.tameDist += this.jumps[tIdx];
        }
        return false;
    }

    /**
     * Dışarıdan (örneğin başka bir sekmeden / BroadcastChannel üzerinden) gelen tuzak noktasını ekle
     */
    addRemoteTrap(pointKeyOrHex, herd, dist) {
        if (!pointKeyOrHex || dist === undefined || dist === null) return;
        const key = String(pointKeyOrHex);
        const distBig = typeof dist === 'bigint' ? dist : BigInt('0x' + dist.toString().replace(/^0x/i, ''));
        if (herd === 'TAME') {
            this.tameTraps.set(key, distBig);
            // Eğer vahşi kangurumuz bu noktaya daha önce bastıysa anında çarpışma!
            if (this.wildTraps.has(key)) {
                this.checkCollision(key, distBig, this.wildTraps.get(key));
            }
        } else if (herd === 'WILD') {
            this.wildTraps.set(key, distBig);
            // Eğer evcil kangurumuz bu noktaya daha önce bastıysa anında çarpışma!
            if (this.tameTraps.has(key)) {
                this.checkCollision(key, this.tameTraps.get(key), distBig);
            }
        }
    }

    checkCollision(pointXHex, tDist, wDist) {
        let diff = (tDist - wDist) % SECP256K1_N;
        if (diff < 0n) diff += SECP256K1_N;

        // Doğrulama: diff * G == targetPoint mi?
        if (this.targetPoint) {
            let checkPt = this.G.mul(diff.toString(16));
            if (checkPt.getX().toString(16) === this.targetPoint.getX().toString(16)) {
                // Y-koordinat paritesi kontrolü (ECDSA simetrisi)
                if (checkPt.getY().toString(16) !== this.targetPoint.getY().toString(16)) {
                    diff = (SECP256K1_N - diff) % SECP256K1_N;
                }
                this.isSolved = true;
                this.solvedKey = diff;
                console.log('🎉 [REAL KANGAROO BINGO!] Çarpışma doğrulandı! Özel Anahtar: 0x' + diff.toString(16));
                return true;
            }
        }
        return false;
    }

    /**
     * Belirtilen adım sayısı kadar eliptik eğri sıçraması yap
     * @param {number} steps
     * @returns {object} { status: 'WIN'|'STEP'|'NO_TARGET', solvedKey, lastKey, steps }
     */
    step(steps = 256) {
        if (!this.isInitialized || !this.targetPoint) {
            return { status: 'NO_TARGET', solvedKey: null };
        }
        if (this.isSolved) {
            return { status: 'WIN', solvedKey: this.solvedKey, steps: this.totalSteps };
        }

        const runTame = (this.role === 'TAME' || this.role === 'DUAL');
        const runWild = (this.role === 'WILD' || this.role === 'DUAL');

        for (let s = 0; s < steps; s++) {
            this.totalSteps++;

            // 1. Evcil (Tame) Sıçraması
            if (runTame && this.tamePoint) {
                const txBig = BigInt('0x' + this.tamePoint.getX().toString(16));
                const pKey = this.getPointKey(this.tamePoint);

                // Belirgin Nokta (DP) kontrolü
                if ((txBig & this.dpMask) === 0n) {
                    if (!this.tameTraps.has(pKey)) {
                        this.tameTraps.set(pKey, this.tameDist);
                        this.newTrapsToBroadcast.push({ pointKey: pKey, pointXHex: pKey, herd: 'TAME', dist: this.tameDist.toString(16) });
                    }
                    // Vahşi kanguru bu noktaya daha önce basmış mı?
                    if (this.wildTraps.has(pKey)) {
                        if (this.checkCollision(pKey, this.tameDist, this.wildTraps.get(pKey))) {
                            return { status: 'WIN', solvedKey: this.solvedKey, steps: this.totalSteps };
                        }
                    }
                }

                // Deterministik sıçrama (Noktanın X koordinatının alt bitlerine göre)
                const tIdx = Number(txBig % 32n);
                this.tamePoint = this.tamePoint.add(this.jumpPoints[tIdx]);
                this.tameDist += this.jumps[tIdx];
            }

            // 2. Vahşi (Wild) Sıçraması
            if (runWild && this.wildPoint) {
                // Aşma (Overshoot) kontrolü: Eğer vahşi kanguru arama penceresini aştıysa yeni bir ofsetle yeniden doğur (respawn)
                const maxAllowedWildDist = this.span + (this.maxJump * 8n);
                if (this.wildDist > maxAllowedWildDist) {
                    const newShift = (BigInt(Math.floor(Math.random() * 65536)) * this.maxJump) % (this.span > 0n ? this.span : 1n);
                    this.wildDist = newShift;
                    this.wildPoint = this.targetPoint.add(this.G.mul(newShift.toString(16)));
                }

                const wxBig = BigInt('0x' + this.wildPoint.getX().toString(16));
                const wpKey = this.getPointKey(this.wildPoint);

                // Belirgin Nokta (DP) kontrolü
                if ((wxBig & this.dpMask) === 0n) {
                    if (!this.wildTraps.has(wpKey)) {
                        this.wildTraps.set(wpKey, this.wildDist);
                        this.newTrapsToBroadcast.push({ pointKey: wpKey, pointXHex: wpKey, herd: 'WILD', dist: this.wildDist.toString(16) });
                    }
                    // Evcil kanguru bu noktaya daha önce basmış mı? (ÇARPIŞMA!)
                    if (this.tameTraps.has(wpKey)) {
                        if (this.checkCollision(wpKey, this.tameTraps.get(wpKey), this.wildDist)) {
                            return { status: 'WIN', solvedKey: this.solvedKey, steps: this.totalSteps };
                        }
                    }
                }

                // Deterministik sıçrama
                const wIdx = Number(wxBig % 32n);
                this.wildPoint = this.wildPoint.add(this.jumpPoints[wIdx]);
                this.wildDist += this.jumps[wIdx];
            }
        }

        const candidateKey = this.rangeMin + (this.tameDist % this.span);
        return {
            status: 'STEP',
            solvedKey: null,
            lastKey: candidateKey.toString(16).padStart(64, '0'),
            steps: this.totalSteps,
            trapCount: this.tameTraps.size + this.wildTraps.size
        };
    }
}

// Global Export
if (typeof window !== 'undefined') {
    window.RealPollardKangaroo = RealPollardKangaroo;
}
if (typeof self !== 'undefined') {
    self.RealPollardKangaroo = RealPollardKangaroo;
}
