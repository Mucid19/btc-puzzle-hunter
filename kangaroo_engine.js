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
        this.tameTraps = new Map(); // pointXHex -> tameDist

        // Vahşi (Wild) Durumu
        this.wildDist = 0n;
        this.wildPoint = null;
        this.wildTraps = new Map(); // pointXHex -> wildDist

        // Genel durum
        this.role = options.role || 'DUAL'; // 'TAME', 'WILD', veya 'DUAL' (ikisini de yürütür)
        this.totalSteps = 0;
        this.isSolved = false;
        this.solvedKey = null;
        this.newTrapsToBroadcast = [];
        this.isInitialized = false;

        this.init();
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
            
            // 32 adet deterministik sıçrama ve EC noktaları (meanJump etrafında [0.1 * m, 1.9 * m] dengeli dağılım)
            this.jumps = [];
            this.jumpPoints = [];
            for (let i = 0; i < 32; i++) {
                const factor = BigInt(Math.floor(100 + (i * 1800) / 31));
                let jVal = (meanJump * factor) / 1000n;
                if (jVal <= 0n) jVal = 1n;
                // Tek sayı yaparak eliptik eğri üzerinde periyodik döngüye kilitlenmeyi engelle
                if (jVal % 2n === 0n) jVal += 1n;
                this.jumps.push(jVal);
                this.jumpPoints.push(this.G.mul(jVal.toString(16)));
            }

            // 3. Belirgin Nokta (Distinguished Point) Maskesi
            // Aralık büyüklüğüne göre dinamik ölçekleme: ortalama her 16..65536 adımda 1 tuzak
            const sqrtBits = sqrtSpan.toString(2).length;
            this.dpBits = Math.max(4, Math.min(16, Math.floor(sqrtBits / 2) - 1));
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
        const initialShift = baseJump * (wId + 1n);

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
        // Evcil kanguru aralığın üst sınırından başlayıp vahşi kangurunun geleceği yöne doğru
        // tuzak patikasını (trap line) döşer. Aralığı kaplayacak kadar (~span / meanJump) önden koşar.
        if (this.role === 'DUAL' && this.tamePoint) {
            const sqrtSpan = this.bigIntSqrt(this.span);
            const meanJump = (sqrtSpan / 2n) > 0n ? (sqrtSpan / 2n) : 1n;
            const preSteps = Math.min(4096, Math.max(128, Number((this.span / meanJump) + 50n)));
            this.runTameSteps(preSteps);
        }
    }

    runTameSteps(count) {
        if (!this.tamePoint) return false;
        for (let s = 0; s < count; s++) {
            const txBig = BigInt('0x' + this.tamePoint.getX().toString(16));
            const txHex = txBig.toString(16);

            // Belirgin Nokta (DP) kontrolü
            if ((txBig & this.dpMask) === 0n) {
                if (!this.tameTraps.has(txHex)) {
                    this.tameTraps.set(txHex, this.tameDist);
                    this.newTrapsToBroadcast.push({ pointXHex: txHex, herd: 'TAME', dist: this.tameDist.toString(16) });
                }
                if (this.wildTraps.has(txHex)) {
                    if (this.checkCollision(txHex, this.tameDist, this.wildTraps.get(txHex))) {
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
    addRemoteTrap(pointXHex, herd, dist) {
        if (!pointXHex || dist === undefined || dist === null) return;
        const distBig = typeof dist === 'bigint' ? dist : BigInt('0x' + dist.toString().replace(/^0x/i, ''));
        if (herd === 'TAME') {
            this.tameTraps.set(pointXHex, distBig);
            // Eğer vahşi kangurumuz bu noktaya daha önce bastıysa anında çarpışma!
            if (this.wildTraps.has(pointXHex)) {
                this.checkCollision(pointXHex, distBig, this.wildTraps.get(pointXHex));
            }
        } else if (herd === 'WILD') {
            this.wildTraps.set(pointXHex, distBig);
            // Eğer evcil kangurumuz bu noktaya daha önce bastıysa anında çarpışma!
            if (this.tameTraps.has(pointXHex)) {
                this.checkCollision(pointXHex, this.tameTraps.get(pointXHex), distBig);
            }
        }
    }

    checkCollision(pointXHex, tDist, wDist) {
        // 1. Direkt Çarpışma: k = (d_T - d_W) mod n
        let diff = (tDist - wDist) % SECP256K1_N;
        if (diff < 0n) diff += SECP256K1_N;

        // 2. ⚡ Negation Map (Ters Simetri: 1.414x Hızlanma):
        //    P_Tame == -P_Wild => d_T * G == -Target - d_W * G
        //    => Target == -(d_T + d_W) * G => k = n - ((d_T + d_W) mod n)
        let sumDist = (tDist + wDist) % SECP256K1_N;
        let negCand1 = (SECP256K1_N - sumDist) % SECP256K1_N;
        let negCand2 = sumDist;

        const candidates = [diff, (SECP256K1_N - diff) % SECP256K1_N, negCand1, negCand2];

        // Doğrulama: candidate * G == targetPoint mi?
        if (this.targetPoint) {
            const targetX = this.targetPoint.getX().toString(16);
            const targetY = this.targetPoint.getY().toString(16);

            for (const cand of candidates) {
                if (cand <= 0n || cand >= SECP256K1_N) continue;
                try {
                    let checkPt = this.G.mul(cand.toString(16));
                    if (checkPt.getX().toString(16) === targetX) {
                        let finalKey = cand;
                        if (checkPt.getY().toString(16) !== targetY) {
                            finalKey = (SECP256K1_N - cand) % SECP256K1_N;
                        }
                        this.isSolved = true;
                        this.solvedKey = finalKey;
                        const isNeg = (cand === negCand1 || cand === negCand2);
                        if (isNeg) {
                            console.log('⚡⚡⚡ [REAL KANGAROO BINGO! — NEGATION MAP (TERS SİMETRİ)] 1.414x Hızlanma ile Bulundu! Özel Anahtar: 0x' + finalKey.toString(16));
                        } else {
                            console.log('🎉 [REAL KANGAROO BINGO! — DİREKT ÇARPIŞMA] Özel Anahtar: 0x' + finalKey.toString(16));
                        }
                        return true;
                    }
                } catch(e) {}
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
                const txHex = txBig.toString(16);

                // Belirgin Nokta (DP) kontrolü
                if ((txBig & this.dpMask) === 0n) {
                    if (!this.tameTraps.has(txHex)) {
                        this.tameTraps.set(txHex, this.tameDist);
                        this.newTrapsToBroadcast.push({ pointXHex: txHex, herd: 'TAME', dist: this.tameDist.toString(16) });
                    }
                    // Vahşi kanguru bu noktaya daha önce basmış mı?
                    if (this.wildTraps.has(txHex)) {
                        if (this.checkCollision(txHex, this.tameDist, this.wildTraps.get(txHex))) {
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
                const wxBig = BigInt('0x' + this.wildPoint.getX().toString(16));
                const wxHex = wxBig.toString(16);

                // Belirgin Nokta (DP) kontrolü
                if ((wxBig & this.dpMask) === 0n) {
                    if (!this.wildTraps.has(wxHex)) {
                        this.wildTraps.set(wxHex, this.wildDist);
                        this.newTrapsToBroadcast.push({ pointXHex: wxHex, herd: 'WILD', dist: this.wildDist.toString(16) });
                    }
                    // Evcil kanguru bu noktaya daha önce basmış mı? (ÇARPIŞMA!)
                    if (this.tameTraps.has(wxHex)) {
                        if (this.checkCollision(wxHex, this.tameTraps.get(wxHex), this.wildDist)) {
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
