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
            const spanNum = Number(this.span);
            const sqrtSpan = (spanNum > 0 && spanNum < Number.MAX_SAFE_INTEGER) 
                           ? Math.sqrt(spanNum) 
                           : Number(this.span >> (BigInt(this.span.toString(2).length / 2)));
            
            const meanJump = Math.max(1, Math.floor(sqrtSpan / 2));
            
            // 32 adet deterministik sıçrama ve EC noktaları
            this.jumps = [];
            this.jumpPoints = [];
            for (let i = 0; i < 32; i++) {
                const jVal = BigInt(Math.max(1, Math.floor(meanJump * (0.5 + i / 31.0))));
                this.jumps.push(jVal);
                this.jumpPoints.push(this.G.mul(jVal.toString(16)));
            }

            // 3. Belirgin Nokta (Distinguished Point) Maskesi
            this.dpBits = Math.max(2, Math.min(18, Math.floor(Math.log2(Math.max(16, sqrtSpan)) / 2)));
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
        // Evcil Kanguru: Aralığın sonundan başlar
        this.tameDist = this.rangeMax;
        this.tamePoint = this.G.mul(this.tameDist.toString(16));

        // Vahşi Kanguru: Hedef Açık Anahtardan (W) başlar
        this.wildDist = 0n;
        this.wildPoint = this.targetPoint ? this.targetPoint : null;

        this.tameTraps.clear();
        this.wildTraps.clear();
        this.totalSteps = 0;
        this.isSolved = false;
        this.solvedKey = null;
        this.newTrapsToBroadcast = [];

        // Pollard's Kangaroo İlkesi (DUAL Modu):
        // Evcil kanguru aralığın üst sınırından başlayıp vahşi kangurunun geleceği yöne doğru
        // tuzak patikasını (trap line) döşer (~2.5 * √N adım önden koşar).
        if (this.role === 'DUAL' && this.tamePoint) {
            const spanNum = Number(this.span > 0n && this.span < 10000000000n ? this.span : 1000000n);
            const sqrtNum = Math.floor(Math.sqrt(spanNum));
            const preLeadSteps = Math.min(2500, Math.max(32, Math.floor(2.5 * sqrtNum)));
            this.runTameSteps(preLeadSteps);
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
