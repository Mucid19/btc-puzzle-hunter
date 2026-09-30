# ⚡ BTC Puzzle Hunter — Single-File Offline Edition (`index.html`)

[![GitHub Repo](https://img.shields.io/badge/GitHub-Mucid19%2Fbtc--puzzle--hunter-181717?logo=github)](https://github.com/Mucid19/btc-puzzle-hunter)
[![Live Demo](https://img.shields.io/badge/GitHub%20Pages-Live%20Demo-brightgreen?logo=githubpages)](https://mucid19.github.io/btc-puzzle-hunter/)
[![Bitcoin](https://img.shields.io/badge/Bitcoin-BTC-orange?logo=bitcoin)](https://bitcoin.org)
[![Zero Dependency](https://img.shields.io/badge/Dependencies-0%20(Pure%20Single%20File)-blue)](#)
[![Multi-Thread](https://img.shields.io/badge/Multi--Thread-Web%20Workers-blue)](#)
[![Hardware Accel](https://img.shields.io/badge/Hardware%20Accel-WebGPU-purple)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

![BTC Puzzle Hunter Preview](ui_preview.png)

**`index.html`**, Bitcoin Puzzle işlemlerini çözmek için geliştirilmiş; **%100 bağımsız, sıfır dış bağımlılık (Zero External Dependency)** içeren ve doğrudan tarayıcı üzerinden çalışan tek dosyalık (single-file), yüksek performanslı bir arama ve kriptografik analiz aracıdır.

Herhangi bir sunucu kurulumu, `npm`, Node.js veya internet bağlantısı gerektirmez. Dosyayı tarayıcınızda açtığınız anda tüm secp256k1 ve hash kütüphaneleri cihazınızın yerel RAM belleğinde tamamen bağımsız çalışır.

---

## 🌐 Canlı Kullanım (GitHub Pages)

Doğrudan tarayıcınız üzerinden hiçbir şey indirmeden çalıştırabilirsiniz:
👉 **[https://mucid19.github.io/btc-puzzle-hunter/](https://mucid19.github.io/btc-puzzle-hunter/)**

---

## 🌟 Öne Çıkan Özellikler

### 1. 🦘 Pollard's Kangaroo Hibrit Motoru ($O(\sqrt{N})$)
* Klasik kaba kuvvet (brute-force) yerine $O(\sqrt{N})$ zaman karmaşıklığı ile çalışan gerçek eliptik eğri sıçrama ve tuzak motoru.
* **Hibrit Aday Tarama:** Eliptik eğri tuzak noktaları hesaplanırken, eşzamanlı olarak aralıktan aday anahtarlar doğrudan secp256k1 Hash160 / Adres karşılaştırıcısına sokulur; dar test pencerelerinde ve önek dilimlerinde saniyeler içinde hedefi yakalar.
* İfşa edilmiş açık anahtar (Exposed Public Key) desteği ile tuzak (trap) oluşturma ve anında doğrulama.

### 2. ⚡ Çoklu İş Parçacığı (Multi-Thread Web Workers)
* Cihazınızın donanımsal çekirdek sayısını (`navigator.hardwareConcurrency`) otomatik algılar.
* 1 ila 32+ bağımsız Web Worker ile CPU çekirdeklerini %100 verimle paralel olarak çalıştırır.
* Dinamik önbellek geçersiz kılma (Revoke Blob URL) ile her başlatmada güncel motor kodunu temiz olarak devreye alır.

### 3. 🚀 WebGPU Donanım Hızlandırma
* Uyumlu tarayıcı ve GPU'larda saniyede milyonlarca anahtar kombinasyonunu doğrudan grafik kartı çekirdeklerinde işleme opsiyonu (`gpu_engine.js`).

### 4. 🎯 12+ Zengin Matematiksel Arama Algoritması
* **🎲 Saf Rastgele (Random / Uniform RNG):** Kriptografik güvenli homojen global tarama.
* **📐 Sobol Düşük Tutarsızlık (Quasi-Random):** Boşluk bırakmayan alan örtüşümü.
* **📏 Van der Corput:** Sayı teorisi tabanlı taban-2 bit tersleme dağılımı.
* **🌟 Weyl Altın Oran ($\phi$):** İrrasyonel adım frekans modülasyonlu altın kafes.
* **♾️ Asal Adım (Coprime Stride):** Modüler döngüsüz tam kapsama taraması.
* **🌀 Hilbert Uzay Doldurma Eğrisi:** %100 space-filling topolojik yakınlık taraması.
* **⚡ Zayıf Entropi (Weak Entropy):** Kusurlu PRNG ve zayıf tohum kalıpları.
* **🌌 Deterministik Kaos (Deterministic Chaos):** Lojistik ve çadır haritalı dinamik yürüyüş.
* **🦘 Pollard's Kangaroo:** Eliptik eğri tuzak ve çarpışma ağı $O(\sqrt{N})$.
* **🎲 Halton Dizisi:** 2, 3, 5, 7 asal tabanlı çok boyutlu homojen dağılım.
* **🔀 Scrambled Sobol:** Owen sayısal kaydırmalı dinamik boşluk örtüşümü.
* **🎰 MCMC (Markov Zinciri):** Metropolis-Hastings ve Lévy sıçramalı uyarlanabilir arama.

### 5. 🛡️ %100 Çevrimdışı ve Sıfır Ağ İsteği
* **Gömülü Kütüphaneler:** CryptoJS, secp256k1, BigInteger, Bech32, Base58 ve RIPEMD160 kütüphaneleri doğrudan tek bir dosya içine entegre edilmiştir.

### 6. 🔑 Tam 256-Bit Standart Bitcoin Özel Anahtar Formatı
* Bulunan ve aranan tüm anahtarlar standart 64 onaltılık karakter (`0x0000000000000000000000000000000000000000000000000...`) formatında sunulur.
* WIF (Wallet Import Format), Sıkıştırılmış (Compressed) ve Sıkıştırılmamış (Uncompressed) adreslerle otomatik doğrulanır.

---

## 🚀 Hızlı Başlangıç (Yerel Kullanım)

1. Depoyu indirin veya klonlayın:
   ```bash
   git clone https://github.com/Mucid19/btc-puzzle-hunter.git
   ```
2. Klasör içerisindeki **`index.html`** dosyasına herhangi bir modern web tarayıcısında çift tıklayın:
   * Google Chrome
   * Microsoft Edge
   * Brave Browser
   * Mozilla Firefox
3. **Hepsi bu kadar!** İnternet bağlantınız olmasa dahi anında çalışmaya başlar.

---

## 🖥️ Kullanım Rehberi

1. **Hedef Belirleme:**
   * Sol paneldeki listeden aramak istediğiniz Bitcoin Puzzle numarasını seçin (örneğin `#57`, `#66` vb.).
   * İsterseniz **"Özel Aralık / Hex"** veya **"Yüzde Dilimi"** sekmesinden kendi aralığınızı girin.
2. **Algoritma Seçimi:**
   * Üst menüden arama metodolojinizi belirleyin (örn: `🔄 Strateji Döngüsü`, `🦘 Kangaroo`, `🎲 Rastgele`, `📐 Sobol`, `🎲 Halton`, `🎰 MCMC`).
3. **Thread / Donanım Ayarı:**
   * Cihazınızın çekirdek sayısına göre dilediğiniz iş parçacığı (Worker) sayısını seçin veya GPU motorunu devreye alın.
4. **Aramayı Başlatın:**
   * **`▶ Aramayı Başlat`** butonuna tıklayın.
   * Hedef anahtar bulunduğunda sesli ve görsel bildirimle ekranda 64 karakter tam özel anahtarı ve cüzdan adresi listelenecektir.

---

## 📂 Dosya Yapısı

Depo tamamen optimize edilmiş ve yüklemeye hazır temiz bir yapıdadır:

| Dosya | Açıklama |
| :--- | :--- |
| **`index.html`** | %100 bağımsız, sıfır dış bağımlılıklı ana web uygulaması |
| **`gpu_engine.js`** | WebGPU donanım hızlandırma ve paralel anahtar üretim motoru |
| **`kangaroo_engine.js`** | Pollard's Kangaroo eliptik eğri çarpışma ve tuzak kütüphanesi |
| **`all_160_puzzles.json`** | Tüm 160 Bitcoin bulmacasının hedef aralık ve adres veri seti |
| **`manifest.json`** | PWA (Progressive Web App) masaüstü ve mobil kurulum manifestosu |
| **`sw.js`** | %100 çevrimdışı önbellekleme sağlayan Service Worker |
| **`icon-192.png`** / **`icon-512.png`** | Yüksek çözünürlüklü PWA uygulama ikonları |
| **`ui_preview.png`** | GitHub proje arayüz önizleme görseli |
| **`LICENSE`** | MIT Açık Kaynak Lisansı |
| **`.gitignore`** | Git geçici ve yedek dosya hariç tutma kuralları |
| **`README.md`** | GitHub proje tanıtım ve kullanım dokümantasyonu |

---

## ☕ Bağış ve Destek

Bu projeyi faydalı bulduysanız, algoritmaların geliştirilmesine ve açık kaynak araştırma çalışmalarına destek olmak isterseniz Bitcoin ile katkıda bulunabilirsiniz:

### 🪙 Bitcoin (BTC) Bağış Adresi:
```text
bc1qxf5cfrxasshlkt79x0q805l9t3feer868en68nhlxmwetlr6sv4qdfda5s
```

> *Tüm bağışlar yeni algoritmik optimizasyonlar ve açık kaynak Bitcoin kriptografi araçlarının geliştirilmesi için kullanılmaktadır. Desteğiniz için teşekkürler!*

---

## ⚖️ Yasal Uyarı

Bu yazılım yalnızca eğitim, matematiksel araştırma ve kriptografik analiz amacıyla geliştirilmiştir. Kullanıcılar kendi kullanım senaryolarından ve geçerli yerel mevzuatlara uyumdan bizzat sorumludur.
