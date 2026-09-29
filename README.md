# ⚡ BTC Puzzle Hunter — Single-File Offline Edition (`index.html`)

[![GitHub Repo](https://img.shields.io/badge/GitHub-Mucid19%2Fbtc--puzzle--hunter-181717?logo=github)](https://github.com/Mucid19/btc-puzzle-hunter)
[![Live Demo](https://img.shields.io/badge/GitHub%20Pages-Live%20Demo-brightgreen?logo=githubpages)](https://mucid19.github.io/btc-puzzle-hunter/)
[![Bitcoin](https://img.shields.io/badge/Bitcoin-BTC-orange?logo=bitcoin)](https://bitcoin.org)
[![Zero Dependency](https://img.shields.io/badge/Dependencies-0%20(Pure%20Single%20File)-blue)](#)
[![Multi-Thread](https://img.shields.io/badge/Multi--Thread-Web%20Workers-blue)](#)
[![Hardware Accel](https://img.shields.io/badge/Hardware%20Accel-WebGPU-purple)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

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
* Uyumlu tarayıcı ve GPU'larda saniyede milyonlarca anahtar kombinasyonunu doğrudan grafik kartı çekirdeklerinde işleme opsiyonu.

### 4. 🎯 8+ Zengin Matematiksel Arama Algoritması
* **🎲 Rastgele (Random / PRNG):** Kriptografik güvenli homojen dağılım.
* **📐 Sobol Düşük Tutarsızlık (Quasi-Random):** Boşluk bırakmayan alan örtüşümü.
* **📏 Van der Corput:** Sayı teorisi tabanlı taban-2 dağılımı.
* **🌟 Weyl Altın Oran ($\phi$):** İrrasyonel adım frekans modülasyonu.
* **♾️ Asal Adım (Coprime Stride):** Modüler döngüsüz tarama adımları.
* **🌀 Hilbert Uzay Doldurma Eğrisi:** Yüksek boyutlu topolojik yakınlık taraması.
* **⚡ Zayıf Entropi (Weak Entropy):** Düşük popcount / zayıf rastgelelik filtreleri.
* **🌌 Deterministik Kaos (Deterministic Chaos):** Doğrusal olmayan karmaşık dinamik yürüyüş.

### 5. 🛡️ %100 Çevrimdışı ve Sıfır Ağ İsteği
* **Gömülü Kütüphaneler:** CryptoJS, secp256k1, BigInteger, Bech32, Base58 ve RIPEMD160 kütüphaneleri doğrudan tek bir dosya içine entegre edilmiştir.
* **21.953 Satoshi / Patoshi Adres Havuzu** yerel RAM'de taranır.
* **23.669 Özel Cüzdan Bloom Filtresi** yerel bellek üzerinde milisaniyeler içinde sorgulanır.
* Dışarıya hiçbir veri göndermez; gizliliğiniz ve güvenliğiniz tamdır.

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
   * Üst menüden arama metodolojinizi belirleyin (örn: `🦘 Kangaroo`, `🎲 Rastgele`, `📐 Sobol`, `🌌 Kaos`).
3. **Thread / Donanım Ayarı:**
   * Cihazınızın çekirdek sayısına göre dilediğiniz iş parçacığı (Worker) sayısını seçin.
4. **Aramayı Başlatın:**
   * **`▶ Aramayı Başlat`** butonuna tıklayın.
   * Hedef anahtar bulunduğunda sesli ve görsel bildirimle ekranda 64 karakter tam özel anahtarı ve cüzdan adresi listelenecektir.

---

## 📂 Dosya Yapısı

Depo tamamen yalın ve tek dosyadan ibarettir:

| Dosya | Açıklama |
| :--- | :--- |
| **`index.html`** | %100 bağımsız, sıfır dış bağımlılıklı tek dosya (single-file) uygulama |
| **`manifest.json`** | PWA (Progressive Web App) mobil/masaüstü uygulama manifestosu |
| **`icon-192.png`** / **`icon-512.png`** | Uygulama ikonları |
| **`README.md`** | GitHub proje dokümantasyonu |

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
