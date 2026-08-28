# 🪙 BTC Puzzle Hunter (v4.0)

> **Tarayıcı tabanlı yüksek hızlı Bitcoin Puzzle & Satoshi Cüzdan Avcısı — Hash160 optimizasyonu, 21.953 Satoshi hedefi, canlı blokzincir senkronizasyonu ve PWA desteği.**

🔗 **Canlı Sürüm:** [Mucid19.github.io/btc-puzzle-hunter](https://Mucid19.github.io/btc-puzzle-hunter/)

---

## ⚡ v4.0 ile Gelen Yeni Nesil Özellikler

### 1. 🚀 2x – 3x Hız Artışı (Hash160 Doğrudan Karşılaştırma)
* **Eski Yöntem:** Her üretilen anahtarda ağır Base58 ve Checksum dönüşümleri yapılıyordu.
* **Yeni Yöntem:** Açık bulmacaların 20 baytlık `hash160` değerleri başlangıçta bir defa çözülüp önbelleğe alınır. Döngü içinde doğrudan 20 baytlık ham hex karşılaştırılır. Base58 kodlaması yalnızca kazanç durumunda veya ekranda gösterim için çağrılır.

### 2. 🏛️ 21.953 Satoshi Cüzdanı Eşzamanlı Tarama (Always-On)
* Bitcoin'in erken döneminde (2009-2010) Satoshi Nakamoto tarafından kazılmış **21.953 adet bilinen 50 BTC'lik coinbase cüzdanı** (Patoshi veri seti) arka planda otomatik olarak taranır.
* Üretilen her tek anahtar, hem seçtiğiniz ana Puzzle için hem de bu 21.953 Satoshi cüzdanı için **aynı anda** test edilir.

### 3. 🌐 Canlı Blokzincir Senkronizasyonu & Otomatik Arşivleme
* Sayfa ilk açıldığında arka planda `mempool.space API` üzerinden açık bulmacaların (`#71..#160`) güncel durumları sorgulanır.
* Biri tarafından çözülmüş ve bakiyesi `0 BTC` olmuş bulmacalar otomatik olarak algılanır, `localStorage`'a işlenir ve anında **Mor Arşiv Listesine** taşınır.

### 4. 🎯 Tek Satır %3'lük Dilim Seçici (%20 - %35 Altın Bölge)
* Geçmiş 66 bulmacanın matematiksel analizinde çözümlerin **%62'sinin %25-%50 aralığında** çıktığı ve Puzzle #66'nın **%25.62**'de bulunduğu kanıtlanmıştır.
* Tek tıkla devreye giren otomatik `%3'lük dilimler` (`%0-%3`, `%3-%6`...) ve `%20-%35 Altın Bölge` seçicisi eklenmiştir. Arama çalışırken aralık otomatik olarak kilitlenir.

### 5. 📱 Ekran Uyku Kilidi (WakeLock API)
* Mobilde veya dizüstü bilgisayarda tarama başlatıldığında ekranın kapanıp taramayı durdurmasını engelleyen `navigator.wakeLock` devrededir.

---

## 🔍 Arama Motorları & Algoritmalar

| Algoritma | Çalışma Mantığı |
| :--- | :--- |
| **🔄 OTO** | En dengeli iki motor (**Rastgele** ve **Hibrit**) arasında her 25 adımda bir otomatik geçiş yapar. |
| **🎲 RASTGELE** | `window.crypto.getRandomValues()` ile kriptografik güvenli saf Monte Carlo araması. |
| **🔀 HİBRİT** | 14 farklı üslü atlama basamağı ile geniş adımlı arama ve periyodik sıralı blok taraması kombinasyonu. |

---

## 📱 PWA (Progressive Web App) Desteği

- **Tamamen Çevrimdışı (Offline) Çalışır:** Service Worker v4 ile tüm kütüphaneler yerel önbellekten yüklenir.
- **Uygulama Olarak Yükleme:** Android, iOS (Safari Paylaş -> Ana Ekrana Ekle) ve Masaüstü (Chrome/Edge) desteği.
- **Otomatik Güncelleme:** Yeni sürümler çıktığında önbellek kendini otomatik yeniler.

---

## 🏗️ Dosya Mimarisi

```
btc-puzzle-hunter/
├── index.html              # Optimize edilmiş tek parça web uygulaması
├── all_160_puzzles.json    # 1-160 bulmaca veritabanı
├── manifest.json           # PWA manifest dosyası
├── sw.js                   # Service Worker v4 önbellek yöneticisi
├── icon-192.png            # 192x192 PWA uygulama ikonu
├── icon-512.png            # 512x512 PWA uygulama ikonu
└── README.md               # Teknik dokümantasyon
```

---

## ⚙️ Güvenlik & Kütüphaneler

- `elliptic.js 6.5.4` (secp256k1 eliptik eğri hesaplamaları)
- `CryptoJS 4.1.1` (SHA256 ve RIPEMD160)
- `mempool.space API` (CORS uyumlu canlı blokzincir sorgulama)

---

*Not: Bu yazılım eğitim, matematiksel araştırma ve kriptografi analiz amaçlıdır. Bitcoin Puzzle Transaction açık kaynaklı bir topluluk yarışmasıdır.*
