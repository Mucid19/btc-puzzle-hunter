# ⚡ BTC Puzzle Hunter — Single-File Offline Edition (`index.html`)

[![GitHub Repo](https://img.shields.io/badge/GitHub-Mucid19%2Fbtc--puzzle--hunter-181717?logo=github)](https://github.com/Mucid19/btc-puzzle-hunter)
[![Live Demo](https://img.shields.io/badge/GitHub%20Pages-Live%20Demo-brightgreen?logo=githubpages)](https://mucid19.github.io/btc-puzzle-hunter/)
[![Bitcoin](https://img.shields.io/badge/Bitcoin-BTC-orange?logo=bitcoin)](https://bitcoin.org)
[![Zero Dependency](https://img.shields.io/badge/Dependencies-0%20(Pure%20Single%20File)-blue)](#)
[![Multi-Thread](https://img.shields.io/badge/Multi--Thread-Web%20Workers-blue)](#)
[![Hardware Accel](https://img.shields.io/badge/Hardware%20Accel-WebGPU-purple)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

![BTC Puzzle Hunter Preview](ui_preview.png)

**`index.html`** is an ultra-fast, 100% offline, zero-dependency browser-based cryptographic analysis and search engine designed to solve Bitcoin Puzzle transactions. It runs entirely inside your local browser memory using parallel Web Workers and WebGPU hardware acceleration.

---

## 🌐 Live Demo (GitHub Pages)

Run directly in your browser with zero installation:  
👉 **[https://mucid19.github.io/btc-puzzle-hunter/](https://mucid19.github.io/btc-puzzle-hunter/)**

---

## ⚡ Highlights & Key Features

### 1. 🦘 Pollard's Kangaroo Engine ($O(\sqrt{N})$)
* Solves elliptic curve discrete logarithm problems in $O(\sqrt{N})$ time complexity instead of exponential brute force.
* **Elliptic Curve Collision Search**: Computes deterministic pseudorandom jumps and trap points across the puzzle's search space.
* Built-in support for puzzles with exposed public keys (#19, #57, #135, #140, #145, #150, #155, #160).

### 2. ⚡ Multi-Thread Web Workers
* Automatically detects your CPU core count (`navigator.hardwareConcurrency`).
* Spawns parallel Web Workers with 100% CPU utilization.
* Dynamic cache busting via Blob URLs ensures clean memory allocation on every run.

### 3. 🚀 WebGPU Hardware Acceleration
* Harness the raw parallel computing power of your GPU cores directly through WebGPU (`gpu_engine.js`) to process key combinations at maximum throughput.

### 4. 🎯 12+ Specialized Search Algorithms
* **🔄 Strategy Cycle (Agent)**: Automatically cycles through search strategies dynamically.
* **🔄 Auto (Hybrid)**: Blends low-discrepancy and chaotic sequences for uniform coverage.
* **🎲 Pure Random**: Cryptographically secure uniform global exploration.
* **📐 Sobol (Quasi-RNG)**: Low-discrepancy space-filling sequence.
* **📏 Van der Corput**: Number-theoretic bit-reversal sequence.
* **🌟 Weyl Golden Ratio ($\phi$)**: Irrational stride lattice for non-overlapping traversal.
* **♾️ Coprime Stride**: Modular non-repeating cycle walk.
* **🌀 Hilbert Curve**: Multi-dimensional space-filling topological walk.
* **⚡ Weak Entropy**: Targets defective PRNG seeds and low-entropy keys.
* **🌌 Deterministic Chaos**: Logistic and tent map dynamic trajectory.
* **🦘 Pollard's Kangaroo**: Elliptic curve trap and collision network.
* **🎲 Halton Sequence**: Prime-based multi-dimensional uniform coverage.
* **🔀 Scrambled Sobol**: Owen-scrambled dynamic space filling.
* **🎰 MCMC (Markov Chain)**: Adaptive Metropolis-Hastings walk.

### 5. 🔁 Target Management: Circular & Single Puzzle Modes
* **Full Pool & Multi-Select**: Hunt across all active unsolved puzzles simultaneously in circular rotation, or pick a custom subset.
* **Archive & Verification**: Solved puzzles (#1 to #65, etc.) are kept in a dedicated Archive dropdown for test validation and algorithm verification.
* **Space Range Sliders**: Interactive dual hex sliders to focus the search window on specific sections of the keyspace.

### 6. 🌐 Bilingual Interface (EN / TR)
* Instant 1-click language switcher (`🌐 Türkçe` / `🌐 English`) in the navigation bar.
* Fully translated controls, algorithm names, live statistics, and notifications.

### 7. 🛡️ 100% Air-Gapped & Offline Safe
* Embedded cryptographic modules: secp256k1 elliptic curve, BigInteger, SHA-256, and RIPEMD-160.
* Zero external API calls, zero telemetry, zero server-side dependencies. Works completely offline.

---

## 🚀 How to Use

1. **Clone or Download:**
   ```bash
   git clone https://github.com/Mucid19/btc-puzzle-hunter.git
   ```
2. **Open:**
   Double-click **`index.html`** in any modern web browser (Chrome, Brave, Edge, Firefox, Opera).
3. **Configure:**
   * Select your target puzzle (#66, #71, #72, or use **All Puzzles**).
   * Choose your algorithm (e.g., `🔄 Auto (Hybrid)`, `🦘 Pollard's Kangaroo`, `📐 Sobol`).
   * Select your CPU core count or enable WebGPU.
4. **Start:**
   Click **`▶ Start Hunting`** to begin the search.

---

## 📂 Repository File Structure

| File | Description |
| :--- | :--- |
| **`index.html`** | 100% standalone, zero-dependency browser application |
| **`gpu_engine.js`** | WebGPU hardware acceleration engine |
| **`kangaroo_engine.js`** | Pollard's Kangaroo elliptic curve collision library |
| **`all_160_puzzles.json`** | All 160 Bitcoin puzzle target ranges and addresses dataset |
| **`manifest.json`** | Progressive Web App (PWA) manifest |
| **`sw.js`** | Offline Service Worker caching script |
| **`icon-192.png`** / **`icon-512.png`** | High-resolution PWA application icons |
| **`ui_preview.png`** | Interface preview screenshot |
| **`LICENSE`** | MIT Open-Source License |
| **`README.md`** | Project documentation |

---

## ☕ Support & Donations

If this open-source tool helped your research or cryptographic experiments, voluntary Bitcoin donations are warmly appreciated:

```text
bc1qxf5cfrxasshlkt79x0q805l9t3feer868en68nhlxmwetlr6sv4qdfda5s
```

---

## ⚖️ License & Disclaimer

Distributed under the **MIT License**. See `LICENSE` for more information.  
*This software is developed strictly for educational, mathematical research, and cryptographic analysis purposes.*
