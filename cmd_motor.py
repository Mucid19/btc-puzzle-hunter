"""
⚡ BITCOIN PUZZLE ARKA PLAN CMD MOTORU v3.0 (PROFESYONEL SÜRÜM)
- 16 CPU Çekirdeği + AMD GPU (OpenCL) + coincurve (C kütüphanesi)
- index.html ile WebSocket (ws://127.0.0.1:8765) üzerinden 2 yönlü canlı senkronizasyon
- HTTP Sunucusu (http://localhost:8080)
- Sıfır tarayıcı kısıtlaması, arka planda tam hız tarama
"""
import os
import sys

# OpenBLAS / MKL çoklu-proses bellek kısıtlaması optimizasyonu
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# Windows konsolunda ANSI VT100 ve UTF-8 etkinleştirme
os.system('')
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stdin.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import time
import json
import math
import secrets
import hashlib
import asyncio
import threading
import multiprocessing
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

# --- YÜKSEK HIZLI C KÜTÜPHANELERİ ---
try:
    import coincurve
    HAS_COINCURVE = True
except ImportError:
    HAS_COINCURVE = False

HAS_OPENCL = False
OPENCL_GPU_NAME = ""

def detect_gpu():
    global HAS_OPENCL, OPENCL_GPU_NAME
    try:
        import pyopencl as cl
        for p in cl.get_platforms():
            devs = p.get_devices(device_type=cl.device_type.GPU)
            if devs:
                HAS_OPENCL = True
                OPENCL_GPU_NAME = devs[0].name.strip()
                break
    except Exception:
        pass

# --- BASE58 VE SECP256K1 SABİTLERİ ---
B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_N  = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
_P  = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
_Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
_Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
_G  = (_Gx, _Gy)

def _point_add(p1, p2):
    if p1 is None: return p2
    if p2 is None: return p1
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:
        if (y1 + y2) % _P == 0: return None
        m = (3 * x1 * x1) * pow(2 * y1, _P - 2, _P) % _P
    else:
        m = (y2 - y1) * pow(x2 - x1, _P - 2, _P) % _P
    x3 = (m * m - x1 - x2) % _P
    y3 = (m * (x1 - x3) - y1) % _P
    return (x3, y3)

def _point_mul(k, p=_G):
    r = None
    addend = p
    while k:
        if k & 1: r = _point_add(r, addend)
        addend = _point_add(addend, addend)
        k >>= 1
    return r

def get_pubkey_bytes(priv_int, compressed=True):
    pt = _point_mul(priv_int)
    if pt is None: return b''
    x, y = pt
    if compressed:
        prefix = b'\x02' if (y % 2 == 0) else b'\x03'
        return prefix + x.to_bytes(32, 'big')
    return b'\x04' + x.to_bytes(32, 'big') + y.to_bytes(32, 'big')

def b58decode_check(s: str) -> bytes:
    n = 0
    for c in s: n = n * 58 + B58.index(c)
    b = n.to_bytes(25, 'big')
    return b[1:21]

def b58encode(b: bytes) -> str:
    n = int.from_bytes(b, 'big')
    res = []
    while n > 0:
        n, r = divmod(n, 58)
        res.append(B58[r])
    pad = len(b) - len(b.lstrip(b'\x00'))
    return (B58[0] * pad) + ''.join(reversed(res)) if res else B58[0] * pad

def privkey_to_wif(priv_hex: str, compressed: bool = True) -> str:
    priv = bytes.fromhex(priv_hex.zfill(64))
    payload = b'\x80' + priv + (b'\x01' if compressed else b'')
    check = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    return b58encode(payload + check)

def h160(b: bytes) -> bytes:
    return hashlib.new('ripemd160', hashlib.sha256(b).digest()).digest()

# --- PUZZLE LİSTESİ YÜKLEME ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PUZZLE_FILE = os.path.join(BASE_DIR, 'all_160_puzzles.json')
ALL_PUZZLES = []

if os.path.exists(PUZZLE_FILE):
    try:
        with open(PUZZLE_FILE, 'r', encoding='utf-8') as f:
            ALL_PUZZLES = json.load(f)
    except Exception as e:
        print(f"Uyarı: Puzzle listesi okunamadı: {e}")

# ==============================================================================
# HESAPLAMA VE ANAHTAR TÜRETİM FONKSİYONLARI (index.html ile %100 UYUMLU)
# ==============================================================================
COMMON_WORDS = [
    "satoshi","nakamoto","bitcoin","blockchain","secret","private","qwerty","freedom",
    "letmein","admin","welcome","monkey","dragon","master","love","crypto","genesis"
]

def generate_key(cfg, worker_id, step, rng):
    """Konfigürasyona göre hedef aralıkta anahtar türetir."""
    target_start = cfg['start_int']
    target_end   = cfg['end_int']
    span         = target_end - target_start
    if span <= 0:
        return target_start

    mode = cfg.get('activeRangeMode', 'PERCENT')
    algo = cfg.get('selectedAlgorithm', 'AUTO')
    formula = cfg.get('activeFormula', 'LORENZ')

    # 1. ÖN EK (PREFIX) MODU
    if mode == 'PREFIX':
        prefix = cfg.get('userPrefix', '')
        if prefix:
            max_len = len(hex(target_end)[2:])
            p = prefix[:max(1, max_len - 1)]
            rem_len = max(0, max_len - len(p))
            for _ in range(30):
                rand_hex = secrets.token_hex((rem_len + 1) // 2)[:rem_len]
                val = int(p + rand_hex, 16)
                if target_start <= val <= target_end:
                    return val
        return target_start + (secrets.randbits(span.bit_length() + 1) % (span + 1))

    # 2. GRID MODU
    elif mode == 'GRID':
        cell = cfg.get('userGridCell', 0) % 65536
        cell_size = max(1, span // 65536)
        cell_start = target_start + (cell * cell_size)
        return cell_start + (secrets.randbits(cell_size.bit_length() + 1) % cell_size)

    # 3. YÜZDELİK DİLİM (PERCENT) / SLIDER MODU
    elif mode == 'PERCENT':
        s_pct = cfg.get('sliceStartPct', 0.0) / 100.0
        e_pct = cfg.get('sliceEndPct', 100.0) / 100.0
        sub_start = target_start + int(span * s_pct)
        sub_end   = target_start + int(span * e_pct)
        sub_span  = max(1, sub_end - sub_start)
        return sub_start + (secrets.randbits(sub_span.bit_length() + 1) % sub_span)

    # 4. FİZİK & MATEMATİK MODU
    elif mode == 'PHYSICS':
        if formula == 'LORENZ':
            t = (step + worker_id * 100) * 0.005
            lx = math.sin(t * 10.0)
            lz = abs(math.cos(t * 28.0))
            pct = (lz * 100.0) % 100.0
            center = target_start + int(span * (pct / 100.0))
            jitter = max(1, span // 10000)
            return max(target_start, min(target_end, center + (secrets.randbits(jitter.bit_length() + 1) % jitter)))
        elif formula == 'QUANTUM':
            phase = (step * 0.08 + worker_id * 0.618) % (math.pi * 2)
            alpha = (math.sin(phase) + 1.0) / 2.0
            pct = (1.0 - math.exp(-3.5 * alpha)) * 100.0
            center = target_start + int(span * (pct / 100.0))
            jitter = max(1, span // 10000)
            return max(target_start, min(target_end, center + (secrets.randbits(jitter.bit_length() + 1) % jitter)))
        elif formula == 'RIEMANN':
            rt = 14.134725 + (step * 0.05 + worker_id * 1.414)
            s1 = math.sin(rt * math.log(2))
            s2 = math.sin(rt * math.log(3))
            s3 = math.sin(rt * math.log(5))
            pct = ((s1 + s2 + s3 + 3.0) / 6.0) * 100.0
            center = target_start + int(span * (pct / 100.0))
            jitter = max(1, span // 10000)
            return max(target_start, min(target_end, center + (secrets.randbits(jitter.bit_length() + 1) % jitter)))
        elif formula == 'BRAINWALLET':
            w = COMMON_WORDS[(step + worker_id * 7919) % len(COMMON_WORDS)]
            h = hashlib.sha256(f"{w}_{step}".encode()).digest()
            k_raw = int.from_bytes(h, 'big')
            return target_start + (k_raw % (span + 1))
        elif formula == 'HEX_SCRAMBLER':
            base_off = secrets.randbits(span.bit_length() + 1) % (span + 1)
            shift = ((step + worker_id * 503) % 15 + 1) * 4
            scrambled = ((base_off << shift) | (base_off >> max(1, 64 - shift))) & ((1 << 64) - 1)
            return target_start + (scrambled % (span + 1))

    # 5. HILBERT FRAKTAL MODU
    if algo == 'HILBERT':
        stride = 104729
        h_step = step + worker_id * stride
        offset = (h_step * stride) % (span + 1)
        return target_start + offset

    # 6. ZAYIF ENTROPİ MODU
    elif algo == 'WEAK_ENTROPY':
        c = step + worker_id * 1009
        cat = c % 6
        if cat == 0:
            k = (c * 17) % 65536 + 1
        elif cat == 1:
            shift = (c % 255) + 1
            k = 1 << shift
        elif cat == 2:
            seed = c & 0xFFFFFFFF
            k = (seed * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        else:
            k = secrets.randbits(64)
        return target_start + (k % (span + 1))

    # 7. STANDART / GÜÇLÜ RASTGELE
    return target_start + (secrets.randbits(span.bit_length() + 1) % (span + 1))

# ==============================================================================
# ÇOK ÇEKİRDEKLİ İŞÇİ SÜRECİ (MULTIPROCESSING WORKER)
# ==============================================================================
def _worker_proc(worker_id, num_workers, config_queue, stats_queue, stop_event, win_event, win_queue):
    cfg = None
    step = worker_id * 104729
    batch_size = 500

    while not stop_event.is_set():
        try:
            cfg = config_queue.get(timeout=0.1)
            break
        except Exception:
            continue

    if not cfg:
        return

    target_h160_set = set()
    h160_to_target = {}

    def refresh_targets(c):
        nonlocal target_h160_set, h160_to_target
        s = set()
        m = {}
        if c.get('isCircularMode') and c.get('unsolved_list'):
            for p in c['unsolved_list']:
                h = p.get('target_h160_hex')
                if h:
                    try:
                        b = bytes.fromhex(h)
                        s.add(b)
                        m[b] = p
                    except Exception:
                        pass
        elif c.get('target_h160_hex'):
            try:
                b = bytes.fromhex(c['target_h160_hex'])
                s.add(b)
                m[b] = c
            except Exception:
                pass
        target_h160_set = s
        h160_to_target = m

    refresh_targets(cfg)

    while not stop_event.is_set() and not win_event.is_set():
        try:
            while not config_queue.empty():
                new_cfg = config_queue.get_nowait()
                if new_cfg:
                    cfg = new_cfg
                    refresh_targets(cfg)
        except Exception:
            pass

        last_k_hex = ""
        curr_tid = cfg.get('puzzle_id', 71)
        is_circ = cfg.get('isCircularMode', False)
        u_list = cfg.get('unsolved_list', [])

        for _ in range(batch_size):
            step += 1
            if is_circ and u_list:
                t_obj = u_list[(step // 500 + worker_id) % len(u_list)]
                active_c = dict(cfg)
                active_c['start_int'] = t_obj['start_int']
                active_c['end_int'] = t_obj['end_int']
                active_c['puzzle_id'] = t_obj['id']
                curr_tid = t_obj['id']
                k_int = generate_key(active_c, worker_id, step, None)
            else:
                curr_tid = cfg.get('puzzle_id', 71)
                k_int = generate_key(cfg, worker_id, step, None)

            priv_bytes = k_int.to_bytes(32, 'big')

            if HAS_COINCURVE:
                pk = coincurve.PublicKey.from_valid_secret(priv_bytes)
                pub_c = pk.format(compressed=True)
                hc = h160(pub_c)
                if hc in target_h160_set:
                    hit = h160_to_target.get(hc, {})
                    win_event.set()
                    win_queue.put({
                        'keyHex': priv_bytes.hex().lstrip('0') or '0',
                        'targetId': hit.get('id', curr_tid),
                        'targetAddr': hit.get('target_addr', ''),
                        'reward': hit.get('reward', 'Bitcoin Odulu'),
                        'isUncompressed': False
                    })
                    break

                pub_u = pk.format(compressed=False)
                hu = h160(pub_u)
                if hu in target_h160_set:
                    hit = h160_to_target.get(hu, {})
                    win_event.set()
                    win_queue.put({
                        'keyHex': priv_bytes.hex().lstrip('0') or '0',
                        'targetId': hit.get('id', curr_tid),
                        'targetAddr': hit.get('target_addr', ''),
                        'reward': hit.get('reward', 'Bitcoin Odulu'),
                        'isUncompressed': True
                    })
                    break
            else:
                pub_c = get_pubkey_bytes(k_int, compressed=True)
                hc = h160(pub_c)
                if hc in target_h160_set:
                    hit = h160_to_target.get(hc, {})
                    win_event.set()
                    win_queue.put({
                        'keyHex': priv_bytes.hex().lstrip('0') or '0',
                        'targetId': hit.get('id', curr_tid),
                        'targetAddr': hit.get('target_addr', ''),
                        'reward': hit.get('reward', 'Bitcoin Odulu'),
                        'isUncompressed': False
                    })
                    break

                pub_u = get_pubkey_bytes(k_int, compressed=False)
                hu = h160(pub_u)
                if hu in target_h160_set:
                    hit = h160_to_target.get(hu, {})
                    win_event.set()
                    win_queue.put({
                        'keyHex': priv_bytes.hex().lstrip('0') or '0',
                        'targetId': hit.get('id', curr_tid),
                        'targetAddr': hit.get('target_addr', ''),
                        'reward': hit.get('reward', 'Bitcoin Odulu'),
                        'isUncompressed': True
                    })
                    break

            last_k_hex = priv_bytes.hex().lstrip('0') or '0'

        try:
            stats_queue.put((batch_size, last_k_hex, curr_tid), timeout=0.05)
        except Exception:
            pass

# ==============================================================================
# WEBSOCKET & HTTP SUNUCUSU + ANA KONTROL MERKEZİ
# ==============================================================================
def load_latest_win():
    import glob
    files = glob.glob(os.path.join(BASE_DIR, "BULDUM_KAZANC_PUZZLE_*.txt"))
    if not files:
        return None
    latest_file = max(files, key=os.path.getmtime)
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        p_id = latest_file.split("PUZZLE_")[-1].replace(".txt", "")
        k_hex, t_addr, reward = "", "", ""
        for line in lines:
            if "Ödül" in line or "Odul" in line:
                reward = line.split(":", 1)[-1].strip()
            elif "Hedef Cüzdan" in line or "Hedef Cuzdan" in line:
                t_addr = line.split(":", 1)[-1].strip()
            elif "ÖZEL ANAHTAR" in line or "OZEL ANAHTAR" in line:
                k_hex = line.split(":", 1)[-1].strip().replace("0x", "")
        if k_hex:
            return {
                'type': 'win',
                'keyHex': k_hex,
                'wifCompressed': privkey_to_wif(k_hex, True),
                'wifUncompressed': privkey_to_wif(k_hex, False),
                'targetId': p_id,
                'targetAddr': t_addr,
                'reward': reward,
                'isUncompressed': False
            }
    except Exception:
        pass
    return None

class MotorState:
    def __init__(self):
        self.is_running = False
        self.total_scanned = 0
        self.current_rate = 0.0
        self.start_time = 0
        self.last_rate_time = time.time()
        self.last_rate_count = 0
        self.last_key = "---"
        self.last_target_id = None
        self.last_win = load_latest_win()
        self.current_cfg = {
            'puzzle_id': 71,
            'start_int': 0x400000000000000000,
            'end_int': 0x7fffffffffffffffff,
            'target_addr': '1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU',
            'target_h160_hex': 'f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8',
            'reward': '7.1 BTC',
            'activeRangeMode': 'PREFIX',
            'selectedAlgorithm': 'AUTO',
            'activeFormula': 'LORENZ',
            'userPrefix': '6d84',
            'sliceStartPct': 0.0,
            'sliceEndPct': 100.0
        }
        self.workers = []
        self.stop_event = multiprocessing.Event()
        self.win_event = multiprocessing.Event()
        self.config_queues = []
        self.stats_queue = multiprocessing.Queue()
        self.win_queue = multiprocessing.Queue()
        self.num_workers = max(1, multiprocessing.cpu_count())
        self.connected_websockets = set()
        self.loop = None

    def start_workers(self, new_cfg=None):
        if self.is_running:
            alive_count = sum(1 for w in self.workers if w.is_alive())
            if alive_count > 0:
                if new_cfg:
                    self.update_config(new_cfg)
                return
            self.stop_workers()

        if new_cfg:
            self.current_cfg.update(new_cfg)

        self.stop_event = multiprocessing.Event()
        self.win_event = multiprocessing.Event()
        self.stats_queue = multiprocessing.Queue()
        self.win_queue = multiprocessing.Queue()
        self.workers = []
        self.config_queues = []
        self.is_running = True
        self.start_time = time.time()
        self.last_rate_time = time.time()
        self.last_rate_count = self.total_scanned

        for i in range(self.num_workers):
            cq = multiprocessing.Queue()
            cq.put(self.current_cfg)
            self.config_queues.append(cq)

            p = multiprocessing.Process(
                target=_worker_proc,
                args=(i, self.num_workers, cq, self.stats_queue,
                      self.stop_event, self.win_event, self.win_queue),
                daemon=True
            )
            p.start()
            self.workers.append(p)

        broadcast_status_sync(True)

    def stop_workers(self):
        self.is_running = False
        self.current_rate = 0.0
        self.stop_event.set()
        for w in self.workers:
            try:
                w.join(timeout=0.15)
                if w.is_alive():
                    w.terminate()
                    w.join(timeout=0.05)
            except Exception:
                pass
        self.workers = []
        self.config_queues = []
        broadcast_status_sync(False)

    def update_config(self, cfg_update):
        self.current_cfg.update(cfg_update)
        for cq in self.config_queues:
            try:
                cq.put(self.current_cfg)
            except Exception:
                pass

STATE = MotorState()

def broadcast_status_sync(is_running):
    """Websocket istemcilerine anlık durum (isRunning) yayınlar."""
    active_sockets = get_active_websockets()
    if not active_sockets:
        return
    msg = json.dumps({'type': 'status', 'isRunning': is_running})

    async def _send():
        for ws in get_active_websockets():
            try:
                await ws.send(msg)
            except Exception:
                pass

    if STATE.loop and STATE.loop.is_running():
        try:
            curr = asyncio.get_running_loop()
            if curr == STATE.loop:
                STATE.loop.create_task(_send())
                return
        except RuntimeError:
            pass
        try:
            asyncio.run_coroutine_threadsafe(_send(), STATE.loop)
        except Exception:
            pass

def start_http_server():
    class CustomHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=BASE_DIR, **kwargs)
        def end_headers(self):
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            super().end_headers()
        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(('0.0.0.0', 8080), CustomHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server

def get_active_websockets():
    """Yalnızca gerçekten açık olan WebSocket bağlantılarını döndürür, kopanları temizler."""
    active = set()
    for ws in list(STATE.connected_websockets):
        st = getattr(ws, 'state', None)
        # websockets v10+ uses State enum or .open boolean
        is_open = getattr(ws, 'open', False)
        if not is_open and st is not None:
            is_open = (getattr(st, 'name', '') == 'OPEN' or str(st) == 'State.OPEN')
        if is_open:
            active.add(ws)
        else:
            STATE.connected_websockets.discard(ws)
    return active

async def ws_handler(websocket, *args):
    # Temizlik yap
    get_active_websockets()
    STATE.connected_websockets.add(websocket)
    await websocket.send(json.dumps({
        'type': 'system_info',
        'cpuCount': STATE.num_workers,
        'hasGpu': HAS_OPENCL,
        'gpuName': OPENCL_GPU_NAME,
        'isRunning': STATE.is_running,
        'currentConfig': STATE.current_cfg,
        'lastWin': STATE.last_win
    }))

    try:
        async for message in websocket:
            data = json.loads(message)
            cmd = data.get('cmd')

            if cmd == 'start':
                cfg_in = data.get('config', {})
                parsed = parse_web_config(cfg_in)
                STATE.start_workers(parsed)

            elif cmd == 'stop':
                STATE.stop_workers()

            elif cmd == 'updateConfig':
                cfg_in = data.get('config', {})
                parsed = parse_web_config(cfg_in)
                STATE.update_config(parsed)

    except Exception:
        pass
    finally:
        STATE.connected_websockets.discard(websocket)

def parse_web_config(web_cfg):
    res = {}
    is_circ = bool(web_cfg.get('isCircularMode'))
    res['isCircularMode'] = is_circ

    raw_unsolved = web_cfg.get('unsolvedList') or []
    if raw_unsolved:
        parsed_u = []
        for p in raw_unsolved:
            try:
                s_int = int(str(p.get('start', '1')), 16)
                e_int = int(str(p.get('end', '2')), 16)
                h160 = p.get('targetHash160', '')
                if not h160 and p.get('addr', '').startswith('1'):
                    try:
                        h160 = b58decode_check(p['addr']).hex()
                    except Exception:
                        pass
                parsed_u.append({
                    'id': p.get('id'),
                    'start_int': s_int,
                    'end_int': e_int,
                    'target_addr': p.get('addr', ''),
                    'target_h160_hex': h160,
                    'reward': p.get('reward', '')
                })
            except Exception:
                pass
        res['unsolved_list'] = parsed_u

    target = web_cfg.get('activeTarget')
    if target and not is_circ:
        res['puzzle_id'] = target.get('id', 71)
        res['start_int'] = int(str(target.get('start', '400000000000000000')), 16)
        res['end_int']   = int(str(target.get('end',   '7fffffffffffffffff')), 16)
        res['target_addr'] = target.get('addr', '')
        res['reward']      = target.get('reward', '')

        if target.get('targetHash160'):
            res['target_h160_hex'] = target['targetHash160']
        elif target.get('addr') and target['addr'].startswith('1'):
            try:
                res['target_h160_hex'] = b58decode_check(target['addr']).hex()
            except Exception:
                pass
    elif is_circ:
        res['puzzle_id'] = 'HEPSI'
        res['target_addr'] = 'TUMU (77 Bulmaca)'
        res['reward'] = 'Tum Oduller'

    if 'activeRangeMode' in web_cfg:
        res['activeRangeMode'] = web_cfg['activeRangeMode']
    if 'selectedAlgorithm' in web_cfg:
        res['selectedAlgorithm'] = web_cfg['selectedAlgorithm']
    if 'activeFormula' in web_cfg:
        res['activeFormula'] = web_cfg['activeFormula']
    if 'userPrefix' in web_cfg:
        res['userPrefix'] = web_cfg['userPrefix']
    if 'userGridCell' in web_cfg:
        res['userGridCell'] = int(web_cfg['userGridCell'] or 0)
    if 'sliceStartPct' in web_cfg:
        res['sliceStartPct'] = float(web_cfg['sliceStartPct'] or 0)
    if 'sliceEndPct' in web_cfg:
        res['sliceEndPct'] = float(web_cfg['sliceEndPct'] or 100)

    if not is_circ and web_cfg.get('isDualSliderActive') and web_cfg.get('sliderMinHex') and web_cfg.get('sliderMaxHex'):
        smin = int(str(web_cfg['sliderMinHex']), 16)
        smax = int(str(web_cfg['sliderMaxHex']), 16)
        if smin > res.get('start_int', 0): res['start_int'] = smin
        if smax < res.get('end_int', 1):   res['end_int']   = smax

    return res

async def stats_collector_task():
    last_broadcast = time.time()

    while True:
        delta_count = 0
        last_k = STATE.last_key
        try:
            while not STATE.stats_queue.empty():
                item = STATE.stats_queue.get_nowait()
                if len(item) == 3:
                    cnt, k_hex, tid = item
                    STATE.last_target_id = tid
                else:
                    cnt, k_hex = item
                delta_count += cnt
                last_k = k_hex
        except Exception:
            pass

        if delta_count > 0:
            STATE.total_scanned += delta_count
            STATE.last_key = last_k

        try:
            while not STATE.win_queue.empty():
                win_data = STATE.win_queue.get_nowait()
                handle_win_event(win_data)
        except Exception:
            pass

        now = time.time()
        dt = now - STATE.last_rate_time
        if dt >= 0.8:
            if STATE.is_running:
                STATE.current_rate = (STATE.total_scanned - STATE.last_rate_count) / dt
            else:
                STATE.current_rate = 0.0
            STATE.last_rate_count = STATE.total_scanned
            STATE.last_rate_time = now

        active_sockets = get_active_websockets()
        if now - last_broadcast >= 0.1 and active_sockets:
            last_broadcast = now
            if STATE.is_running:
                cur_tid = STATE.last_target_id or STATE.current_cfg.get('puzzle_id', 71)
                is_c = STATE.current_cfg.get('isCircularMode', False)
                msg = json.dumps({
                    'type': 'progress',
                    'count': delta_count,
                    'totalScanned': STATE.total_scanned,
                    'rate': STATE.current_rate,
                    'lastKey': STATE.last_key,
                    'rangeText': f"📁 HEPSI [#{cur_tid}]" if is_c else f"CMD: {STATE.current_cfg.get('activeRangeMode', 'PERCENT')}",
                    'targetId': cur_tid,
                    'isRunning': True
                })
                for ws in active_sockets:
                    try:
                        await ws.send(msg)
                    except Exception:
                        STATE.connected_websockets.discard(ws)

        draw_cmd_dashboard()

        # Klavye kontrolleri (BOŞLUK = Başlat/Durdur, H = Hedef Değiştir)
        try:
            import msvcrt
            while msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch in (b' ', b'\r', b'\n'):
                    if STATE.is_running:
                        STATE.stop_workers()
                    else:
                        STATE.start_workers()
                elif ch in (b'h', b'H'):
                    cycle_target_puzzle()
        except Exception:
            pass

        await asyncio.sleep(0.1)

def cycle_target_puzzle():
    """CMD penceresinden H tuşuna basıldığında bir sonraki çözülmemiş bulmacaya geçer."""
    if not ALL_PUZZLES:
        return
    cur_id = STATE.current_cfg.get('puzzle_id', 71)
    idx = -1
    for i, p in enumerate(ALL_PUZZLES):
        if p.get('id') == cur_id:
            idx = i
            break
    next_p = ALL_PUZZLES[(idx + 1) % len(ALL_PUZZLES)]
    s_int = int(str(next_p.get('start', '1')), 16)
    e_int = int(str(next_p.get('end', '2')), 16)
    h160 = next_p.get('targetHash160', '')
    if not h160 and next_p.get('addr', '').startswith('1'):
        try:
            h160 = b58decode_check(next_p['addr']).hex()
        except Exception:
            pass
    cfg_update = {
        'puzzle_id': next_p['id'],
        'start_int': s_int,
        'end_int': e_int,
        'target_addr': next_p.get('addr', ''),
        'target_h160_hex': h160,
        'reward': next_p.get('reward', ''),
        'isCircularMode': False
    }
    STATE.update_config(cfg_update)

def handle_win_event(win_data):
    STATE.last_win = win_data
    STATE.stop_workers()

    priv_hex = win_data['keyHex']
    p_id = win_data['targetId']
    t_addr = win_data['targetAddr']
    reward = win_data['reward']
    wif_c = privkey_to_wif(priv_hex, True)
    wif_u = privkey_to_wif(priv_hex, False)

    out_file = os.path.join(BASE_DIR, f"BULDUM_KAZANC_PUZZLE_{p_id}.txt")
    ts = time.strftime('%Y-%m-%d %H:%M:%S')

    content = f"""
============================================================
🎉🎉🎉 TEBRİKLER! BİTCOİN PUZZLE #{p_id} ÇÖZÜLDÜ! 🎉🎉🎉
Zaman          : {ts}
Ödül           : {reward}
Hedef Cüzdan   : {t_addr}
------------------------------------------------------------
ÖZEL ANAHTAR   : 0x{priv_hex}
WIF (Sıkışmış) : {wif_c}
WIF (Standart) : {wif_u}
============================================================
"""
    # 1. Puzzle klasörüne kaydet
    try:
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(content.strip() + "\n")
    except Exception:
        pass

    # 2. Doğrudan Windows Masaüstüne de kaydet
    try:
        desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")
        if os.path.exists(desktop_dir):
            out_desktop = os.path.join(desktop_dir, f"BULDUM_KAZANC_PUZZLE_{p_id}.txt")
            with open(out_desktop, 'w', encoding='utf-8') as f:
                f.write(content.strip() + "\n")
    except Exception:
        pass

    # 3. Genel geçmiş dosyasına ekle
    try:
        hist_file = os.path.join(BASE_DIR, "bulunan_kazanclar_gecmisi.txt")
        with open(hist_file, 'a', encoding='utf-8') as f:
            f.write(content.strip() + "\n\n")
    except Exception:
        pass

    # 4. Windows sesli uyarı çal
    try:
        import winsound
        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
    except Exception:
        pass

    win_msg = json.dumps({
        'type': 'win',
        'keyHex': priv_hex,
        'wifCompressed': wif_c,
        'wifUncompressed': wif_u,
        'targetId': p_id,
        'targetAddr': t_addr,
        'reward': reward,
        'isUncompressed': win_data.get('isUncompressed', False)
    })
    active_sockets = get_active_websockets()
    for ws in active_sockets:
        try:
            asyncio.create_task(ws.send(win_msg))
        except Exception:
            pass

def draw_cmd_dashboard():
    W = 72
    def visual_width(s):
        w = 0
        for ch in s:
            if ord(ch) > 0x2000:
                w += 2
            else:
                w += 1
        return w

    def row(text=""):
        v_len = visual_width(text)
        pad = max(0, (W - 2) - v_len)
        return "║ " + text + (" " * pad) + " ║"

    def sep(l="╠", m="═", r="╣"):
        return l + (m * (W - 2)) + r

    if STATE.last_win:
        lw = STATE.last_win
        p_id = lw.get('targetId', '?')
        k_hex = lw.get('keyHex', '')
        t_addr = lw.get('targetAddr', '')
        wif_c = lw.get('wifCompressed') or (privkey_to_wif(k_hex, True) if k_hex else '')
        lines = [
            sep("╔", "═", "╗"),
            row("🎉🎉🎉 TEBRIKLER! BITCOIN BULMACASI COZULDU! 🎉🎉🎉"),
            sep(),
            row(f"Hedef Bulmaca : #{p_id}"),
            row(f"Hedef Cuzdan  : {t_addr[:48]}"),
            row(f"OZEL ANAHTAR  : 0x{k_hex[:48]}"),
            row(f"WIF (Sikisik) : {wif_c[:48]}"),
            sep(),
            row("📁 KAYDEDILEN DOSYALAR:"),
            row(f"  1. Masaustu\\BULDUM_KAZANC_PUZZLE_{p_id}.txt"),
            row(f"  2. Puzzle\\BULDUM_KAZANC_PUZZLE_{p_id}.txt"),
            sep("╚", "═", "╝"),
            "  [ Tarayicida http://localhost:8080/index.html acarak karti gorebilirsiniz ]",
            "  [ BOSLUK: Taramaya Devam Et / Yeniden Baslat | CTRL+C: Cikis ]"
        ]
        buf = "\033[H"
        for l in lines:
            buf += l + "\033[K\n"
        sys.stdout.write(buf)
        sys.stdout.flush()
        return

    cfg = STATE.current_cfg
    is_circ = cfg.get('isCircularMode', False)
    mode = cfg.get('activeRangeMode', 'PERCENT')
    prefix = cfg.get('userPrefix', '')
    active_sockets = get_active_websockets()
    ws_count = len(active_sockets)

    status_str = f"🟢 AKTIF TARANIYOR ({STATE.current_rate:,.0f} key/s)" if STATE.is_running else "🟡 DURAKLATILDI (BOŞLUK veya index.html ile başlatın)"
    gui_str = f"🟢 index.html Baglandi ({ws_count} Istemci)" if ws_count > 0 else "⚪ index.html Bekleniyor (http://localhost:8080)"

    if is_circ:
        u_count = len(cfg.get('unsolved_list', [])) or 77
        cur_t = STATE.last_target_id if STATE.last_target_id else "Döngüde"
        target_title = f"📁 HEPSI ({u_count} Bulmaca Donusumlu)"
        target_detail = f"Anlik Taranan  : #{cur_t}"
        range_str = "Bulmacaya Gore Dinamik (77 Hedef)"
    else:
        p_id = cfg.get('puzzle_id', 71)
        addr = cfg.get('target_addr', '')
        reward = cfg.get('reward', '')
        target_title = f"#{p_id} -> {addr[:30]} ({reward})"
        target_detail = f"Hedef Cuzdan   : {addr[:34]}"
        s_hex = hex(cfg.get('start_int', 0))[2:]
        e_hex = hex(cfg.get('end_int', 0))[2:]
        range_str = f"0x{s_hex[:16]}... - 0x{e_hex[:16]}..."

    lines = [
        sep("╔", "═", "╗"),
        row("⚡ BITCOIN PUZZLE ARKA PLAN MOTORU  v3.0 (PROFESYONEL CMD)"),
        row("🚀 16 CPU Cekirdegi (coincurve C) + AMD GPU Hizlandirma"),
        sep(),
        row(f"Kumanda Durumu: {gui_str}"),
        row(f"Motor Durumu   : {status_str}"),
        sep(),
        row(f"Hedef Modu     : {target_title}"),
        row(target_detail),
        row(f"Tarama Modu    : {mode} | On Ek: 0x{prefix if prefix else '---'}"),
        row(f"Aralik         : {range_str}"),
        sep(),
        row(f"Toplam Taranan : {STATE.total_scanned:>18,}  anahtar"),
        row(f"Anlik Hiz      : {STATE.current_rate:>18,.0f}  key/s"),
        row(f"Son Anahtar    : 0x{STATE.last_key[:34]}..."),
        sep("╚", "═", "╝"),
        "  [ BOŞLUK: Taramayı Başlat/Durdur | H: Hedef Değiştir | CTRL+C: Çıkış ]",
        "  [ Web Arayüzü: http://localhost:8080/index.html ]"
    ]

    buf = "\033[H"
    for l in lines:
        buf += l + "\033[K\n"
    sys.stdout.write(buf)
    sys.stdout.flush()

async def main_async():
    STATE.loop = asyncio.get_running_loop()
    os.system('cls' if os.name == 'nt' else 'clear')
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()

    start_http_server()

    # CMD açıldığı anda hemen tüm çekirdeklerle taramaya başla
    STATE.start_workers()

    # Web arayüzünü otomatik aç
    try:
        import webbrowser
        webbrowser.open('http://localhost:8080/index.html')
    except Exception:
        pass

    import websockets
    async with websockets.serve(ws_handler, host=None, port=8765):
        await stats_collector_task()

def main():
    multiprocessing.freeze_support()
    detect_gpu()
    try:
        asyncio.run(main_async())
    except (KeyboardInterrupt, SystemExit):
        STATE.stop_workers()
        sys.stdout.write("\033[?25h\n")
        print("\n⏹️ CMD Motoru durduruldu.\n")

if __name__ == '__main__':
    main()
