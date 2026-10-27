"""Fireworks embeddings (qwen3-embedding-8b, 4096-d) with an on-disk cache keyed by sha256 of the text.
Cache: results/T11/embeddings.npz (keys: 'hashes' (str array), 'emb' (float16 [n, 4096])).
Usage: from tasks.T11.embed import embed_texts; E = embed_texts(list_of_str)  -> float32 [n, 4096], l2-normalised rows."""
import hashlib, os, time, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np, requests
from harness import config
from harness.client import _load_env
from tasks.T11.common import OUT

_load_env()
API_KEY = os.environ.get("FIREWORKS_API_KEY")
URL = "https://api.fireworks.ai/inference/v1/embeddings"
MODEL = "accounts/fireworks/models/qwen3-embedding-8b"
CACHE = OUT / "embeddings.npz"
BATCH = 64
_lock = threading.Lock()

def _h(t): return hashlib.sha256(t.encode("utf-8")).hexdigest()

def _load_cache():
    if not CACHE.exists(): return {}, None
    z = np.load(CACHE, allow_pickle=False)
    return {h: i for i, h in enumerate(z["hashes"].tolist())}, z["emb"]

def _save_cache(hashes, emb):
    tmp = CACHE.with_suffix(".tmp.npz")
    np.savez(tmp, hashes=np.array(hashes), emb=emb.astype(np.float16))
    os.replace(tmp, CACHE)

def _call(batch, retries=8):
    last = None
    for a in range(retries):
        try:
            r = requests.post(URL, json={"model": MODEL, "input": batch}, headers={"Authorization": f"Bearer {API_KEY}"}, timeout=180)
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"{r.status_code}: {r.text[:200]}"; time.sleep(min(60, 2 ** a)); continue
            d = r.json()
            if "data" not in d: last = str(d)[:300]; time.sleep(min(60, 2 ** a)); continue
            rows = sorted(d["data"], key=lambda x: x["index"])
            return np.array([x["embedding"] for x in rows], dtype=np.float32), d.get("usage", {}).get("total_tokens", 0)
        except (requests.RequestException, ValueError) as e:
            last = repr(e); time.sleep(min(60, 2 ** a))
    raise RuntimeError(f"embedding failed after {retries} retries: {last}")

def embed_texts(texts, workers=4, verbose=True):
    """Returns float32 [len(texts), 4096], l2-normalised. Empty strings get a zero vector (never sent to the API)."""
    texts = [t if isinstance(t, str) else "" for t in texts]
    idx, emb = _load_cache()
    hashes = [_h(t) for t in texts]
    need = sorted({h for h, t in zip(hashes, texts) if t.strip() and h not in idx})
    if need:
        by_h = {}
        for h, t in zip(hashes, texts):
            if h in need and h not in by_h: by_h[h] = t
        order = list(need); batches = [order[i:i + BATCH] for i in range(0, len(order), BATCH)]
        if verbose: print(f"embed: {len(order)} new texts in {len(batches)} batches ({len(idx)} cached)", flush=True)
        new = {}; toks = [0]; t0 = time.time()
        def one(bi, hs):
            # single-text fallback so one bad text never kills a batch
            try:
                E, n = _call([by_h[h] for h in hs])
            except Exception as e:
                print(f"  batch {bi} failed ({str(e)[:100]}); retrying one by one", flush=True)
                E = np.zeros((len(hs), 4096), np.float32); n = 0
                for j, h in enumerate(hs):
                    try: e1, n1 = _call([by_h[h]]); E[j] = e1[0]; n += n1
                    except Exception as e2: print(f"  text {h[:10]} failed: {str(e2)[:100]}", flush=True)
            return bi, hs, E, n
        with ThreadPoolExecutor(workers) as ex:
            for k, (bi, hs, E, n) in enumerate(ex.map(lambda a: one(*a), enumerate(batches))):
                for h, e in zip(hs, E):
                    if np.abs(e).sum() > 0: new[h] = e
                toks[0] += n
                if verbose and (k + 1) % 20 == 0: print(f"  {k + 1}/{len(batches)} batches, {toks[0]} tokens, {time.time() - t0:.0f}s", flush=True)
        if new:
            with _lock:
                idx, emb = _load_cache()
                add_h = [h for h in new if h not in idx]
                add = np.stack([new[h] for h in add_h]).astype(np.float16)
                hashes_all = list(idx.keys()) + add_h if idx else add_h
                emb = np.vstack([emb, add]) if emb is not None else add
                _save_cache(hashes_all, emb); idx = {h: i for i, h in enumerate(hashes_all)}
        if verbose: print(f"embed: done, {toks[0]} tokens, {time.time() - t0:.0f}s; cache {len(idx)}", flush=True)
    out = np.zeros((len(texts), 4096), np.float32)
    for i, (h, t) in enumerate(zip(hashes, texts)):
        if t.strip() and h in idx: out[i] = emb[idx[h]].astype(np.float32)
    n = np.linalg.norm(out, axis=1, keepdims=True); n[n == 0] = 1
    return out / n
