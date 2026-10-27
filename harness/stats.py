"""Wilson intervals for rates, bootstrap for differences."""
import math, random
import numpy as np

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"), float("nan"))
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (p, c - h, c + h)

def fmt_rate(k, n):
    p, lo, hi = wilson(k, n)
    return f"{100*p:.0f}% [{100*lo:.0f}, {100*hi:.0f}] (n={n})"

def bootstrap_diff(a, b, iters=5000, seed=0):
    """a, b: arrays of 0/1 (or floats). Returns (mean diff a-b, lo, hi) 95% percentile CI."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) == 0 or len(b) == 0: return (float("nan"),)*3
    diffs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(iters)]
    return (a.mean() - b.mean(), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))

def cluster_bootstrap_diff(items_a, items_b, iters=5000, seed=0):
    """items: dict cluster_id -> list of 0/1. Resamples clusters (e.g. scenarios) then observations."""
    rng = np.random.default_rng(seed)
    ka, kb = list(items_a), list(items_b)
    def one(items, keys):
        ks = rng.choice(keys, len(keys))
        vals = np.concatenate([rng.choice(items[k], len(items[k])) for k in ks])
        return vals.mean()
    obs = np.concatenate(list(items_a.values())).mean() - np.concatenate(list(items_b.values())).mean()
    diffs = [one(items_a, ka) - one(items_b, kb) for _ in range(iters)]
    return (obs, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))

def cluster_bootstrap_mean(items, iters=5000, seed=0):
    """items: dict cluster_id -> list of values (e.g. scenario -> the P(preferred) of its counterbalance cells).
    Resamples CLUSTERS only (not observations within a cluster: for log-prob cells the within-scenario values are a fixed
    design, not a sample). Returns (observed mean over all values, lo, hi) 95% percentile CI."""
    rng = np.random.default_rng(seed)
    keys = list(items)
    if not keys: return (float("nan"),)*3
    obs = float(np.concatenate([np.asarray(items[k], float) for k in keys]).mean())
    means = []
    for _ in range(iters):
        ks = rng.choice(keys, len(keys))
        means.append(np.concatenate([np.asarray(items[k], float) for k in ks]).mean())
    return (obs, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))

def paired_cluster_bootstrap_diff(items_a, items_b, iters=5000, seed=0):
    """Paired version of cluster_bootstrap_diff: the SAME resampled clusters are used for a and b (e.g. the same
    scenarios under two framings), and only clusters are resampled. Clusters missing from either side are dropped.
    Returns (observed mean(a) - mean(b), lo, hi) 95% percentile CI."""
    rng = np.random.default_rng(seed)
    keys = [k for k in items_a if k in items_b]
    if not keys: return (float("nan"),)*3
    def m(items, ks): return np.concatenate([np.asarray(items[k], float) for k in ks]).mean()
    obs = float(m(items_a, keys) - m(items_b, keys))
    diffs = []
    for _ in range(iters):
        ks = rng.choice(keys, len(keys))
        diffs.append(m(items_a, ks) - m(items_b, ks))
    return (obs, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
