"""FEC cache stage: farthest-point fingerprint rows, behaviour cache key, cached fitness evaluator.

FECEvaluator is used as ge.evolve's `evaluate`: for each training evaluation it computes the
individual's outputs on the fingerprint rows, hashes them into a key, and returns the cached
fitness on a hit; on a miss it evaluates on all training rows and stores the result. In
diagnostic mode each hit is also evaluated in full to count fake hits; the cached value is still
returned, so a diagnostic run evolves exactly like a timing run with the same seed.

Ported from the reference implementation (sampling_methods.sample_farthest_point, util_ge.create_fec_fitness,
SimpleFECCache), keeping only the farthest-point sampler and the behaviour key; the validity rule
is ge.predict's, the same as the baseline's.
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np

import ge
from measurement import operator_calls


# ---------------------------------------------------------------- farthest-point sampler

class _FeatureRowDistances:
    """distances_to(idx, out): Euclidean distance of every sample to sample idx for feature-major X,
    bit-identical to np.linalg.norm(X.T - X.T[idx], axis=1): the same pairwise summation numpy uses
    (< 8 values one by one; <= 128 eight running partials; > 128 split at n//2 rounded down to a
    multiple of 8), run over feature rows into preallocated buffers. Ported unchanged."""

    def __init__(self, X: np.ndarray) -> None:
        self.rows = np.ascontiguousarray(X, dtype=np.float64)
        n_total = self.rows.shape[1]
        self.term = np.empty(n_total, dtype=np.float64)
        self.partial = np.empty((8, n_total), dtype=np.float64)
        self.split_bufs: dict[int, np.ndarray] = {}

    def _squared_diff(self, f: int, idx: int, out: np.ndarray) -> np.ndarray:
        np.subtract(self.rows[f], self.rows[f, idx], out=out)
        return np.multiply(out, out, out=out)

    def _pairwise_sum(self, lo: int, hi: int, idx: int, out: np.ndarray, depth: int) -> None:
        n = hi - lo
        if n < 8:
            self._squared_diff(lo, idx, out)
            for f in range(lo + 1, hi):
                np.add(out, self._squared_diff(f, idx, self.term), out=out)
        elif n <= 128:
            r, t = self.partial, self.term
            for j in range(8):
                self._squared_diff(lo + j, idx, r[j])
            end = lo + n - n % 8
            for f in range(lo + 8, end, 8):
                for j in range(8):
                    np.add(r[j], self._squared_diff(f + j, idx, t), out=r[j])
            np.add(r[0], r[1], out=out)
            np.add(r[2], r[3], out=t)
            np.add(out, t, out=out)
            np.add(r[4], r[5], out=t)
            np.add(r[6], r[7], out=r[0])
            np.add(t, r[0], out=t)
            np.add(out, t, out=out)
            for f in range(end, hi):
                np.add(out, self._squared_diff(f, idx, t), out=out)
        else:
            n2 = n // 2
            n2 -= n2 % 8
            self._pairwise_sum(lo, lo + n2, idx, out, depth + 1)
            right = self.split_bufs.get(depth)
            if right is None:
                right = self.split_bufs[depth] = np.empty_like(out)
            self._pairwise_sum(lo + n2, hi, idx, right, depth + 1)
            np.add(out, right, out=out)

    def __call__(self, idx: int, out: np.ndarray) -> np.ndarray:
        self._pairwise_sum(0, self.rows.shape[0], idx, out, 0)
        return np.sqrt(out, out=out)


def farthest_point(X: np.ndarray, n_samples: int, seed: int, times: bool = False):
    """Indices of n_samples columns of X (features × rows): a random first row (seeded), then each
    time the row farthest from all picked rows. O(rows × samples × features), O(rows) memory.
    With times=True also returns the cumulative seconds after each pick, so the time of any prefix
    (a smaller fraction of the same run) is known exactly."""
    t0 = time.perf_counter()
    np.random.seed(seed)
    n_total = X.shape[1]
    if n_samples >= n_total:
        picked = np.arange(n_total)
        return (picked, np.full(n_total, time.perf_counter() - t0)) if times else picked
    distances_to = _FeatureRowDistances(X)
    picked = [np.random.randint(0, n_total)]
    min_d = distances_to(picked[0], np.empty(n_total))
    new_d = np.empty(n_total)
    cum = [time.perf_counter() - t0]
    for _ in range(n_samples - 1):
        picked.append(int(np.argmax(min_d)))
        np.minimum(min_d, distances_to(picked[-1], new_d), out=min_d)
        cum.append(time.perf_counter() - t0)
    return (np.array(picked), np.array(cum)) if times else np.array(picked)


def farthest_point_reference(X: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    """The plain form (np.linalg.norm per step), only to check farthest_point gives the same picks."""
    np.random.seed(seed)
    n_total = X.shape[1]
    if n_samples >= n_total:
        return np.arange(n_total)
    points = X.T
    picked = [np.random.randint(0, n_total)]
    min_d = np.linalg.norm(points - points[picked[0]], axis=1)
    for _ in range(n_samples - 1):
        picked.append(int(np.argmax(min_d)))
        min_d = np.minimum(min_d, np.linalg.norm(points - points[picked[-1]], axis=1))
    return np.array(picked)


def random_rows(X: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    """Uniform random rows (the reference implementation sample_random). The legacy generator's choice without
    replacement is a prefix of one permutation, so a run's smaller samples nest in its larger ones."""
    np.random.seed(seed)
    n_total = X.shape[1]
    if n_samples >= n_total:
        return np.arange(n_total)
    return np.random.choice(n_total, size=n_samples, replace=False)


def stratified_rows(y: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    """Equal share per class, remainder to the first classes, random within a class
    (the reference implementation sample_stratified)."""
    np.random.seed(seed)
    n_total = len(y)
    if n_samples >= n_total:
        return np.arange(n_total)
    classes = np.unique(y)
    per_class, remainder = divmod(n_samples, len(classes))
    picked = []
    for i, c in enumerate(classes):
        idx = np.where(y == c)[0]
        k = min(per_class + (1 if i < remainder else 0), len(idx))
        if k > 0:
            picked.extend(np.random.choice(idx, size=k, replace=False))
    return np.array(picked)


SAMPLERS = ("farthest_point", "random", "stratified")


def sample_rows(sampler: str, X: np.ndarray, y: np.ndarray, n_samples: int, seed: int,
                cache: dict | None = None) -> tuple[np.ndarray, float]:
    """(fingerprint rows, seconds the sampling took). For farthest_point, `cache` (from
    load_fps_cache) holds the picks of a larger sample of the same run; its first n_samples picks
    are this sample's picks (prefix property) and its cumulative time at n_samples is this
    sample's sampling time."""
    t0 = time.perf_counter()
    if sampler == "random":
        return random_rows(X, n_samples, seed), time.perf_counter() - t0
    if sampler == "stratified":
        return stratified_rows(y, n_samples, seed), time.perf_counter() - t0
    if sampler != "farthest_point":
        raise ValueError(f"unknown sampler {sampler!r}; choose one of {SAMPLERS}")
    if cache is not None and len(cache["indices"]) >= n_samples:
        return cache["indices"][:n_samples], float(cache["cum_times"][n_samples - 1])
    rows, cum = farthest_point(X, n_samples, seed, times=True)
    return rows, float(cum[-1])


def fps_cache_path(name: str, seed: int, setup: str = "default") -> Path:
    import datasets
    return datasets.ROOT / "results" / "fps_cache" / setup / name / f"seed{seed}.npz"


def load_fps_cache(name: str, seed: int, table_checksum: str, n_train: int, setup: str = "default") -> dict | None:
    """The cached picks of (dataset, seed) if they were made on this table and split, else None."""
    path = fps_cache_path(name, seed, setup)
    if not path.is_file():
        return None
    c = np.load(path)
    if str(c["table_checksum"]) != table_checksum or int(c["seed"]) != seed or int(c["n_train"]) != n_train:
        return None
    return {"indices": c["indices"], "cum_times": c["cum_times"]}


def prepare_fps(name: str, seed: int, fraction: float, overrides: dict | None = None) -> Path:
    """Compute and cache the farthest-point picks of run `seed` at `fraction` (the largest used)."""
    import datasets
    X, y, info = datasets.load_table(name)
    test_size = {**ge.PARAMS, **(overrides or {})}["test_size"]
    (Xt, _), _ = ge.split(X, y, seed, test_size, True, info)
    rows, cum = farthest_point(Xt, sample_size(fraction, Xt.shape[1]), seed, times=True)
    path = fps_cache_path(name, seed, ge.setup_id(overrides))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, indices=rows, cum_times=cum, table_checksum=datasets.dataset_record(name)["checksum"],
             seed=seed, n_train=Xt.shape[1])
    return path


def sample_size(fraction: float, n_train: int) -> int:
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"sample fraction must be in (0, 1], got {fraction}")
    return max(1, int(round(fraction * n_train)))


# ---------------------------------------------------------------- cache and evaluator

def cache_key(outputs: np.ndarray) -> str:
    """Behaviour key: the fingerprint outputs rounded to 6 decimals, hashed (the reference implementation behavior_hash)."""
    return hashlib.blake2b(np.round(outputs, 6).tobytes(), digest_size=16).hexdigest()


class FECEvaluator:
    """Callable(ind, points_train) -> (fitness,) with a behaviour cache; see the module docstring."""

    def __init__(self, sample_x: np.ndarray, diagnostic: bool = False, compile_fn=None, phases=None) -> None:
        """compile_fn(individual) -> x -> outputs (None if invalid), or None for an invalid individual;
        GE's by default (GP passes its own). Called once per evaluation: the same compiled individual
        runs on the fingerprint rows and, on a miss, on the training rows. phases: a measurement.PhaseClock or None; it only reads the clock, so a run evolves the same with or without it."""
        self.compile = compile_fn or ge.compile_ge
        self.full_fitness = lambda run, points: ge.error_of(run, points)
        self.sample_x = np.asarray(sample_x, dtype=np.float64)
        self.diagnostic = diagnostic
        self.fitness: dict[str, float] = {}
        self.phenotype: dict[str, str] = {}
        from measurement import Meter
        self.meter = Meter()
        self.total = {"hits": 0, "misses": 0, "same_phenotype_hits": 0, "fake_hits": 0}
        self.gen = dict.fromkeys(self.total, 0)
        self._reset_generation()
        self.last_outcome = ""                         # hit / miss / invalid, for the individuals file
        self.phases = phases

    def _reset_generation(self) -> None:
        self.gen_keys: set[str] = set()
        self.gen_size = {"hit": [], "miss": []}
        self.gen_diff: list[float] = []

    def __call__(self, ind, points) -> tuple[float]:
        m, ph = self.meter, self.phases
        m.start()                                      # evaluation time (measurement.py) starts
        if ph:
            ph.begin()
        run = self.compile(ind)
        self.last_outcome = "invalid"
        if run is None:                                # invalid individual (GE: not mapped)
            if ph:
                ph.lap("invalid_pre")
                ph.count("n_invalid_pre")
            m.stop()
            return (float("nan"),)
        if ph:
            ph.lap("compile")
        text = ge.text_of(ind)
        outputs = run(self.sample_x)
        if ph:
            ph.count("n_reached_fp")
        if outputs is None:
            if ph:
                ph.lap("invalid_fp")
                ph.count("n_invalid_fp")
            m.stop()
            m.add_sample(text, self.sample_x.shape[1])
            return (float("nan"),)
        if ph:
            ph.lap("fp_eval")
        key = cache_key(outputs)
        if ph:
            ph.lap("fp_key")
        self.gen_keys.add(key)
        size = operator_calls(text)                    # complexity of hit / miss individuals
        if ph:
            ph.lap("bookkeeping")
        found = key in self.fitness
        if ph:
            ph.lap("lookup")
        if found:
            cached = self.fitness[key]
            if ph:
                ph.lap("hit_return")
                ph.count("n_hits")
            m.stop()                                   # the cached value is ready
            m.add_sample(text, self.sample_x.shape[1])
            self._count("hits")
            self.last_outcome = "hit"
            self.gen_size["hit"].append(size)
            if self.phenotype[key] == text:
                self._count("same_phenotype_hits")
            if self.diagnostic:                        # check only; never changes the value used
                full = self.full_fitness(run, points)
                self.gen_diff.append(cached - full)
                if not np.isclose(cached, full, atol=0.0, rtol=0.0, equal_nan=True):
                    self._count("fake_hits")
            return (cached,)
        value = self.full_fitness(run, points)
        if ph:
            ph.lap("full_eval")
            ph.count("n_misses")
        if not np.isnan(value):
            self.fitness[key] = value
            self.phenotype[key] = text
        if ph:
            ph.lap("store")
        m.stop()                                       # full evaluation and store done
        m.add_sample(text, self.sample_x.shape[1])
        m.add_full(text, len(points[1]))
        self._count("misses")
        self.last_outcome = "miss"
        self.gen_size["miss"].append(size)
        return (value,)

    def _count(self, what: str) -> None:
        self.total[what] += 1
        self.gen[what] += 1

    def generation_stats(self) -> dict:
        """This generation's counters (then reset), for ge.evolve's per-generation rows."""
        mean = lambda v: float(np.mean(v)) if v else None         # None: nothing this generation
        out = {**{f"fec_{k}": v for k, v in self.gen.items()}, "fec_cache_size": len(self.fitness),
               "fec_unique_keys": len(self.gen_keys), "fec_hit_operators": mean(self.gen_size["hit"]),
               "fec_miss_operators": mean(self.gen_size["miss"])}
        if not self.diagnostic:
            del out["fec_fake_hits"]
        else:                                          # cached - full fitness on this generation's hits
            d = [x for x in self.gen_diff if not np.isnan(x)]
            out.update({"fec_fitness_diff": mean(d), "fec_abs_fitness_diff": mean([abs(x) for x in d]),
                        "fec_max_abs_fitness_diff": max((abs(x) for x in d), default=None)})
        if self.phases:
            out.update(self.phases.generation_stats())
        self.gen = dict.fromkeys(self.total, 0)
        self._reset_generation()
        return out

    def summary(self) -> dict:
        n = self.total["hits"] + self.total["misses"]
        out = {**self.total, "hit_rate": self.total["hits"] / n if n else float("nan"),
               "cache_size": len(self.fitness)}
        if not self.diagnostic:
            del out["fake_hits"]
        else:
            out["fake_hit_ratio"] = self.total["fake_hits"] / self.total["hits"] if self.total["hits"] else float("nan")
        return out


if __name__ == "__main__":             # python fec.py prepare <name> <seed> <fraction> [KEY=VALUE ...]
    if len(sys.argv) < 5 or sys.argv[1] != "prepare":
        sys.exit("usage: python fec.py prepare <dataset> <seed> <fraction> [KEY=VALUE ...]")
    print(prepare_fps(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), ge.parse_overrides(sys.argv[5:])))
