"""Motif hits → per-base coverage evidence of every motif (bits) → bins → summed evidence of every motif in every bin."""
import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import zarr
from scipy.signal import find_peaks
from . import _core, evidence

SITE = np.dtype([("start", "<u4"), ("end", "<u4"), ("motif", "<u4"), ("weight", "<f4")])   # sites.bin: segments of constant evidence, sorted by end
MERGE = np.dtype([("hits", "<u4"), ("total", "<f4")])                                       # merge.bin, row i ↔ segment i


PRIMARY = tuple(f"chr{c}" for c in [*range(1, 23), "X", "Y"])


@dataclass
class Params:
    method: str = "score"         # how L of a hit is computed: "score" (from its UCSC score) or "exact" (from its sequence)
    background: tuple = PRIMARY   # chromosomes whose sequence the third-order Markov background is fitted on
    sigma: float = 3.0            # Gaussian smoothing of the summed evidence density
    peak_width: float = 5.0       # find_peaks width; bin edges are the crossings of each peak at half its prominence
    max_bin: int = 50             # stretches longer than max_bin bp are cut every `step` bp
    step: int = 25
    block: int = 100_000          # bins written per block
    # UCSC JASPAR track (gbdb/<assembly>/jaspar/<release>.bb) the command line downloads hits from when none are given;
    # kept in the model, so tokenize downloads the release the model was fitted on
    assembly: str = "hg38"
    release: str = "JASPAR2026"
    # optional outputs, written next to the tokens
    profile: bool = False         # run.json: seconds and peak resident memory of every step (fit: TFTokenizer.log)
    sites: bool = False           # sites.bin: all segments of constant evidence, i.e. the per-base track of every motif (read_sites, track)
    merge: bool = False           # merge.bin: number of hits covering every segment and their Σa (keeps sites.bin too)
    peaks: bool = False           # peaks.npz: the smoothed-density peaks behind the bin edges, and which edges are cuts


def _bins(density, params):
    """Bin edges, and the peaks they come from. Smooths `density` in place."""
    smooth = _core.smooth(density, params.sigma)
    peak, info = find_peaks(smooth, width=params.peak_width)
    edges = np.unique(np.r_[0, len(density), info["right_ips"], info["left_ips"]].astype(np.int64))
    gap = np.diff(edges)
    k = np.where(gap > params.max_bin, -(-gap // params.step), 1)
    offset = np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k)
    edges = np.r_[np.repeat(edges[:-1], k) + params.step * offset, edges[-1]]
    return edges, dict(peak=peak, height=smooth[peak], prominence=info["prominences"], left=info["left_ips"],
                       right=info["right_ips"], cut=np.r_[offset > 0, False])


def bin_edges(density, params):
    """Crossings of the smoothed-density peaks at half their prominence; long stretches are cut every `step` bp."""
    return _bins(np.array(density, np.float64), params)[0]       # copy: the caller's density stays unchanged


def read_sites(out):
    """Segments kept by tokenize (sites / merge on) as read-only SITE records — motif is the column in columns.json,
    weight the per-base coverage evidence in bits — and the MERGE records of the same segments (None without merge.bin)."""
    out = Path(out)
    read = lambda f, dtype: np.memmap(f, dtype, "r") if f.stat().st_size else np.zeros(0, dtype)
    return read(out / "sites.bin", SITE), read(out / "merge.bin", MERGE) if (out / "merge.bin").exists() else None


def track(sites, start, end):
    """Per-base evidence on [start, end) summed over `sites`, e.g. one motif: track(sites[sites["motif"] == k], 0, 10**6)."""
    s, n = sites[(sites["end"] > start) & (sites["start"] < end)], end - start
    lo, hi = np.maximum(s["start"].astype(np.int64), start) - start, np.minimum(s["end"].astype(np.int64), end) - start
    return np.cumsum(np.bincount(lo, s["weight"], n + 1) - np.bincount(hi, s["weight"], n + 1))[:n]


def _rss_reader():
    """Function returning the resident memory of this process in bytes (/proc on Linux, else psutil), or None."""
    if os.path.exists("/proc/self/statm"):
        page = os.sysconf("SC_PAGE_SIZE")
        return lambda: int(Path("/proc/self/statm").read_text().split()[1]) * page
    try:
        import psutil
    except ImportError:
        return None
    process = psutil.Process()
    return lambda: process.memory_info().rss


class _Monitor:
    """watch(step) records the seconds and peak resident memory (GB, sampled every 10 ms) of a step in .steps."""

    def __init__(self, on):
        self.on, self.steps, self.peak = on, {}, 0
        self.rss, self.lock, self.stop = _rss_reader() if on else None, threading.Lock(), threading.Event()

    def __enter__(self):
        if self.rss:
            threading.Thread(target=self._sample, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.stop.set()

    def _sample(self):
        while not self.stop.wait(0.01):
            with self.lock:
                self.peak = max(self.peak, self.rss())

    @contextmanager
    def __call__(self, step):
        if not self.on:
            yield
            return
        with self.lock:
            self.peak = self.rss() if self.rss else 0
        t = time.perf_counter()
        yield
        with self.lock:
            peak = round(max(self.peak, self.rss()) / 2**30, 3) if self.rss else None
        self.steps[step] = {"seconds": round(time.perf_counter() - t, 3), "peak_gb": peak}


class TFTokenizer:
    def __init__(self, params=None):
        self.params = params or Params()
        self.motifs, self.kmers, self.prob, self.width, self.lr, self.log = [], None, None, None, None, {}

    def fit(self, train, genome, matrices):
        """train: [(hits, chrom), ...], whose motifs become the token columns. genome: [(FASTA, chrom), ...], the sequences the
        third-order Markov background is fitted on (both strands). matrices: the JASPAR MEME file of the hits' release.
        Keeps the 4-mer counts, the smoothed motif probabilities and log2 L of every motif and UCSC score."""
        with _Monitor(self.params.profile) as watch:
            names = set()
            for hits, chrom in train:
                with watch(f"histogram {chrom}"):
                    names.update(_core.histogram(hits, chrom)[0])
            self.kmers = np.zeros(256, np.uint64)
            for fasta, chrom in genome:
                with watch(f"background {chrom}"):
                    self.kmers += _core.kmers(fasta, chrom)
            with watch("likelihood ratios"):
                meme = evidence.read_meme(matrices)
                missing = sorted(names - set(meme))
                if missing:
                    raise ValueError(f"{len(missing)} motifs of the hits are not in {matrices}, e.g. {missing[:3]}")
                self.motifs = sorted(names)
                start, trans = evidence.background(self.kmers)
                self.width = np.array([len(meme[m][2]) for m in self.motifs], np.uint32)
                self.prob = np.full((len(self.motifs), self.width.max(), 4), 0.25)
                for i, m in enumerate(self.motifs):
                    self.prob[i, :self.width[i]] = evidence.smoothed(meme[m][2], meme[m][1])
                self.lr = np.array([evidence.score_log2_lr(meme[m][2], meme[m][1], start, trans) for m in self.motifs])
        self.log = watch.steps
        return self

    def table(self):
        """a[motif, score] = L / (2W): L from the UCSC score, W the motif width (2W alignments of a covered base)."""
        return (2.0 ** self.lr / (2 * self.width[:, None])).astype(np.float32)

    def tokenize(self, hits, chrom, out, genome=None):
        """Writes bins.npy (token edges), tokens.zarr (tokens × motifs, float32), columns.json (motif IDs) and the optional
        outputs switched on in params to out. Tokens cover the chromosome from base 0 to the end of the last hit.
        genome: FASTA holding `chrom`, needed by method "exact"."""
        out, p = Path(out), self.params
        if p.method not in ("score", "exact"):
            raise ValueError(f"method must be 'score' or 'exact', not {p.method!r}")
        exact = None
        if p.method == "exact":
            if genome is None:
                raise ValueError("method 'exact' needs the FASTA of the chromosome (genome=...)")
            exact = (genome, np.log2(self.prob), self.width, np.log2(evidence.background(self.kmers)[1]))
        out.mkdir(parents=True, exist_ok=True)
        with _Monitor(p.profile) as watch, tempfile.TemporaryDirectory(dir=out) as tmp:   # segment list (GBs) unless kept
            site_path = (out if p.sites or p.merge else Path(tmp)) / "sites.bin"
            with watch("scan"):
                n, max_len, length = _core.scan(hits, chrom, self.motifs, self.table(), site_path, out / "merge.bin" if p.merge else None, exact)
            sites = np.memmap(site_path, np.uint32, "r", shape=(n, 4)) if n else np.zeros((0, 4), np.uint32)
            with watch("density"):
                density = _core.density(sites, length)
            with watch("bins"):
                edges, peaks = _bins(density, p)
            del density
            with watch("features"):
                tokens = zarr.open_array(str(out / "tokens.zarr"), mode="w", shape=(len(edges) - 1, len(self.motifs)),
                                         chunks=(p.block, len(self.motifs)), dtype=np.float32, zarr_format=2)
                for i in range(0, len(edges) - 1, p.block):
                    tokens[i:i + p.block] = _core.features(sites, max_len, edges[i:i + p.block + 1], len(self.motifs)).astype(np.float32)
            del sites
            with watch("write"):
                np.save(out / "bins.npy", edges)
                (out / "columns.json").write_text(json.dumps(self.motifs))
                if p.peaks:
                    np.savez(out / "peaks.npz", **peaks)
        if p.profile:
            run = dict(chrom=chrom, hits=str(hits), genome=None if genome is None else str(genome), threads=_core.threads(), sites=n,
                       longest_site=max_len, end=length,
                       bins=len(edges) - 1, steps=watch.steps, params=asdict(p))
            (out / "run.json").write_text(json.dumps(run, indent=1))
        return out

    def save(self, path):
        np.savez(path, motifs=np.array(self.motifs), kmers=self.kmers, prob=self.prob, width=self.width, lr=self.lr,
                 params=json.dumps(asdict(self.params)), log=json.dumps(self.log))

    @classmethod
    def load(cls, path):
        f = np.load(path)
        if "kmers" not in f:
            raise ValueError(f"{path} was fitted with tftokenizer 0.1 or 0.2 (no background model): fit it again")
        params = json.loads(str(f["params"]))
        tok = cls(Params(**params | {"background": tuple(params["background"])}))
        tok.motifs, tok.kmers, tok.prob, tok.width, tok.lr = f["motifs"].tolist(), f["kmers"], f["prob"], f["width"], f["lr"]
        tok.log = json.loads(str(f["log"]))
        return tok
