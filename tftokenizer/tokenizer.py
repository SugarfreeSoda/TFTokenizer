"""Motif hits → per-motif occupancy track → bins → per-motif occupancy of every bin."""
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import zarr
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from . import _core


@dataclass
class Params:
    quantile: float = 0.90        # μ_m: this quantile of motif m's scores on the training chromosomes
    beta: float = 0.5             # binding strength a = exp(β (S − μ_m)), S = score / 100 · ln 10 (= −ln p)
    theta_max: float = 0.99       # site occupancy θ = Σa / (1 + Σa) is capped here before H = −ln(1 − θ)
    sigma: float = 3.0            # Gaussian smoothing of the summed occupancy density
    peak_width: float = 5.0       # find_peaks width; bin edges are the half-height crossings of each peak
    max_bin: int = 50             # stretches longer than max_bin bp are cut every `step` bp
    step: int = 25
    block: int = 100_000          # bins written per block


def bin_edges(density, params):
    """Half-height crossings of the peaks of the smoothed density; long stretches are cut every `step` bp."""
    _, peaks = find_peaks(gaussian_filter(density, params.sigma), width=params.peak_width)
    edges = np.unique(np.r_[0, len(density), peaks["right_ips"], peaks["left_ips"]].astype(np.int64))
    gap = np.diff(edges)
    k = np.where(gap > params.max_bin, -(-gap // params.step), 1)
    offset = np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k)
    return np.r_[np.repeat(edges[:-1], k) + params.step * offset, edges[-1]]


class TFTokenizer:
    def __init__(self, params=None):
        self.params = params or Params()
        self.motifs, self.mu = [], None

    def fit(self, train):
        """train: [(hits, chrom), ...]. μ_m = the `quantile` score of motif m over all training hits."""
        counts = {}
        for hits, chrom in train:
            for motif, row in zip(*_core.histogram(hits, chrom)):
                counts[motif] = counts.get(motif, 0) + row
        self.motifs = sorted(counts)
        C = np.array([counts[m] for m in self.motifs], np.float64)
        self.mu = (np.cumsum(C, 1) / np.maximum(C.sum(1, keepdims=True), 1) < self.params.quantile).sum(1)
        return self

    def table(self):
        """Relative binding strength a[motif, score]."""
        S = np.arange(_core.SCORES) / 100 * np.log(10)
        return np.exp(np.clip(self.params.beta * (S - self.mu[:, None] / 100 * np.log(10)), -50, 50)).astype(np.float32)

    def tokenize(self, hits, chrom, out):
        """Writes bins.npy (token edges), tokens.zarr (tokens × motifs, float32) and columns.json (motif IDs) to out.
        Tokens cover the chromosome from base 0 to the end of the last hit."""
        out, p = Path(out), self.params
        out.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out) as tmp:          # site list (GBs), removed on exit, also when interrupted
            n, max_len, length = _core.scan(hits, chrom, self.motifs, self.table(), p.theta_max, Path(tmp) / "sites.bin")
            sites = np.memmap(Path(tmp) / "sites.bin", np.uint32, "r", shape=(n, 4)) if n else np.zeros((0, 4), np.uint32)
            edges = bin_edges(_core.density(sites, length), p)
            tokens = zarr.open_array(str(out / "tokens.zarr"), mode="w", shape=(len(edges) - 1, len(self.motifs)),
                                     chunks=(p.block, len(self.motifs)), dtype=np.float32, zarr_format=2)
            for i in range(0, len(edges) - 1, p.block):
                tokens[i:i + p.block] = _core.features(sites, max_len, edges[i:i + p.block + 1], len(self.motifs)).astype(np.float32)
            del sites
        np.save(out / "bins.npy", edges)
        (out / "columns.json").write_text(json.dumps(self.motifs))
        return out

    def save(self, path):
        np.savez(path, motifs=np.array(self.motifs), mu=self.mu, params=json.dumps(asdict(self.params)))

    @classmethod
    def load(cls, path):
        f = np.load(path)
        tok = cls(Params(**json.loads(str(f["params"]))))
        tok.motifs, tok.mu = f["motifs"].tolist(), f["mu"]
        return tok
