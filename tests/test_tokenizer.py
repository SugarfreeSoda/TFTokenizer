import gzip
import itertools
import json
import numpy as np
import pytest
import zarr
from scipy.ndimage import gaussian_filter1d
from tftokenizer import Params, TFTokenizer, _core, bin_edges, evidence, read_sites, track

LENGTH = 20_000
MOTIFS, WIDTHS = ["MA0001.1", "MA0002.1", "MA0003.1"], np.array([6, 10, 15])
RC = np.array([3, 2, 1, 0, 4])


@pytest.fixture
def inputs(tmp_path):
    """Genome (chr1, plus a decoy record), JASPAR-style matrices and hits on chr1; returns paths and the hits."""
    rng = np.random.default_rng(0)
    seq = rng.choice(4, LENGTH, p=[0.3, 0.2, 0.2, 0.3])
    seq[5000:5020] = 4                                              # an N run: hits over it fall back to their score
    text = "".join("ACGTN"[b] for b in seq)
    text = text[:100] + text[100:300].lower() + text[300:]           # soft-masked bases count as well
    lines = lambda s: "\n".join(s[i:i + 60] for i in range(0, len(s), 60))
    fasta = tmp_path / "genome.fa.gz"
    with gzip.open(fasta, "wt") as f:
        f.write(f">chr0 decoy\n{lines('A' * 500)}\n>chr1\n{lines(text)}\n")
    meme = tmp_path / "matrices.meme"
    blocks = []
    for k, (m, w) in enumerate(zip(MOTIFS, WIDTHS)):
        prob = rng.dirichlet(np.full(4, 0.3), w)
        rows = "\n".join(" ".join(f"{v:.6f}" for v in r) for r in prob)
        blocks.append(f"MOTIF {m} TF{k}\nletter-probability matrix: alength= 4 w= {w} nsites= {[20, 300, 4000][k]} E= 0\n{rows}\n")
    meme.write_text("MEME version 4\n\nALPHABET= ACGT\n\n" + "\n".join(blocks))
    n_hits = 3000
    m, start, score = rng.integers(0, 3, n_hits), np.sort(rng.integers(0, LENGTH - 20, n_hits)), rng.integers(0, 1001, n_hits)
    start[:3] = 0                                                   # hits whose context leaves the sequence
    strand = rng.choice(["+", "-"], n_hits)
    hits = tmp_path / "hits.tsv"
    rows = [f"chr1\t{s}\t{s + WIDTHS[k]}\t{MOTIFS[k]}\t{sc}\t{st}\tTF" for s, k, sc, st in zip(start, m, score, strand)]
    hits.write_text("chrom\tchromStart\tchromEnd\tname\tscore\tstrand\tTFName\n" + "\n".join(rows) + "\n")
    return dict(fasta=fasta, meme=meme, hits=hits, seq=seq, start=start, end=start + WIDTHS[m], motif=m, score=score, minus=strand == "-")


def fitted(inputs, **params):
    return TFTokenizer(Params(block=97, **params)).fit([(inputs["hits"], "chr1")], [(inputs["fasta"], "chr1")], inputs["meme"])


def reference(tok, d, exact):
    """Per-base evidence max(0, log2 Σ L / 2W) of each motif, built directly from the hits."""
    a, total = tok.table().astype(np.float64), np.zeros((len(tok.motifs), LENGTH))
    trans = evidence.background(tok.kmers)[1]
    for s, e, k, sc, minus in zip(d["start"], d["end"], d["motif"], d["score"], d["minus"]):
        v = a[k, sc]
        if exact:
            w = e - s
            ext = RC[d["seq"][s:e + 3]][::-1] if minus else d["seq"][max(s - 3, 0):e]
            if len(ext) == w + 3 and (ext < 4).all():
                v = 2.0 ** evidence.window_log2_lr(ext[None], np.log2(tok.prob[k, :w]), trans)[0] / (2 * w)
        total[k, s:e] += v
    return np.where(total > 1, np.log2(np.maximum(total, 1)), 0).astype(np.float32)


def test_smooth_matches_scipy():
    rng = np.random.default_rng(1)
    for n in [1, 5, 25, 10_007]:
        x = rng.exponential(1, n)
        assert np.allclose(_core.smooth(x.copy(), 3.0), gaussian_filter1d(x, 3.0), rtol=1e-12, atol=0)   # equal up to rounding
    y = x.copy()
    bin_edges(y, Params())
    assert np.array_equal(y, x)


def test_kmers_match_brute_force(inputs):
    seq = inputs["seq"]
    want = np.zeros(256, np.uint64)
    for i in range(len(seq) - 3):
        w = seq[i:i + 4]
        if (w < 4).all():
            want[int(w @ [64, 16, 4, 1])] += 1
            want[int(RC[w][::-1] @ [64, 16, 4, 1])] += 1
    assert np.array_equal(_core.kmers(inputs["fasta"], "chr1"), want)
    with pytest.raises(RuntimeError, match="no sequence chr9"):
        _core.kmers(inputs["fasta"], "chr9")


def test_score_lr_matches_enumeration():
    """log2 h1(k)/h0(k) against all 4^W windows, each scored, filtered and assigned its UCSC score as on the track."""
    rng = np.random.default_rng(2)
    start = rng.dirichlet(np.ones(64)); trans = rng.dirichlet(np.ones(4), 64)
    for w, n in ((4, 30), (6, 500)):
        prob = rng.dirichlet(np.full(4, 0.4), w)
        p = evidence.smoothed(prob, n)
        pwm = np.round(100 * np.log2(p / 0.25)).astype(np.int64)
        x = np.array(list(itertools.product(range(4), repeat=w)))
        S = pwm[np.arange(w), x].sum(1)
        p_bg = start[x[:, 0] * 16 + x[:, 1] * 4 + x[:, 2]] * np.prod([trans[x[:, l - 3] * 16 + x[:, l - 2] * 4 + x[:, l - 1], x[:, l]] for l in range(3, w)], 0)
        p_m = np.prod(p[np.arange(w), x], 1)
        tail = np.array([np.mean(S >= s) for s in S])                       # uniform background: every window equally likely
        keep = (S >= pwm.min(1).sum() + 0.8 * (pwm.max(1).sum() - pwm.min(1).sum())) & (tail < 0.05)
        score = np.minimum(1000, (-100 * np.log10(tail)).astype(np.int64))
        h0, h1 = np.bincount(score[keep], p_bg[keep], 1001), np.bincount(score[keep], p_m[keep], 1001)
        k = np.flatnonzero(h0 > 0)
        assert np.allclose(evidence.score_log2_lr(prob, n, start, trans)[k], np.log2(h1[k] / h0[k]), atol=1e-9)


@pytest.mark.parametrize("method", ["score", "exact"])
def test_tokens_match_reference(inputs, tmp_path, method):
    tok = fitted(inputs, method=method)
    out = tok.tokenize(inputs["hits"], "chr1", tmp_path / method, inputs["fasta"] if method == "exact" else None)
    edges, X = np.load(out / "bins.npy"), zarr.open_array(str(out / "tokens.zarr"), mode="r")[:]
    expected = np.add.reduceat(reference(tok, inputs, method == "exact")[:, :edges[-1]], edges[:-1], axis=1).T
    assert sorted(f.name for f in out.iterdir()) == ["bins.npy", "columns.json", "tokens.zarr"]
    assert tok.motifs == MOTIFS and json.loads((out / "columns.json").read_text()) == MOTIFS
    assert edges[0] == 0 and edges[-1] == inputs["end"].max() and np.allclose(X, expected, rtol=1e-4, atol=1e-5)
    tok.save(tmp_path / "model.npz")
    again = TFTokenizer.load(tmp_path / "model.npz").tokenize(inputs["hits"], "chr1", tmp_path / "again", inputs["fasta"])
    assert np.array_equal(zarr.open_array(str(again / "tokens.zarr"))[:], X)


def test_exact_needs_genome_and_new_model(inputs, tmp_path):
    tok = fitted(inputs, method="exact")
    with pytest.raises(ValueError, match="needs the FASTA"):
        tok.tokenize(inputs["hits"], "chr1", tmp_path / "x")
    np.savez(tmp_path / "old.npz", motifs=np.array(MOTIFS), counts=np.zeros((3, 1001)), params="{}", log="{}")
    with pytest.raises(ValueError, match="fit it again"):
        TFTokenizer.load(tmp_path / "old.npz")


def test_optional_outputs(inputs, tmp_path):
    plain = fitted(inputs).tokenize(inputs["hits"], "chr1", tmp_path / "plain")
    tok = fitted(inputs, profile=True, merge=True, peaks=True)
    out = tok.tokenize(inputs["hits"], "chr1", tmp_path / "all")
    edges, read = np.load(out / "bins.npy"), lambda d: zarr.open_array(str(d / "tokens.zarr"), mode="r")[:]
    assert sorted(f.name for f in out.iterdir()) == ["bins.npy", "columns.json", "merge.bin", "peaks.npz", "run.json", "sites.bin", "tokens.zarr"]
    assert np.array_equal(edges, np.load(plain / "bins.npy")) and np.array_equal(read(out), read(plain))
    run = json.loads((out / "run.json").read_text())
    assert list(tok.log) == ["histogram chr1", "background chr1", "likelihood ratios"]
    assert list(run["steps"]) == ["scan", "density", "bins", "features", "write"]
    sites, merge = read_sites(out)
    ref = reference(tok, inputs, False)[:, :edges[-1]]
    assert run["sites"] == len(sites) == len(merge) and (merge["hits"] >= 1).all()
    assert all(np.allclose(track(sites[sites["motif"] == k], 0, edges[-1]), ref[k], rtol=1e-4, atol=1e-5) for k in range(len(MOTIFS)))
    total = merge["total"].astype(np.float64)
    assert (total > 1).all() and np.allclose(sites["weight"], np.log2(total), rtol=1e-5, atol=1e-6)
    peaks = np.load(out / "peaks.npz")
    cut = peaks["cut"]
    assert np.all(edges[cut] - edges[np.flatnonzero(cut) - 1] == tok.params.step)
    assert np.isin(edges[~cut][1:-1], np.r_[peaks["left"], peaks["right"]].astype(np.int64)).all()
