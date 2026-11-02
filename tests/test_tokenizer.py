import json
import numpy as np
import zarr
from scipy.ndimage import gaussian_filter1d
from tftokenizer import Params, TFTokenizer, _core, bin_edges, read_sites, track

LENGTH = 20_000


def make_hits(path, n_hits=3_000, seed=0):
    rng = np.random.default_rng(seed)
    motifs, widths = ["MA0001.1", "MA0002.1", "MA0003.1"], np.array([6, 10, 15])
    m, start, score = rng.integers(0, 3, n_hits), np.sort(rng.integers(0, LENGTH - 20, n_hits)), rng.integers(0, 1000, n_hits)
    rows = [f"chr1\t{s}\t{s + widths[k]}\t{motifs[k]}\t{sc}\t{'+-'[s % 2]}\tTF" for s, k, sc in zip(start, m, score)]
    path.write_text("chrom\tchromStart\tchromEnd\tname\tscore\tstrand\tTFName\n" + "\n".join(rows) + "\n")
    return motifs, start, start + widths[m], m, score


def reference(tok, start, end, motif, score):
    """Per-base occupancy track of each motif, built directly from the hits."""
    a, track = tok.table(), np.zeros((len(tok.motifs), LENGTH))
    for k in range(len(tok.motifs)):
        sel = np.flatnonzero(motif == k)
        sites, cur = [], None
        for i in sel[np.lexsort((end[sel], start[sel]))]:
            if cur and start[i] < cur[1]:
                cur = [cur[0], max(cur[1], end[i]), cur[2] + float(a[k, score[i]])]
            else:
                sites += [cur] if cur else []
                cur = [start[i], end[i], float(a[k, score[i]])]
        for s, e, total in sites + [cur]:
            track[k, s:e] += np.float32(-np.log1p(-min(total / (1 + total), tok.params.theta_max)) / (e - s))
    return track


def test_smooth_matches_scipy():
    rng = np.random.default_rng(1)
    for n in [1, 5, 25, 10_007]:
        x = rng.exponential(1, n)
        assert np.allclose(_core.smooth(x.copy(), 3.0), gaussian_filter1d(x, 3.0), rtol=1e-12, atol=0)   # equal up to rounding
    y = x.copy()
    bin_edges(y, Params())
    assert np.array_equal(y, x)


def test_tokens_match_reference(tmp_path):
    hits = tmp_path / "hits.tsv"
    motifs, start, end, motif, score = make_hits(hits)
    tok = TFTokenizer(Params(block=97)).fit([(hits, "chr1")])
    out = tok.tokenize(hits, "chr1", tmp_path / "chr1")
    edges, X = np.load(out / "bins.npy"), zarr.open_array(str(out / "tokens.zarr"), mode="r")[:]
    expected = np.add.reduceat(reference(tok, start, end, motif, score)[:, :edges[-1]], edges[:-1], axis=1).T
    assert sorted(f.name for f in out.iterdir()) == ["bins.npy", "columns.json", "tokens.zarr"]
    assert tok.motifs == motifs and json.loads((out / "columns.json").read_text()) == motifs
    assert edges[0] == 0 and edges[-1] == end.max() and np.allclose(X, expected, rtol=1e-5, atol=1e-7)
    tok.save(tmp_path / "model.npz")
    again = TFTokenizer.load(tmp_path / "model.npz").tokenize(hits, "chr1", tmp_path / "again")
    assert np.array_equal(zarr.open_array(str(again / "tokens.zarr"))[:], X)


def test_optional_outputs(tmp_path):
    hits = tmp_path / "hits.tsv"
    motifs, start, end, motif, score = make_hits(hits)
    plain = TFTokenizer(Params(block=97)).fit([(hits, "chr1")]).tokenize(hits, "chr1", tmp_path / "plain")
    tok = TFTokenizer(Params(block=97, profile=True, merge=True, peaks=True)).fit([(hits, "chr1")])
    out = tok.tokenize(hits, "chr1", tmp_path / "all")
    edges, read = np.load(out / "bins.npy"), lambda d: zarr.open_array(str(d / "tokens.zarr"), mode="r")[:]
    assert sorted(f.name for f in out.iterdir()) == ["bins.npy", "columns.json", "merge.bin", "peaks.npz", "run.json", "sites.bin", "tokens.zarr"]
    assert np.array_equal(edges, np.load(plain / "bins.npy")) and np.array_equal(read(out), read(plain))
    run = json.loads((out / "run.json").read_text())
    assert list(tok.log) == ["histogram chr1"] and list(run["steps"]) == ["scan", "density", "bins", "features", "write"]
    sites, merge = read_sites(out)
    ref = reference(tok, start, end, motif, score)[:, :edges[-1]]
    assert run["sites"] == len(sites) == len(merge) and merge["hits"].sum() == len(start)
    assert all(np.allclose(track(sites[sites["motif"] == k], 0, edges[-1]), ref[k], rtol=1e-5, atol=1e-7) for k in range(len(motifs)))
    total = merge["total"].astype(np.float64)
    H = -np.log1p(-np.minimum(total / (1 + total), tok.params.theta_max))
    assert np.allclose(sites["weight"] * (sites["end"] - sites["start"]), H, rtol=1e-5)
    peaks = np.load(out / "peaks.npz")
    cut = peaks["cut"]
    assert np.all(edges[cut] - edges[np.flatnonzero(cut) - 1] == tok.params.step)
    assert np.isin(edges[~cut][1:-1], np.r_[peaks["left"], peaks["right"]].astype(np.int64)).all()
