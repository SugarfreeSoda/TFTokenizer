"""Motif and background models behind the coverage evidence: JASPAR matrices scored as on the UCSC JASPAR track, a third-order
Markov background of the genome, and the likelihood ratio L = P(window | motif) / P(window | background) of every UCSC score."""
import itertools
import numpy as np

ORDER = 3              # order of the Markov background (kept in sync with ORDER in src/lib.rs)
PSEUDO = 0.1           # pseudocount added to every 4-mer count of the background
REL, P_MAX = 0.8, 0.05  # UCSC JASPAR track: hits have relative score ≥ 0.8 and p < 0.05
SCORES = 1001


def read_meme(path):
    """{motif ID: (name, nsites, probabilities W × 4)} of a JASPAR MEME file, in file order."""
    motifs, cur = {}, None
    for line in open(path):
        f = line.split()
        if line.startswith("MOTIF"):
            cur = f[1]
            motifs[cur] = [f[2] if len(f) > 2 else "", None, []]
        elif line.startswith("letter-probability"):
            motifs[cur][1] = float(f[f.index("nsites=") + 1])
        elif cur and len(f) == 4 and f[0][0].isdigit():
            motifs[cur][2].append([float(v) for v in f])
    return {k: (name, n, np.array(rows)) for k, (name, n, rows) in motifs.items()}


def smoothed(prob, nsites):
    """Probabilities with the UCSC pseudocount: counts prob × nsites plus 0.25 √nsites per base."""
    c = 0.25 * np.sqrt(nsites)
    return (prob * nsites + c) / (nsites + 4 * c)


def background(counts):
    """(start distribution of the first ORDER bases [64], transitions T[state, base] [64 × 4]) from 4-mer counts [256]."""
    c = counts.reshape(4 ** ORDER, 4).astype(np.float64) + PSEUDO
    return c.sum(1) / c.sum(), c / c.sum(1, keepdims=True)


def _scores(w, column_prob):
    """Distribution of the integer score of PWM w (W × 4) when column l draws base b with column_prob[l, b]."""
    lo, d = int(w.min(1).sum()), np.array([1.0])
    for row, pr in zip(w, column_prob):
        k = row - row.min()
        new = np.zeros(len(d) + k.max())
        for b in range(4):
            new[k[b]:k[b] + len(d)] += pr[b] * d
        d = new
    return lo, d


def _markov_scores(w, start, trans):
    """Distribution of the integer score of PWM w under the Markov background (states: the last ORDER bases)."""
    off = w - w.min(1, keepdims=True)
    n, s = int(off.max(1).sum()) + 1, 4 ** (ORDER - 1)
    first = np.array(list(itertools.product(range(4), repeat=ORDER)))
    d = np.zeros((4 ** ORDER, n))
    np.add.at(d, (np.arange(4 ** ORDER), off[np.arange(ORDER), first].sum(1)), start)
    for l in range(ORDER, len(w)):
        d3, t3, new = d.reshape(4, s, n), trans.reshape(4, s, 4), np.zeros((s, 4, n))
        for b in range(4):
            k = off[l, b]
            new[:, b, k:] = np.einsum("or,orn->rn", t3[:, :, b], d3[:, :, :n - k])
        d = new.reshape(4 ** ORDER, n)
    return d.sum(0)


def score_log2_lr(prob, nsites, start, trans):
    """log2 L for every UCSC score 0–1000 of one motif: log2 h1(k) / h0(k), where h1 / h0 is the probability that a window drawn
    from the motif / the background passes the track's filter with score k. The integer PWM, its p-values (uniform background) and
    the filter follow the UCSC JASPAR track; scores the motif cannot reach take the value of the nearest reachable score."""
    if len(prob) < ORDER:
        raise ValueError(f"motif of width {len(prob)} is shorter than the background order {ORDER}")
    p = smoothed(prob, nsites)
    w = np.round(100 * np.log2(p / 0.25)).astype(np.int64)
    lo, uniform = _scores(w, np.full_like(p, 0.25))
    _, h1 = _scores(w, p)
    h0 = _markov_scores(w, start, trans)
    S = lo + np.arange(len(uniform))
    tail = np.cumsum(uniform[::-1])[::-1]
    keep = (S >= S[0] + REL * (S[-1] - S[0])) & (tail < P_MAX) & (uniform > 0)
    score = np.minimum(SCORES - 1, (-100 * np.log10(np.maximum(tail, 1e-300))).astype(np.int64))
    b0, b1 = np.bincount(score[keep], h0[keep], SCORES), np.bincount(score[keep], h1[keep], SCORES)
    ok = np.flatnonzero((b0 > 0) & (b1 > 0))
    near = ok[np.abs(np.arange(SCORES)[:, None] - ok[None, :]).argmin(1)]
    return np.log2(b1[near] / b0[near])


def window_log2_lr(seq, lp, trans):
    """log2 L of the windows seq[:, ORDER:] (rows of base codes 0–3, the first ORDER columns their context) for log2
    probabilities lp (W × 4); reference implementation of the Rust computation, used in tests."""
    lt, total = np.log2(trans), np.zeros(len(seq))
    state = seq[:, 0] * 16 + seq[:, 1] * 4 + seq[:, 2]
    for l in range(lp.shape[0]):
        b = seq[:, ORDER + l]
        total += lp[l, b] - lt[state, b]
        state = (state % 16) * 4 + b
    return total
