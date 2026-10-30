# TFTokenizer

TFTokenizer cuts a chromosome into variable-length tokens from transcription-factor motif hits. Each token
records the expected occupancy of every motif inside it.

It builds on the interval tokenizer of *Motif-aware Tokenization of the Genome: Towards Interpretable Modeling of
Gene Regulation*. That tokenizer marks each motif hit as 0/1. TFTokenizer instead weights each hit by its binding
strength and merges overlapping hits of a motif into occupancy sites.

## Method

1. **Binding strength.** A hit of motif *m* with JASPAR score *s* gets strength a = exp(β (S − μ<sub>m</sub>)), where
   S = s / 100 · ln 10 (= −ln p). μ<sub>m</sub> is the 90th percentile of motif *m*'s scores on the training chromosomes,
   and β = 0.5.
2. **Occupancy sites.** Overlapping hits of the same motif (both strands) are merged into one site, with occupancy
   θ = Σa / (1 + Σa) (capped at 0.99) and load H = −ln(1 − θ). H is spread evenly over the site, so each base carries H / site length.
3. **Bins.** The per-base load of all motifs is summed and smoothed (Gaussian, σ = 3). Every peak contributes its two
   half-height crossings as bin edges. Stretches longer than 50 bp are cut every 25 bp.
4. **Tokens.** For each bin and motif, the token value is Σ site load × fraction of the site inside the bin.

## Install

```bash
pip install tftokenizer
```

Requires Python ≥ 3.11. Prebuilt wheels are published for Linux (x86_64, aarch64), macOS (Apple silicon, Intel) and
Windows (x64), so no Rust toolchain is needed. On other platforms pip builds from source, which needs
[Rust](https://rustup.rs).

## Input

- **Motif hits**: the UCSC JASPAR track, one file per chromosome, sorted by `chromStart` (as written by `bigBedToBed`):

  ```bash
  bigBedToBed -chrom=chr22 -tsv https://hgdownload.soe.ucsc.edu/gbdb/hg38/jaspar/JASPAR2026.bb /dev/stdout | gzip > hits/chr22.tsv.gz
  ```

  Columns used: `chrom chromStart chromEnd name score`.

## Usage

Command line (`{chrom}` is replaced by each chromosome name):

```bash
tftokenizer fit      --hits 'hits/{chrom}.tsv.gz' --chroms chr19 --model model.npz
tftokenizer tokenize --hits 'hits/{chrom}.tsv.gz' --chroms chr19 chr22 --model model.npz --out tokens
```

Python:

```python
from tftokenizer import TFTokenizer

tok = TFTokenizer().fit([("hits/chr19.tsv.gz", "chr19")])
tok.save("model.npz")
tok.tokenize("hits/chr22.tsv.gz", "chr22", "tokens/chr22")
```

All tunable values live in `Params` (`tftokenizer/tokenizer.py`).
[`examples/quickstart.ipynb`](examples/quickstart.ipynb) reproduces the whole run: download, fit on chr19,
tokenize chr19 and chr22, and load the result.



