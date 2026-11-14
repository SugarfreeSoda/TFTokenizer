# TFTokenizer

TFTokenizer cuts a chromosome into variable-length tokens from transcription-factor motif hits. Each token
records, for every motif, the expected number of its bases occupied by that motif.

It builds on the interval tokenizer of *Motif-aware Tokenization of the Genome: Towards Interpretable Modeling of
Gene Regulation*. That tokenizer marks each motif hit as 0/1. TFTokenizer instead weights each hit by its rank among
the hits of its motif.

## Method

1. **Rank odds.** A hit of motif *m* gets a = u / (1 − u). u is the percentile of its JASPAR score among the hits of
   motif *m* on the training chromosomes: (hits scoring lower + half of those scoring the same) / N.
2. **Per-base occupancy.** Every base gets θ<sub>m</sub>(x) = A / (1 + A), where A is the sum of a over the hits of
   motif *m* (both strands) covering it. A base under a single hit has θ = u; overlapping hits add up and stay below 1.
3. **Bins.** θ of all motifs is summed per base and smoothed (Gaussian, σ = 3). Every peak contributes its two
   crossings at half its prominence as bin edges. Stretches longer than 50 bp are cut every 25 bp.
4. **Tokens.** For each bin and motif, the token value is Σ θ<sub>m</sub>(x) over the bases of the bin: the expected
   number of bases of the bin occupied by motif *m*.

There are no tunable weights.

### Changes from 0.1

Version 0.1 used a = exp(β (S − μ<sub>m</sub>)), merged overlapping hits into sites and spread the load
H = −ln(1 − θ) of a site over its length. Input tables, commands and output files are the same in 0.2, but:

- token values differ, and bin edges shift slightly;
- a model fitted with 0.1 cannot be loaded: run `fit` again;
- `sites.bin` holds segments of constant θ instead of merged sites, and `merge.bin` the hits covering each segment.

The old method stays available with `pip install tftokenizer==0.1.0`.

## Install

```bash
pip install tftokenizer
```

Requires Python ≥ 3.11. Prebuilt wheels are published for Linux (x86_64, aarch64), macOS (Apple silicon, Intel) and
Windows (x64), so no Rust toolchain is needed. On other platforms pip builds from source, which needs
[Rust](https://rustup.rs).

## Input

The motif hits of the UCSC JASPAR track, one table per chromosome. TFTokenizer downloads them itself:

```bash
tftokenizer download --chroms chr19 chr22      # writes hits/chr19.tsv.gz and hits/chr22.tsv.gz
```

The track file is large (196 GB for hg38 JASPAR2026), so only the chosen chromosomes are fetched: the records of a
chromosome lie together in the file, are located through its index and downloaded in one range request (2.0 GiB for
chr22). An interrupted download resumes where it stopped, and tables already there are skipped. Each table is what
`bigBedToBed -chrom=<chrom> -tsv` writes; `<chrom>.json` next to it records the source, row count and SHA-256.

Options: `--data` (folder, default `hits`), `--assembly` (default `hg38`), `--release` (default `JASPAR2026`) and
`--mirror` (default `https://hgdownload.soe.ucsc.edu`; the Europe mirror `https://hgdownload-euro.soe.ucsc.edu` can be
faster, depending on the network).

Tables of your own are passed with `--hits`: tab-separated, optionally gzipped, with a header, sorted by `chromStart`.
Columns used: `chrom chromStart chromEnd name score`.

## Usage

Command line. Without `--hits`, `fit` and `tokenize` first download the missing tables into `--data`; `tokenize`
downloads the assembly and release the model was fitted on:

```bash
tftokenizer fit      --chroms chr19 --model model.npz
tftokenizer tokenize --chroms chr19 chr22 --model model.npz --out tokens
```

With tables of your own (`{chrom}` is replaced by each chromosome name):

```bash
tftokenizer fit --chroms chr19 --model model.npz --hits 'my_hits/{chrom}.tsv.gz'
```

Python:

```python
from tftokenizer import TFTokenizer, download

chr19, chr22 = download(["chr19", "chr22"], "hits")
tok = TFTokenizer().fit([(chr19, "chr19")])
tok.save("model.npz")
tok.tokenize(chr22, "chr22", "tokens/chr22")
```

All tunable values live in `Params` (`tftokenizer/tokenizer.py`).
[`examples/quickstart.ipynb`](examples/quickstart.ipynb) reproduces the whole run: download, fit on chr19,
tokenize chr19 and chr22, and load the result.



