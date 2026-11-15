# TFTokenizer

TFTokenizer cuts a chromosome into variable-length tokens from transcription-factor motif hits. Each token
records, for every motif, how strongly its bases are supported as covered by that motif, in bits.

It builds on the interval tokenizer of *Motif-aware Tokenization of the Genome: Towards Interpretable Modeling of
Gene Regulation*. That tokenizer marks each motif hit as 0/1. TFTokenizer instead weighs each hit by how much more likely
its sequence is under the motif than under the genome background.

## Method

1. **Likelihood ratio.** A hit of motif *m* gets L = P(sequence | motif) / P(sequence | background). The motif model is
   the JASPAR matrix of the hit's release, with the pseudocount of the UCSC JASPAR track; the background is a third-order
   Markov chain fitted on the hg38 genome (chr1–22, X, Y, both strands). Two ways to compute L:
   - `score` (default): from the hit's UCSC score k alone, L = h<sub>1</sub>(k) / h<sub>0</sub>(k), the probability that
     a window drawn from the motif / the background passes the track's filter with score k. No sequence is read.
   - `exact`: from the hit's sequence and the 3 bases before it on its strand,
     L = Π p(b<sub>l</sub>, l) / Π T(b<sub>l</sub> | previous 3 bases).
2. **Per-base evidence.** A base covered by motif *m* can sit at any of the 2W positions of an instance (W the motif
   width, two strands), so its Bayes factor for being covered is BF = Σ L / (2W) over the hits of motif *m* covering it.
   Overlapping hits of one motif are alternative instances and add up. The base gets e<sub>m</sub>(x) = max(0, log2 BF).
3. **Bins.** e of all motifs is summed per base and smoothed (Gaussian, σ = 3). Every peak contributes its two
   crossings at half its prominence as bin edges. Stretches longer than 50 bp are cut every 25 bp.
4. **Tokens.** For each bin and motif, the token value is Σ e<sub>m</sub>(x) over the bases of the bin.

L needs no tuning: its mean over random background windows is 1 for every motif, whatever its length.

### Changes from 0.2

Version 0.2 weighted each hit by its rank among the hits of its motif, a = u / (1 − u), and every base got
θ = Σa / (1 + Σa). Input tables, commands and output files are the same in 0.3, but:

- token values are coverage evidence in bits, and bin edges shift;
- `fit` also needs the genome sequence (for the background) and the JASPAR matrices; both are downloaded when not given;
- a model fitted with 0.1 or 0.2 cannot be loaded: run `fit` again;
- `sites.bin` holds segments of constant evidence (segments of evidence 0 are left out), `merge.bin` the hits covering
  each segment and Σ L / (2W).

The older methods stay available with `pip install tftokenizer==0.2.0` (rank odds) or `==0.1.0`.

## Install

```bash
pip install tftokenizer
```

Requires Python ≥ 3.11. Prebuilt wheels are published for Linux (x86_64, aarch64), macOS (Apple silicon, Intel) and
Windows (x64), so no Rust toolchain is needed. On other platforms pip builds from source, which needs
[Rust](https://rustup.rs).

## Input

The motif hits of the UCSC JASPAR track, one table per chromosome; the hg38 chromosome sequences, for the background
(and, with `--method exact`, for the hits' sequences); the JASPAR matrices of the release. TFTokenizer downloads them
itself. The hits:

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
Columns used: `chrom chromStart chromEnd name score strand`.

The sequences come from UCSC goldenPath (`<assembly>/chromosomes/<chrom>.fa.gz`, about 900 MB for chr1–22, X, Y) into
`--genome-dir` (default `genome`); `download --genome` fetches those of `--chroms` together with their hits. Files of your
own are passed with `--genome`, e.g. `'genome/{chrom}.fa.gz'` (FASTA, optionally gzipped, holding a record named after each
chromosome). The matrices come from JASPAR
(`https://jaspar.elixir.no/download/data/2026/CORE/JASPAR2026_CORE_vertebrates_non-redundant_pfms_meme.txt`) into `--data`;
a file of your own is passed with `--matrices`. Every downloaded file gets a `.json` record of its source and SHA-256.

## Usage

Command line. Inputs not given are downloaded first; `tokenize` downloads the assembly and release the model was fitted on.
`fit` takes the motifs of the training chromosomes' hits, fits the background on the genome and computes the likelihood
ratio of every motif and score:

```bash
tftokenizer fit      --chroms chr19 --model model.npz
tftokenizer tokenize --chroms chr19 chr22 --model model.npz --out tokens
tftokenizer tokenize --chroms chr22 --model model.npz --out tokens_exact --method exact
```

With tables of your own (`{chrom}` is replaced by each chromosome name):

```bash
tftokenizer fit --chroms chr19 --model model.npz --hits 'my_hits/{chrom}.tsv.gz'
```

Python:

```python
from tftokenizer import Params, TFTokenizer, download, download_genome, download_matrices

chr19, chr22 = download(["chr19", "chr22"], "hits")
background = Params.background                       # chr1–22, X, Y
genome = download_genome(background, "genome")
tok = TFTokenizer().fit([(chr19, "chr19")], list(zip(genome, background)), download_matrices("hits"))
tok.save("model.npz")
tok.tokenize(chr22, "chr22", "tokens/chr22")
tok.params.method = "exact"
tok.tokenize(chr22, "chr22", "tokens_exact/chr22", genome="genome/chr22.fa.gz")
```

All settings live in `Params` (`tftokenizer/tokenizer.py`).
[`examples/quickstart.ipynb`](examples/quickstart.ipynb) reproduces the whole run: download, fit on chr19,
tokenize chr19 and chr22, and load the result.

The matrices are JASPAR data (CC BY 4.0); please cite JASPAR 2026 when you use them.



