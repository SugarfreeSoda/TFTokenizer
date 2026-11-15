"""tftokenizer download | fit | tokenize. Inputs not given are downloaded first (files already there are skipped): the hits of
each chromosome from the UCSC JASPAR track to DATA/<chrom>.tsv.gz, chromosome sequences from UCSC to GENOME_DIR/<chrom>.fa.gz,
the JASPAR matrices to DATA/. --hits and --genome may contain {chrom}, replaced by each chromosome name."""
import argparse
import sys
from . import ucsc
from .tokenizer import Params, TFTokenizer

OUTPUTS = {"profile": "write run.json: seconds and peak memory of every step",
           "sites": "keep sites.bin: every segment of constant evidence, i.e. the per-base track of every motif",
           "merge": "write merge.bin: hits covering every segment and their Σa (keeps sites.bin too)",
           "peaks": "write peaks.npz: the peaks behind the bin edges"}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tftokenizer", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="download the hits of chromosomes from the UCSC JASPAR track")
    fit = commands.add_parser("fit", help="fit the background on the genome and the likelihood ratios of the training chromosomes' motifs")
    tokenize = commands.add_parser("tokenize", help="write tokens: coverage evidence of every motif in bits")
    for p in (download, fit, tokenize):
        p.add_argument("--chroms", nargs="+", required=True)
        p.add_argument("--data", default="hits", help="downloaded hits go to DATA/<chrom>.tsv.gz (default: hits)")
        p.add_argument("--mirror", default=ucsc.MIRROR, help="UCSC server, e.g. https://hgdownload-euro.soe.ucsc.edu")
        p.add_argument("--genome-dir", default="genome", help="downloaded sequences go to GENOME_DIR/<chrom>.fa.gz (default: genome)")
    for p in (download, fit):
        p.add_argument("--assembly", default=Params.assembly)
        p.add_argument("--release", default=Params.release, help="JASPAR track gbdb/ASSEMBLY/jaspar/RELEASE.bb")
    download.add_argument("--genome", action="store_true", help="also download the sequences of the chromosomes")
    for p in (fit, tokenize):
        p.add_argument("--hits", help="local hit tables (.tsv or .tsv.gz), e.g. 'hits/{chrom}.tsv.gz'; without it they are downloaded")
        p.add_argument("--genome", help="local FASTA (.fa or .fa.gz) holding each chromosome, e.g. 'genome/{chrom}.fa.gz'; "
                                        "without it the sequences are downloaded (fit: the background chromosomes; tokenize: only with --method exact)")
        p.add_argument("--model", required=True, help="model file (.npz)")
    tokenize.add_argument("--out", required=True, help="tokens are written to OUT/<chrom>/")
    fit.add_argument("--profile", action="store_true", help="store seconds and peak memory of every step in the model")
    fit.add_argument("--matrices", help="local JASPAR MEME file of the release; without it, it is downloaded to DATA/")
    tokenize.add_argument("--method", choices=["score", "exact"], default=Params.method,
                          help="L of a hit from its UCSC score (default) or from its sequence and the background")
    for name, help in OUTPUTS.items():
        tokenize.add_argument(f"--{name}", action="store_true", help=help)
    args = parser.parse_args(argv)
    log = lambda message: print(message, file=sys.stderr, flush=True)
    if args.command == "download":
        ucsc.download(args.chroms, args.data, args.assembly, args.release, args.mirror, log)
        if args.genome:
            ucsc.download_genome(args.chroms, args.genome_dir, args.assembly, args.mirror, log)
        return
    if args.command == "fit":
        tok = TFTokenizer(Params(profile=args.profile, assembly=args.assembly, release=args.release))
    else:
        tok = TFTokenizer.load(args.model)
        tok.params.method = args.method
        for name in OUTPUTS:
            setattr(tok.params, name, getattr(args, name))
    p = tok.params
    hits = ([args.hits.format(chrom=c) for c in args.chroms] if args.hits else
            ucsc.download(args.chroms, args.data, p.assembly, p.release, args.mirror, log))
    genome = lambda chroms: ([args.genome.format(chrom=c) for c in chroms] if args.genome else
                             ucsc.download_genome(chroms, args.genome_dir, p.assembly, args.mirror, log))
    if args.command == "fit":
        matrices = args.matrices or ucsc.download_matrices(args.data, p.release, log)
        tok.fit(list(zip(hits, args.chroms)), list(zip(genome(p.background), p.background)), matrices).save(args.model)
    else:
        fasta = genome(args.chroms) if p.method == "exact" else [None] * len(args.chroms)
        for h, c, g in zip(hits, args.chroms, fasta):
            tok.tokenize(h, c, f"{args.out}/{c}", g)


if __name__ == "__main__":
    main()
