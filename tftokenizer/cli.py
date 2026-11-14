"""tftokenizer download | fit | tokenize. Without --hits, the hits of each chromosome are downloaded from the UCSC JASPAR
track to DATA/<chrom>.tsv.gz first (skipped if already there). --hits may contain {chrom}, replaced by each chromosome name."""
import argparse
import sys
from . import ucsc
from .tokenizer import Params, TFTokenizer

OUTPUTS = {"profile": "write run.json: seconds and peak memory of every step",
           "sites": "keep sites.bin: every segment of constant occupancy, i.e. the per-base track of every motif",
           "merge": "write merge.bin: hits covering every segment and their Σa (keeps sites.bin too)",
           "peaks": "write peaks.npz: the peaks behind the bin edges"}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tftokenizer", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="download the hits of chromosomes from the UCSC JASPAR track")
    fit = commands.add_parser("fit", help="count the scores of every motif on training chromosomes (gives the score percentiles)")
    tokenize = commands.add_parser("tokenize", help="write tokens: expected occupied bases of every motif")
    for p in (download, fit, tokenize):
        p.add_argument("--chroms", nargs="+", required=True)
        p.add_argument("--data", default="hits", help="downloaded hits go to DATA/<chrom>.tsv.gz (default: hits)")
        p.add_argument("--mirror", default=ucsc.MIRROR, help="UCSC server, e.g. https://hgdownload-euro.soe.ucsc.edu")
    for p in (download, fit):
        p.add_argument("--assembly", default=Params.assembly)
        p.add_argument("--release", default=Params.release, help="JASPAR track gbdb/ASSEMBLY/jaspar/RELEASE.bb")
    for p in (fit, tokenize):
        p.add_argument("--hits", help="local hit tables (.tsv or .tsv.gz), e.g. 'hits/{chrom}.tsv.gz'; without it they are downloaded")
        p.add_argument("--model", required=True, help="model file (.npz)")
    tokenize.add_argument("--out", required=True, help="tokens are written to OUT/<chrom>/")
    fit.add_argument("--profile", action="store_true", help="store seconds and peak memory of every step in the model")
    for name, help in OUTPUTS.items():
        tokenize.add_argument(f"--{name}", action="store_true", help=help)
    args = parser.parse_args(argv)
    log = lambda message: print(message, file=sys.stderr, flush=True)
    if args.command == "download":
        ucsc.download(args.chroms, args.data, args.assembly, args.release, args.mirror, log)
        return
    if args.command == "fit":
        tok = TFTokenizer(Params(profile=args.profile, assembly=args.assembly, release=args.release))
    else:
        tok = TFTokenizer.load(args.model)
        for name in OUTPUTS:
            setattr(tok.params, name, getattr(args, name))
    hits = ([args.hits.format(chrom=c) for c in args.chroms] if args.hits else
            ucsc.download(args.chroms, args.data, tok.params.assembly, tok.params.release, args.mirror, log))
    if args.command == "fit":
        tok.fit(list(zip(hits, args.chroms))).save(args.model)
    else:
        for h, c in zip(hits, args.chroms):
            tok.tokenize(h, c, f"{args.out}/{c}")


if __name__ == "__main__":
    main()
