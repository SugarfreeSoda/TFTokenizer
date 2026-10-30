"""tftokenizer fit | tokenize. --hits may contain {chrom}, replaced by each chromosome name."""
import argparse
from .tokenizer import TFTokenizer


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tftokenizer", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="estimate the score scale μ_m of every motif on training chromosomes")
    tokenize = commands.add_parser("tokenize", help="write per-motif occupancy tokens")
    for p in (fit, tokenize):
        p.add_argument("--hits", required=True, help="JASPAR hit table (.tsv or .tsv.gz), e.g. hits/{chrom}.tsv.gz")
        p.add_argument("--chroms", nargs="+", required=True)
        p.add_argument("--model", required=True, help="model file (.npz)")
    tokenize.add_argument("--out", required=True, help="tokens are written to OUT/<chrom>/")
    args = parser.parse_args(argv)
    if args.command == "fit":
        TFTokenizer().fit([(args.hits.format(chrom=c), c) for c in args.chroms]).save(args.model)
    else:
        tok = TFTokenizer.load(args.model)
        for c in args.chroms:
            tok.tokenize(args.hits.format(chrom=c), c, f"{args.out}/{c}")


if __name__ == "__main__":
    main()
