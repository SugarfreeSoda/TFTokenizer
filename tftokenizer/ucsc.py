"""Inputs from public servers. Motif hits of chosen chromosomes from the UCSC JASPAR bigBed track, without the whole file
(196 GB for hg38 JASPAR2026): the data blocks of one chromosome lie contiguously in the file; their byte span is found through the
chromosome B+ tree and the R-tree index (a few small range requests), fetched in one resumable range request and unpacked to
<chrom>.tsv.gz, the table `bigBedToBed -chrom=<chrom> -tsv` writes. Chromosome sequences from UCSC goldenPath
(<assembly>/chromosomes/<chrom>.fa.gz) and the JASPAR matrices (MEME format) of the track's release from the JASPAR website."""
import hashlib
import http.client
import json
import re
import struct
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from . import _core

MIRROR = "https://hgdownload.soe.ucsc.edu"   # Europe: https://hgdownload-euro.soe.ucsc.edu
LEVEL = 1         # gzip level of the written tables
RETRIES = 5       # attempts per chromosome; each resumes where the previous one stopped
TIMEOUT = 60      # seconds without data before an attempt fails
LOG_EVERY = 30    # seconds between progress messages


COLLECTION = "CORE_vertebrates_non-redundant"   # JASPAR collection of the UCSC JASPAR track


def url(assembly="hg38", release="JASPAR2026", mirror=MIRROR):
    return f"{mirror.rstrip('/')}/gbdb/{assembly}/jaspar/{release}.bb"


def genome_url(chrom, assembly="hg38", mirror=MIRROR):
    return f"{mirror.rstrip('/')}/goldenPath/{assembly}/chromosomes/{chrom}.fa.gz"


def matrices_url(release="JASPAR2026"):
    return f"https://jaspar.elixir.no/download/data/{release[-4:]}/CORE/{release}_{COLLECTION}_pfms_meme.txt"


class _Remote:
    """A file on an HTTP server that answers range requests."""

    def __init__(self, url):
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=TIMEOUT) as r:
            self.url, self.modified, self.size = url, r.headers.get("Last-Modified", ""), int(r.headers.get("Content-Length", -1))

    def open(self, start, end):
        """Response streaming bytes [start, end); refused if the file changed since __init__ (If-Range)."""
        headers = {"Range": f"bytes={start}-{end - 1}"} | ({"If-Range": self.modified} if self.modified else {})
        r = urllib.request.urlopen(urllib.request.Request(self.url, headers=headers), timeout=TIMEOUT)
        if r.status != 206:
            r.close()
            raise RuntimeError(f"{self.url} answered HTTP {r.status} to a range request; the file may have changed on the server")
        return r

    def read(self, offset, size):
        with self.open(offset, offset + size) as r:
            return r.read()


def locate(read, chrom):
    """(field names, chromosome id, first byte, end byte) of the records of `chrom` in a bigBed file read by read(offset, size).
    Raises KeyError if the file has no records on `chrom`."""
    magic, _, _, tree, _, index, fields, _, sql, *_ = struct.unpack("<IHHQQQHHQQIQ", read(0, 64))
    if magic != 0x8789F2EB:
        raise ValueError("not a little-endian bigBed file")
    names = re.findall(r"^\s*\S+\s+(\w+)\s*;", read(sql, 1 << 16).split(b"\0")[0].decode(), re.M)
    if not sql or len(names) != fields:
        raise ValueError(f"autoSql of the bigBed file does not name its {fields} fields")
    # chromosome B+ tree: in an inner node follow the last child whose first key is ≤ chrom
    _, fan, key, _, _, _ = struct.unpack("<IIIIQQ", read(tree, 32))
    want, node = chrom.encode().ljust(key, b"\0"), tree + 32
    while True:
        data = read(node, 4 + fan * (key + 8))
        leaf, _, n = struct.unpack_from("<BBH", data)
        items = [data[4 + i * (key + 8):4 + (i + 1) * (key + 8)] for i in range(n)]
        below = [item for item in items if (item[:key] == want if leaf else item[:key] <= want)]
        if not below:
            raise KeyError(chrom)
        if leaf:
            chrom_id = struct.unpack_from("<I", below[0], key)[0]
            break
        node = struct.unpack_from("<Q", below[-1], key)[0]
    # R-tree: the first and the last data block of the chromosome
    fan = struct.unpack_from("<I", read(index, 8), 4)[0]
    def edge(node, last):
        data = read(node, 4 + fan * 32)
        leaf, _, n = struct.unpack_from("<BBH", data)
        size = 32 if leaf else 24
        items = [struct.unpack_from("<IIIIQQ" if leaf else "<IIIIQ", data, 4 + i * size) for i in range(n)]
        items = [item for item in items if item[0] <= chrom_id <= item[2]]   # (first chrom, base, last chrom, base, offset[, size])
        if not items:
            raise KeyError(chrom)
        item = items[-1 if last else 0]
        return (item[4] + item[5] if last else item[4]) if leaf else edge(item[4], last)
    return names, chrom_id, edge(index + 48, False), edge(index + 48, True)


def _fetch(remote, start, end, part, chrom, log):
    """Appends bytes [start, end) of the remote file to `part`, resuming from its current size."""
    total = end - start
    for attempt in range(RETRIES):
        done = part.stat().st_size if part.exists() else 0
        if done >= total:
            break
        try:
            with remote.open(start + done, end) as r, open(part, "ab") as f:
                shown = time.time()
                while chunk := r.read(1 << 22):
                    f.write(chunk)
                    done += len(chunk)
                    if log and time.time() - shown > LOG_EVERY:
                        log(f"{chrom}: {done / 2**30:.2f} / {total / 2**30:.2f} GiB")
                        shown = time.time()
        except (OSError, http.client.HTTPException) as e:
            if attempt == RETRIES - 1:
                raise
            if log:
                log(f"{chrom}: {e!r}; retrying")
            time.sleep(2 ** attempt)
    if part.stat().st_size != total:
        raise RuntimeError(f"{chrom}: {part} has {part.stat().st_size} bytes, expected {total}")


def download(chroms, out="hits", assembly="hg38", release="JASPAR2026", mirror=MIRROR, log=None):
    """Writes the hits of each chromosome to out/<chrom>.tsv.gz, and to out/<chrom>.json the source, byte span, row count
    and SHA-256. Chromosomes whose table exists are skipped; an interrupted download resumes. Returns the table paths.
    log: optional function receiving progress messages."""
    out, paths, remote = Path(out), [], None
    out.mkdir(parents=True, exist_ok=True)
    for chrom in [chroms] if isinstance(chroms, str) else chroms:
        path = out / f"{chrom}.tsv.gz"
        paths.append(path)
        if path.exists():
            continue
        t, remote = time.time(), remote or _Remote(url(assembly, release, mirror))
        names, chrom_id, start, end = locate(remote.read, chrom)
        part, tmp = out / f".{chrom}.{start}-{end}.part", out / f".{chrom}.tsv.gz.tmp"
        for stale in set(out.glob(f".{chrom}.*.part")) - {part}:   # left by a download from an older version of the file
            stale.unlink()
        if log:
            log(f"{chrom}: downloading {(end - start) / 2**30:.2f} GiB from {remote.url}")
        _fetch(remote, start, end, part, chrom, log)
        rows = _core.unpack(part, chrom, chrom_id, "\t".join(names), tmp, LEVEL)
        with open(tmp, "rb") as f:
            sha256 = hashlib.file_digest(f, "sha256").hexdigest()
        source = dict(url=remote.url, last_modified=remote.modified, chrom=chrom, bytes=[start, end], rows=rows, sha256=sha256,
                      seconds=round(time.time() - t, 1), downloaded=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        (out / f"{chrom}.json").write_text(json.dumps(source, indent=1))
        tmp.replace(path)
        part.unlink()
        if log:
            log(f"{chrom}: {rows:,} hits → {path} ({source['seconds']:.0f} s)")
    return paths


def _record(path, remote, t, **extra):
    with open(path, "rb") as f:
        sha256 = hashlib.file_digest(f, "sha256").hexdigest()
    source = dict(url=remote.url, last_modified=remote.modified, bytes=path.stat().st_size, sha256=sha256, **extra,
                  seconds=round(time.time() - t, 1), downloaded=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    path.with_name(path.name.split(".")[0] + ".json").write_text(json.dumps(source, indent=1))


def download_genome(chroms, out="genome", assembly="hg38", mirror=MIRROR, log=None):
    """Writes the sequence of each chromosome to out/<chrom>.fa.gz (UCSC goldenPath) and its source and SHA-256 to
    out/<chrom>.json. Chromosomes whose file exists are skipped; an interrupted download resumes. Returns the FASTA paths."""
    out, paths = Path(out), []
    out.mkdir(parents=True, exist_ok=True)
    for chrom in [chroms] if isinstance(chroms, str) else chroms:
        path = out / f"{chrom}.fa.gz"
        paths.append(path)
        if path.exists():
            continue
        t, remote = time.time(), _Remote(genome_url(chrom, assembly, mirror))
        if remote.size < 0:
            raise RuntimeError(f"{remote.url} gives no Content-Length")
        part = out / f".{chrom}.fa.gz.part"
        if log:
            log(f"{chrom}: downloading {remote.size / 2**20:.0f} MiB from {remote.url}")
        _fetch(remote, 0, remote.size, part, chrom, log)
        part.replace(path)
        _record(path, remote, t, chrom=chrom)
    return paths


def download_matrices(out=".", release="JASPAR2026", log=None):
    """Writes the JASPAR matrices of the release (MEME format, collection COLLECTION) to out/ and its source and SHA-256 next to it.
    Skipped if the file exists. Returns its path."""
    path = Path(out) / f"{release}_{COLLECTION}_pfms_meme.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        t, remote = time.time(), _Remote(matrices_url(release))
        if log:
            log(f"matrices: downloading {remote.url}")
        tmp = path.with_name(f".{path.name}.tmp")
        with urllib.request.urlopen(remote.url, timeout=TIMEOUT) as r, open(tmp, "wb") as f:
            f.write(r.read())
        tmp.replace(path)
        _record(path, remote, t, release=release)
    return path
