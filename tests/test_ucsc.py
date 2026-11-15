import gzip
import http.server
import itertools
import json
import struct
import threading
import zlib
import numpy as np
import pytest
from tftokenizer import download, download_genome, download_matrices, ucsc

SQL = b"""table JASPAR_TFBSs
"JASPAR TFBSs genome tracks"
(
string  chrom;  "Reference sequence chromosome or scaffold"
uint    chromStart;  "Start position of feature on chromosome"
uint    chromEnd;  "End position of feature on chromosome"
string  name;  "JASPAR profile ID"
uint    score;  "Score"
char[1] strand;  "+ or - for strand"
string  TFName;  "TF Name"
)
"""
HEADER = "chrom\tchromStart\tchromEnd\tname\tscore\tstrand\tTFName"


def two_levels(base, items, fan, inner):
    """Nodes padded to `fan` items: an inner root at `base`, whose items inner(chunk of leaf items, offset) point to the leaves."""
    chunks = [items[i:i + fan] for i in range(0, len(items), fan)]
    assert 1 < len(chunks) <= fan
    node = lambda leaf, entries, size: struct.pack("<BBH", leaf, 0, len(entries)) + b"".join(entries).ljust(fan * size, b"\0")
    size, inner_size = len(items[0]), len(inner(chunks[0], 0))
    offsets = [base + 4 + fan * inner_size + i * (4 + fan * size) for i in range(len(chunks))]
    return node(0, [inner(c, o) for c, o in zip(chunks, offsets)], inner_size) + b"".join(node(1, c, size) for c in chunks)


def bigbed(records, per_block=2, fan=2):
    """Minimal bigBed (no zoom levels) with two-level chromosome and R-trees; records: {chrom: [(start, end, rest bytes)]}."""
    chroms = sorted(records)
    key = max(map(len, chroms))
    blocks, index = [], []
    for cid, c in enumerate(chroms):
        for i in range(0, len(records[c]), per_block):
            part = records[c][i:i + per_block]
            blocks.append(zlib.compress(b"".join(struct.pack("<III", cid, s, e) + rest + b"\0" for s, e, rest in part)))
            index.append((cid, part[0][0], cid, max(e for _, e, _ in part), len(blocks[-1])))
    sql, tree = 64, 64 + len(SQL) + 1
    leaves = [c.encode().ljust(key, b"\0") + struct.pack("<II", i, 10**6) for i, c in enumerate(chroms)]
    tree_bytes = struct.pack("<IIIIQQ", 0x78CA8C91, fan, key, 8, len(chroms), 0) + two_levels(
        tree + 32, leaves, fan, lambda chunk, o: chunk[0][:key] + struct.pack("<Q", o))
    data = tree + len(tree_bytes)
    offsets = itertools.accumulate([data + 8] + [item[4] for item in index])
    rleaves = [struct.pack("<IIIIQQ", *item[:4], o, item[4]) for item, o in zip(index, offsets)]
    idx = data + 8 + sum(map(len, blocks))
    rtree = struct.pack("<IIQIIIIQII", 0x2468ACE0, fan * 2, len(index), 0, 0, 0, 0, idx, per_block, 0) + two_levels(
        idx + 48, rleaves, fan * 2, lambda chunk, o: chunk[0][:8] + chunk[-1][8:16] + struct.pack("<Q", o))
    head = struct.pack("<IHHQQQHHQQIQ", 0x8789F2EB, 4, 0, tree, data, idx, 7, 6, sql, 0, 1 << 16, 0)
    return head + SQL + b"\0" + tree_bytes + struct.pack("<Q", sum(map(len, records.values()))) + b"".join(blocks) + rtree


@pytest.fixture
def server():
    """Serves `server.data` with range requests on localhost; server.ranges records the requested ranges."""
    class Handler(http.server.BaseHTTPRequestHandler):
        def head(self, status, length):
            self.send_response(status)
            self.send_header("Content-Length", str(length))
            self.send_header("Last-Modified", "Sun, 18 Jan 2026 18:21:24 GMT")
            self.end_headers()

        def do_HEAD(self):
            self.head(200, len(httpd.data))

        def do_GET(self):
            if self.headers["Range"] is None:
                httpd.ranges.append(None)
                self.head(200, len(httpd.data))
                self.wfile.write(httpd.data)
                return
            lo, hi = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
            hi = min(hi, len(httpd.data) - 1)
            httpd.ranges.append((lo, hi + 1))
            self.head(206, hi + 1 - lo)
            self.wfile.write(httpd.data[lo:hi + 1])

        def log_message(self, *args):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.ranges, httpd.mirror = [], f"http://127.0.0.1:{httpd.server_port}"
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd
    httpd.shutdown()


def test_download(tmp_path, server):
    rng = np.random.default_rng(0)
    records = {}
    for c, n in [("chr1", 5), ("chr10", 3), ("chr2", 7), ("chrX", 1)]:
        start = np.sort(rng.integers(0, 10**5, n))
        records[c] = [(int(s), int(s) + 8, f"MA{rng.integers(1000):04d}.1\t{rng.integers(1000)}\t{'+-'[k % 2]}\tTF{k}".encode())
                      for k, s in enumerate(start)]
    server.data = bigbed(records)
    table = lambda c: HEADER + "\n" + "".join(f"{c}\t{s}\t{e}\t{rest.decode()}\n" for s, e, rest in records[c])
    read = lambda offset, size: server.data[offset:offset + size]
    names, cid, start, end = ucsc.locate(read, "chr2")
    assert "\t".join(names) == HEADER and cid == 2

    out = tmp_path / "hits"
    out.mkdir()
    (out / f".chr2.{start}-{end}.part").write_bytes(server.data[start:start + 5])    # an interrupted download of chr2
    (out / ".chr2.0-9.part").write_bytes(b"stale")
    paths = download(["chr2", "chrX", "chr10", "chr1"], out, mirror=server.mirror)
    for c, path in zip(["chr2", "chrX", "chr10", "chr1"], paths):
        assert gzip.decompress(path.read_bytes()).decode() == table(c)
        source = json.loads((out / f"{c}.json").read_text())
        assert source["rows"] == len(records[c]) and source["url"] == f"{server.mirror}/gbdb/hg38/jaspar/JASPAR2026.bb"
    assert (start + 5, end) in server.ranges                                        # resumed after the 5 bytes already there
    assert sorted(p.name for p in out.iterdir()) == sorted(f"{c}.{x}" for c in records for x in ["json", "tsv.gz"])

    server.ranges.clear()
    download("chr1", out, mirror=server.mirror)                                      # already there: no request
    assert server.ranges == []
    with pytest.raises(KeyError):
        download("chr3", out, mirror=server.mirror)


def test_download_genome_and_matrices(tmp_path, server, monkeypatch):
    server.data = gzip.compress(b">chr5\nACGTNacgt\nTTGA\n")
    out = tmp_path / "genome"
    out.mkdir()
    (out / ".chr5.fa.gz.part").write_bytes(server.data[:7])                       # an interrupted download
    [path] = download_genome(["chr5"], out, mirror=server.mirror)
    assert path.read_bytes() == server.data and (7, len(server.data)) in server.ranges
    source = json.loads((out / "chr5.json").read_text())
    assert source["url"] == f"{server.mirror}/goldenPath/hg38/chromosomes/chr5.fa.gz" and source["bytes"] == len(server.data)
    assert sorted(p.name for p in out.iterdir()) == ["chr5.fa.gz", "chr5.json"]
    server.ranges.clear()
    download_genome("chr5", out, mirror=server.mirror)                              # already there: no request
    assert server.ranges == []

    server.data = b"MEME version 4\n"
    monkeypatch.setattr(ucsc, "matrices_url", lambda release: f"{server.mirror}/{release}.meme")
    path = download_matrices(tmp_path, "JASPAR2026")
    assert path.name == "JASPAR2026_CORE_vertebrates_non-redundant_pfms_meme.txt" and path.read_bytes() == server.data
    assert json.loads((tmp_path / "JASPAR2026_CORE_vertebrates_non-redundant_pfms_meme.json").read_text())["release"] == "JASPAR2026"
    server.ranges.clear()
    download_matrices(tmp_path, "JASPAR2026")
    assert server.ranges == []
    assert ucsc.matrices_url.__name__ == "<lambda>" and ucsc.genome_url("chr1").endswith("/goldenPath/hg38/chromosomes/chr1.fa.gz")
