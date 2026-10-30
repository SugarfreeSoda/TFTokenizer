"""ctypes bindings to the compiled Rust core, shipped inside the package as tftokenizer/_native*."""
import ctypes as ct
import os
from pathlib import Path
import numpy as np

SCORES, MAX_MOTIFS = 1001, 8192
_lib = None


def lib():
    global _lib
    if _lib is None:
        _lib = ct.CDLL(str(next(Path(__file__).parent.glob("_native*"))))
        ptr = lambda dtype: np.ctypeslib.ndpointer(dtype=dtype, flags="C_CONTIGUOUS")
        err = [ct.c_char_p, ct.c_size_t]
        _lib.tt_histogram.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_size_t, ptr(np.uint64), ct.c_char_p, ct.c_size_t, ct.POINTER(ct.c_size_t)] + err
        _lib.tt_scan.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_char_p, ptr(np.float32), ct.c_double, ct.c_char_p,
                                 ct.POINTER(ct.c_uint64), ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_uint32)] + err
        _lib.tt_density.argtypes = [ptr(np.uint32), ct.c_size_t, ct.c_size_t, ptr(np.float64)]
        _lib.tt_density.restype = None
        _lib.tt_features.argtypes = [ptr(np.uint32), ct.c_size_t, ct.c_uint32, ptr(np.int64), ct.c_size_t, ct.c_size_t, ptr(np.float64)]
        _lib.tt_features.restype = None
    return _lib


def _check(code, error):
    if code:
        raise RuntimeError(error.value.decode())


def histogram(hits, chrom):
    """(motif names, hit counts per score 0–1000) for one chromosome."""
    counts, names, n = np.zeros((MAX_MOTIFS, SCORES), np.uint64), ct.create_string_buffer(1 << 20), ct.c_size_t()
    error = ct.create_string_buffer(1024)
    _check(lib().tt_histogram(os.fsencode(hits), chrom.encode(), MAX_MOTIFS, counts, names, len(names), ct.byref(n), error, len(error)), error)
    return names.value.decode().split("\n")[:n.value], counts[:n.value]


def scan(hits, chrom, motifs, table, theta_max, sites_path):
    """Writes occupancy sites to sites_path; returns (number of sites, longest site, end of the last hit)."""
    n, longest, end, error = ct.c_uint64(), ct.c_uint32(), ct.c_uint32(), ct.create_string_buffer(1024)
    _check(lib().tt_scan(os.fsencode(hits), chrom.encode(), "\n".join(motifs).encode(), np.ascontiguousarray(table, np.float32), theta_max,
                         os.fsencode(sites_path), ct.byref(n), ct.byref(longest), ct.byref(end), error, len(error)), error)
    return n.value, longest.value, end.value


def density(sites, length):
    """Summed per-base load of all sites."""
    out = np.empty(length)
    lib().tt_density(sites, len(sites), length, out)
    return out


def features(sites, max_len, edges, n_motifs):
    """Bin × motif matrix of Σ site load × fraction of the site inside the bin."""
    out = np.empty((len(edges) - 1, n_motifs))
    lib().tt_features(sites, len(sites), max_len, np.ascontiguousarray(edges, np.int64), len(edges), n_motifs, out)
    return out
