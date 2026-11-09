"""
zlib streams of HDF5 deflate chunks, checked and inflated on the GPU.

HDF5's deflate filter stores each chunk as a zlib stream: a 2-byte header,
the raw Deflate body and the big-endian Adler-32 of the uncompressed bytes.
nvCOMP inflates the raw bodies; the header and the Adler-32 are checked
here, so a corrupt chunk raises instead of giving wrong counts.
"""

from __future__ import annotations

import numpy as np

_SOURCE = r"""
// out[c] = Adler-32 of src[c * chunk : (c + 1) * chunk]. With n = chunk,
// a = 1 + sum x_i and b = n + sum (n - i) x_i (mod 65521); the raw sums fit
// in 64 bits for chunks below about 380 MB.
extern "C" __global__ void adler32_chunks(const unsigned char* __restrict__ src, long long chunk,
                                          unsigned int* __restrict__ out) {
    const long long c = blockIdx.x;
    const unsigned char* s = src + c * chunk;
    unsigned long long a = 0, b = 0;
    for (long long i = threadIdx.x; i < chunk; i += blockDim.x) {
        unsigned long long v = s[i];
        a += v;
        b += (unsigned long long)(chunk - i) * v;
    }
    __shared__ unsigned long long sa[256], sb[256];
    sa[threadIdx.x] = a;
    sb[threadIdx.x] = b;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) {
            sa[threadIdx.x] += sa[threadIdx.x + o];
            sb[threadIdx.x] += sb[threadIdx.x + o];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        unsigned long long A = (1ULL + sa[0]) % 65521ULL;
        unsigned long long B = ((unsigned long long)chunk % 65521ULL + sb[0] % 65521ULL) % 65521ULL;
        out[c] = (unsigned int)((B << 16) | A);
    }
}
"""

_module = None


def _kernel(name: str):
    import cupy as cp

    global _module
    if _module is None:
        _module = cp.RawModule(code=_SOURCE)
    return _module.get_function(name)


def adler32_chunks(buf, chunk_bytes: int, n_chunks: int):
    """Adler-32 of each of the ``n_chunks`` consecutive ``chunk_bytes`` blocks of the device bytes ``buf``."""
    import cupy as cp

    out = cp.empty(n_chunks, cp.uint32)
    if n_chunks:
        _kernel("adler32_chunks")((n_chunks,), (256,), (buf, np.int64(chunk_bytes), out))
    return out


def check_headers(cmf: np.ndarray, flg: np.ndarray) -> np.ndarray:
    """
    Whether each zlib header (CMF, FLG bytes) is one HDF5's deflate filter can have written.

    Deflate with a window of at most 32 KiB, a valid check value, and no preset dictionary.
    """
    cmf = cmf.astype(np.int64)
    flg = flg.astype(np.int64)
    return ((cmf & 0x0F) == 8) & ((cmf >> 4) <= 7) & (((cmf << 8) | flg) % 31 == 0) & ((flg & 0x20) == 0)
