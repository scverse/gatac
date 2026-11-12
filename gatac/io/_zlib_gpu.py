"""
zlib streams of HDF5 deflate chunks, checked and inflated on the GPU.

HDF5's deflate filter stores each chunk as a zlib stream: a 2-byte header,
the raw Deflate body and the big-endian Adler-32 of the uncompressed bytes.
nvCOMP inflates the raw bodies; the header and the Adler-32 are checked
here, so a corrupt chunk raises instead of giving wrong counts. Written
chunks are framed the same way around nvCOMP's raw Deflate bodies.
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

// dst[off[c]:] = 0x78 0x01 | the size[c] Deflate bytes at ptrs[c] | adler[c] big-endian.
// One block per chunk.
extern "C" __global__ void zlib_frame(const unsigned long long* __restrict__ ptrs,
                                      unsigned char* __restrict__ dst, const long long* __restrict__ off,
                                      const long long* __restrict__ size,
                                      const unsigned int* __restrict__ adler) {
    const int c = blockIdx.x;
    const unsigned char* s = (const unsigned char*)ptrs[c];
    unsigned char* d = dst + off[c];
    const long long n = size[c];
    if (threadIdx.x == 0) {
        const unsigned int a = adler[c];
        d[0] = 0x78;
        d[1] = 0x01;
        d[2 + n] = a >> 24;
        d[3 + n] = (a >> 16) & 255;
        d[4 + n] = (a >> 8) & 255;
        d[5 + n] = a & 255;
    }
    for (long long i = threadIdx.x; i < n; i += blockDim.x) d[2 + i] = s[i];
}
"""

# zlib header (CMF 0x78: Deflate, 32 KiB window; FLG 0x01: fastest level, no
# dictionary, check bits) and Adler-32 trailer around each Deflate body
ZLIB_FRAMING = 6

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


def zlib_frame(bodies, adler):
    """
    The zlib streams of the Deflate ``bodies`` (device byte arrays) with Adler-32 ``adler``.

    Returns the streams packed one after another on the device, with each
    one's offset and size (host int64 arrays).
    """
    import cupy as cp

    k = len(bodies)
    ifaces = [b.__cuda_array_interface__ for b in bodies]
    body = np.fromiter((i["shape"][0] for i in ifaces), np.int64, k)
    sizes = body + ZLIB_FRAMING
    offs = np.zeros(k, np.int64)
    np.cumsum(sizes[:-1], out=offs[1:])
    packed = cp.empty(int(sizes.sum()), cp.uint8)
    if k:
        ptrs = cp.asarray(np.fromiter((i["data"][0] for i in ifaces), np.uint64, k))
        _kernel("zlib_frame")((k,), (256,), (ptrs, packed, cp.asarray(offs), cp.asarray(body), adler))
    return packed, offs, sizes


def check_headers(cmf: np.ndarray, flg: np.ndarray) -> np.ndarray:
    """
    Whether each zlib header (CMF, FLG bytes) is one HDF5's deflate filter can have written.

    Deflate with a window of at most 32 KiB, a valid check value, and no preset dictionary.
    """
    cmf = cmf.astype(np.int64)
    flg = flg.astype(np.int64)
    return ((cmf & 0x0F) == 8) & ((cmf >> 4) <= 7) & (((cmf << 8) | flg) % 31 == 0) & ((flg & 0x20) == 0)
