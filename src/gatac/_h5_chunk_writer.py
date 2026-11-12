"""
The process that writes compressed chunks for :class:`gatac.io._csr_writer.CSRWriter`.

h5py runs every HDF5 call under one lock, so a thread writing chunks gets no
overlap with GPU compression; this process does the writing instead. It is
started from this file's path (``runpy.run_path``), not imported through
``gatac``: it needs only h5py and numpy, and it never runs the caller's
``__main__`` (as a ``multiprocessing`` spawn would).

Usage, by the parent only::

    python -c "import runpy; runpy.run_path(THIS_FILE, run_name='__main__')" FD_IN FD_OUT BUF ...

``BUF`` are files in ``/dev/shm`` the parent fills with zlib streams; this
process maps them, then replies ``("ready",)`` so that the parent can
unlink them. Commands on ``FD_IN`` (pickled through
``multiprocessing.connection``)::

    ("open", path, {dataset: chunk elements})   reply ("opened",)
    ("batch", dataset, buf, c0, offsets, sizes) reply ("free", buf), once written
    ("close",)                                  reply ("closed", seconds spent writing)
    ("exit",)

Any error is replied as ``("error", traceback)`` and ends the process.
"""

from __future__ import annotations

import mmap
import os
import sys
import time
import traceback
from multiprocessing.connection import Connection


def serve(conn_in: Connection, conn_out: Connection, buffers: list[str]) -> None:
    import h5py
    import numpy as np

    maps = []
    for path in buffers:
        fd = os.open(path, os.O_RDONLY)
        try:
            maps.append(mmap.mmap(fd, os.fstat(fd).st_size, prot=mmap.PROT_READ))
        finally:
            os.close(fd)
    views = [np.frombuffer(m, np.uint8) for m in maps]
    conn_out.send(("ready",))
    f = None
    datasets: dict = {}
    elems: dict = {}
    spent = 0.0
    try:
        while True:
            msg = conn_in.recv()
            if msg[0] == "open":
                _, path, elems = msg
                f = h5py.File(path, "r+")
                datasets = {k: f[k] for k in elems}
                spent = 0.0
                conn_out.send(("opened",))
            elif msg[0] == "batch":
                _, key, b, c0, offs, sizes = msg
                t = time.perf_counter()
                d, n = datasets[key], elems[key]
                end = (c0 + len(sizes)) * n
                if d.shape[0] < end:
                    d.resize((end,))
                for i in range(len(sizes)):
                    d.id.write_direct_chunk(((c0 + i) * n,), views[b][offs[i] : offs[i] + sizes[i]])
                spent += time.perf_counter() - t
                conn_out.send(("free", b))
            elif msg[0] == "close":
                datasets = {}
                if f is not None:
                    f.close()
                    f = None
                conn_out.send(("closed", spent))
            elif msg[0] == "exit":
                break
    except BaseException:  # noqa: BLE001 - any failure goes to the parent, which raises it
        try:
            conn_out.send(("error", traceback.format_exc()))
        except OSError:
            pass
    finally:
        datasets = {}
        if f is not None:
            f.close()
        del views
        for m in maps:
            m.close()


if __name__ == "__main__":
    fd_in, fd_out = int(sys.argv[1]), int(sys.argv[2])
    serve(Connection(fd_in, writable=False), Connection(fd_out, readable=False), sys.argv[3:])
