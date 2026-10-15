"""Lazy remote HDF5 access via HTTP range requests (fsspec).

WHY: `mdcath.fetch.download_domain()` pulls the ENTIRE per-domain .h5 file
(all 5 temperatures x 5 replicates), averaging ~611MB/domain (3.3TB total
across 5,398 domains). A resolvability triage only needs ONE replicate at
ONE temperature per domain -- a tiny slice. This module resolves the
domain's direct HTTPS URL on the HF Hub and opens it lazily via fsspec's
HTTPFileSystem, so h5py only fetches the byte ranges it actually touches
(group/attribute metadata + the one `coords` dataset read) instead of the
whole file. Nothing is written to local disk.

USAGE:
    from pdukit.lib.pdu.extract import remote_io, io as mio
    f = remote_io.open_domain_remote("1a3dA00")
    traj = mio.load_domain(f, domain="1a3dA00", temperature="320",
                           replicate="0", dt_ns=1.0)

CAVEATS (not verified against the live HF CDN from this environment --
no network access at write time; please sanity-check wall-clock time and
correctness on a handful of domains on your Mac before trusting this at
scale):

  1. HDF5's chunked storage means even a "targeted" read of one dataset can
     issue MANY small HTTP range requests if that dataset is chunked into
     many pieces on disk. `block_size` below controls fsspec's read-ahead
     granularity -- larger reduces request count but wastes more transfer
     per request. Tune if a domain is unexpectedly slow.
  2. This assumes the HF CDN honours HTTP Range requests (standard for
     S3/CDN-backed hosting, which HF uses, but not guaranteed for every
     mirror/region). fsspec will raise if range requests aren't supported;
     if you see failures here, fall back to `fetch.download_domain()`.
  3. Good fit for a ONE-PASS triage across many domains (read once,
     discard). If you'll revisit the SAME domain repeatedly, a normal
     download + local cache is more efficient than re-fetching ranges.
  4. Running many ProcessPoolExecutor workers means many concurrent HTTP
     connections to the HF CDN -- fine in general, but if you hit
     throttling/429s, reduce --workers.
"""

from __future__ import annotations

from typing import Optional

from .hdf5_layout import HF_FILE_TEMPLATE, HF_REPO_ID


def _require_fsspec():
    try:
        import fsspec
    except ImportError as exc:
        raise ImportError(
            "fsspec is required for remote HDF5 access: "
            "pip install fsspec aiohttp requests"
        ) from exc
    return fsspec


def remote_domain_url(
    domain: str, repo_id: str = HF_REPO_ID, token: Optional[str] = None
) -> str:
    """Return the HTTPS URL of a domain's .h5 file on the HF Hub."""
    from huggingface_hub import hf_hub_url

    return hf_hub_url(
        repo_id=repo_id,
        repo_type="dataset",
        filename=HF_FILE_TEMPLATE.format(domain=domain),
    )


def open_domain_remote(
    domain: str,
    repo_id: str = HF_REPO_ID,
    token: Optional[str] = None,
    block_size: int = 4 * 1024 * 1024,
):
    """Return a seekable file-like object over the remote .h5, backed by HTTP
    range requests. Pass this directly as the `path` argument to
    `mdcath.io.load_domain()` (or `_load_h5` internals) in place of a local
    path string -- h5py.File() accepts file-like objects transparently.

    `domain` must ALSO be passed explicitly to `load_domain(...,
    domain=domain)` since there is no filename to infer it from.

    block_size: fsspec read-ahead chunk size in bytes. 4MB is a reasonable
    default for HDF5 metadata + a single CA-coords dataset; increase if you
    see excessive request counts, decrease if you're wasting bandwidth on a
    small domain.
    """
    fsspec = _require_fsspec()
    url = remote_domain_url(domain, repo_id=repo_id, token=token)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    of = fsspec.open(url, mode="rb", headers=headers, block_size=block_size)
    return of.open()


def load_domain_remote(
    domain: str,
    temperature="320",
    replicate="0",
    stride: int = 1,
    max_frames: Optional[int] = None,
    dt_ns: Optional[float] = None,
    atom_selection: str = "CA",
    repo_id: str = HF_REPO_ID,
    token: Optional[str] = None,
    block_size: int = 4 * 1024 * 1024,
):
    """Open and load one domain/temperature/replicate over HTTP.

    Uses range requests; no local file is written.  Thin wrapper around
    `open_domain_remote()` + `mdcath.io.load_domain()`.
    """
    from . import io as mio

    f = open_domain_remote(
        domain, repo_id=repo_id, token=token, block_size=block_size
    )
    try:
        traj = mio.load_domain(
            f,
            domain=domain,
            temperature=temperature,
            replicate=replicate,
            stride=stride,
            max_frames=max_frames,
            dt_ns=dt_ns,
            atom_selection=atom_selection,
        )
        # io._load_h5() stores whatever `path` it was given as `traj.source` --
        # here that's the raw fsspec HTTPFile object, not a string. Downstream
        # code (run_domain.run(), batch_k3.fetch_and_run()) JSON-serializes the
        # full result dict including `source`, which fails on a non-JSON-able
        # object. Overwrite with a descriptive string before handing back.
        traj.source = f"remote:{remote_domain_url(domain, repo_id=repo_id)}"
        return traj
    finally:
        try:
            f.close()
        except Exception:
            pass
