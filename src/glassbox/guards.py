"""Guards that keep setup, tests, and smoke checks offline."""

from __future__ import annotations


class DownloadDisabledError(RuntimeError):
    """A call would fetch a model or dataset while downloads are disabled."""


def reject_download(allow_download: bool, *, resource: str) -> None:
    """Raise when a caller has not explicitly armed a download.

    Passing ``allow_download=True`` clears this guard and lets an explicit demo or
    training entry point load weights. Tests and the smoke check leave it false.
    """
    if allow_download:
        return
    raise DownloadDisabledError(
        f"Download of {resource} is disabled. Setup, tests, and smoke checks stay offline. "
        "Pass allow_download=True only from an explicit experiment entry point, after the "
        "asset license and schema have been reviewed."
    )
