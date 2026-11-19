import os
import logging
from pathlib import Path

import boto3

logger = logging.getLogger(__name__)

REGION = os.environ.get("AWS_REGION", "us-west-2")

# There is one fractional cover input file, the endmember library.
# This file is stored at ASD wavelengths and is resampled a run-time to
# processing chunk wavelengths. This introduces the only constraint on the
# processing grouping, that the rows have to share a wavelength grid.
#
# Placeholder location:
#   <FRAC_COVER_INPUTS_URI>/merged_asd_interp.csv
#
# This file is a requirement in the processing chain
# See README.md "ASD Spectral library".

INPUTS_URI = os.environ.get(
    "FRAC_COVER_INPUTS_URI",
    "s3://vswir-plants-config/frac-cover-inputs"
)

FILES = {
    "merged_asd_interp.csv":    "spectral_library"
}

s3 = boto3.client("s3", region_name=REGION)


def _download_prefix(bucket: str, prefix: str, dest: Path) -> list:
    """
    Download every object under prefix into dest, keeping relative paths. 
    Returns the relative paths.

    To Define: S3 structure for source
    """
    downloaded = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix):]
            if not rel or rel.endswith("/"):
                continue

            local = dest / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, obj["Key"], str(local))
            downloaded.append(rel)
    logger.info(
        f"inputs from s3://{bucket}/{prefix}: {downloaded or 'none — using defaults'}"
    )
    return downloaded


def fetch_inputs(dest: Path) -> dict:
    """
    Download the ASD spectral library into dest/

    Returns { "spectral_library" } -> local path 
    To Define: 
        Behavior if spectral library file is not found
    """

    bucket, _, root = INPUTS_URI.removeprefix("s3://").partition("/")
    root = root.strip("/")

    def prefix(*parts):
        return "/".join(p for p in (root, *parts) if p) + "/"

    out = {}
    local_dir = dest / "data"
    downloaded = _download_prefix(
        bucket,
        prefix("data"),     # Placeholder
        local_dir
    )
    for name, key in FILES.items():
        out[key] = str(local_dir / name) if name in downloaded else ""

    return out
