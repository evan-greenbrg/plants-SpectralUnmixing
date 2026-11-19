import copy
import tempfile
import os
from pathlib import Path

import pandas
import numpy as np
from spectral import envi

from isofit.inversion.inverse_simple import invert_liquid_water


def full_spectrum(lib, values):
    """Put good-band values back onto the full wavelength grid (ones elsewhere).
    values: (n_bands_good,) or (n_spectra, n_bands_good)."""
    values = np.asarray(values)
    good = np.asarray(lib.good_bands, dtype=bool)
    out = np.ones(values.shape[:-1] + (good.size,))
    out[..., good] = values
    return out


def calculate_spectral_angle(spectrum, reference):
    """Spectral angle (SAM) between spectra, in degrees, along the last axis.

    Bands where either spectrum is non-finite are ignored. Inputs broadcast,
    so a single reference can be compared against many spectra.

    Args:
        spectrum (array-like): (..., n_bands) retrieved spectra.
        reference (array-like): (..., n_bands) reference spectra.

    Returns:
        float or ndarray: Angle in degrees per spectrum; NaN where there is
        no valid overlap or a spectrum has zero norm.
    """
    a, b = np.broadcast_arrays(np.asarray(spectrum, float), np.asarray(reference, float))
    m = np.isfinite(a) & np.isfinite(b)
    a, b = np.where(m, a, 0.0), np.where(m, b, 0.0)

    norm_product = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(a * b, axis=-1) / norm_product
    angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    angle = np.where(norm_product > 0, angle, np.nan)
    return angle.item() if angle.ndim == 0 else angle


def calculate_rmse(spectrum, reference):
    """RMSE between spectra along the last axis, ignoring non-finite bands."""
    a, b = np.broadcast_arrays(np.asarray(spectrum, float), np.asarray(reference, float))
    m = np.isfinite(a) & np.isfinite(b)
    n = m.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rmse = np.sqrt(np.sum(np.where(m, (a - b) ** 2, 0.0), axis=-1) / n)
    rmse = np.where(n > 0, rmse, np.nan)
    return rmse.item() if rmse.ndim == 0 else rmse


def calculate_ewt(args):
    rfl, wl, ewt_detection_limit = args
    return invert_liquid_water(rfl, wl, ewt_detection_limit=ewt_detection_limit)[0]
