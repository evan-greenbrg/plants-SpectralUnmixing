from datetime import datetime
from functools import partial
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import logging
import os
import traceback

import numpy as np
import pandas
from pathlib import Path

os.environ.setdefault("PYTHON_JULIACALL_THREADS", "auto")
os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes") 
from juliacall import Main as jl

from utils import (
    full_spectrum,
    calculate_spectral_angle,
    calculate_rmse,
    calculate_ewt
    
)

logger = logging.getLogger(__name__)


# Julia imports
jl.seval("using SpectralUnmixing")
jl.include(str(Path(__file__).resolve().parent / "unmix_batch.jl"))
SU = jl.SpectralUnmixing
UB = jl.UnmixBatch

# Map database sensor names to isofit short keys expected by FID() and sensor_name_to_dt()
SENSOR_KEY_MAP = {
    'NEON AIS 1':      'neon',
    'NEON AIS 2':      'neon',
    'NEON AIS 3':      'neon',
    'AVIRIS-Classic':  'avcl',
    'AVIRIS-NG':       'ang',
    'AVIRIS-3':        'av3',
    'AVIRIS-5':        'av5',
}

# Enforce Julia data types
JVEC_STR = lambda xs: jl.Vector[jl.String](xs)
JVEC_INT = lambda xs: jl.Vector[jl.Int64](xs)
JVEC_FLOAT = lambda xs: jl.Vector[jl.Float64](np.asarray(xs, dtype=np.float64))

# Config matches EMIT configuration
UNMIXING_CONFIG = dict(
    mode="sma",
    n_mc=50,
    num_endmembers=JVEC_INT([10]),
    normalization="brightness",
    optimization="bvls",
    max_combinations=100,
    combination_type="class-even",
)


def retrieve_fractional_cover(
    pixels: list,
    wl,
    sensor_name: str,
    rundir: str,
    job_id: str,
    inputs: dict,
    n_cores: int = 1,
) -> dict:
    try:
        isofit_sensor = SENSOR_KEY_MAP.get(sensor_name)
        if not isofit_sensor:
            raise ValueError(
                f"Unknown sensor name '{sensor_name}'. "
                "Add it to SENSOR_KEY_MAP in inversion.py"
            )

        logger.info(
            f"Running inversion on {len(pixels)} pixels — "
            "sensor: {sensor_name} → {isofit_sensor}"
        )

        # ── Diagnostic logging ────────────────────────────────────────────
        wl_arr = wl[isofit_sensor]

        logger.info(
            f"wl type={type(wl_arr).__name__} "
            f"len={len(wl_arr)} "
            f"dtype={np.array(wl_arr).dtype} "
            f"range={wl_arr[0]:.1f}–{wl_arr[-1]:.1f} nm"
        )

        # Load in Library and convolve to pixel wavelengths
        endmember_library = SU.SpectralLibrary(
            inputs["spectral_library"],
            "level_1", 1, 7, JVEC_STR(["soil", "npv", "pv"])
        )

        SU.load_data_b(endmember_library)
        SU.filter_by_class_b(endmember_library)
        SU.interpolate_library_to_new_wavelengths_b(
            endmember_library,
            JVEC_FLOAT(wl)
        )
        full_el = jl.deepcopy(endmember_library)
        SU.remove_wavelength_region_inplace_b(full_el, False)

        # Set up input data to only use "good channels"
        good_channels = np.asarray(
            endmember_library.good_bands,
            dtype=bool
        )
        # Package pixels
        # TODO: What does pixels look like coming out of the schema?
        rfl = np.ascontiguousarray(np.array(pixels))

        # TODO uncertainty
        # unc = np.ascontiguousarray(np.array(pixel_unc)

        assert rfl.shape[1] == np.asarray(full_el.spectra).shape[1]

        class_idx = SU.prepare_combinations(
            full_el, UNMIXING_CONFIG["combination_type"]
        )
        options = SU.prepare_options(
            full_el,
            UNMIXING_CONFIG["combination_type"],
            UNMIXING_CONFIG["num_endmembers"],
            class_idx
        )
        mr, mv, cfr, cfv = UB.unmix_matrix(
            full_el, rfl, None, class_idx, options,
            n_cores=n_cores,
            **UNMIXING_CONFIG,
        )
        mr, cfr = np.asarray(mr), np.asarray(cfr)
        class_keys = list(full_el.class_valid_keys)

        # Reconstruct
        E = np.asarray(full_el.spectra)
        E_norm = E / np.sqrt(np.mean(E**2, axis=1, keepdims=True))
        mod_rfl = (cfr[:, :-1] @ E_norm) * cfr[:, -1:]
        mod_full_rfl = full_spectrum(endmember_library, mod_rfl)[:, good_channels]

        # 5. Parse Outputs
        # mr packaged as:
        # Soil, NPV, PV, Brightness
        soil_frac = mr[0]
        npv_frac = mr[1]
        pv_frac = mr[2]
        brightness = mr[3]
        sa = calculate_spectral_angle(rfl, mod_full_rfl)
        rmse = calculate_rmse(rfl, mod_full_rfl)

        result = {
            "soil_fraction": mr[0],
            "npv_fraction": mr[1],
            "pv_fraction": mr[2],
            "brightness": mr[3],
            "model_fit_spectral_angle": sa,
            "model_fit_rmse": rmse,
        }

        return {"status": "success", "result": result}

    except Exception as e:
        logger.error(f"Inversion failed: {e}")

        return {
            "status": "error",
            "message": str(e),
            "traceback": str(traceback.format_exc()),
            "error_type": type(e).__name__
        }


def retrieve_ewt(
    pixels: list,
    wl: np.ndarray,
    ewt_detection_limit: float,
    rundir: str,
    job_id: str,
    n_cores: int = 1,
):
    try:
        # TODO: What does pixels look like coming out of the schema?
        rfl = np.ascontiguousarray(np.array(pixels))
        tasks = [(r, wl, ewt_detection_limit) for r in rfl]

        if n_cores <= 1:
            solutions = [invert_pixel(t) for t in tasks]
        else:
            with ProcessPoolExecutor(
                max_workers=n_cores,
                mp_context=mp.get_context("spawn"),
            ) as pool:
                chunksize = max(1, len(tasks) // (n_cores * 4))
                solutions = list(pool.map(calculate_ewt, tasks, chunksize=chunksize))

        result = {
            "equivalent_water_thickness_ewt": solutions
        }

        return {"status": "success", "result": result}

    except Exception as e:
        logger.error(f"Inversion failed: {e}")

        return {
            "status": "error",
            "message": str(e),
            "traceback": str(traceback.format_exc()),
            "error_type": type(e).__name__
        }

