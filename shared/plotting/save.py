from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.pyplot import Figure

from shared.paths import get_figures_folder


# =================================================================================================
#  Save figures in various formats
# =================================================================================================
def save_fig(
    fig: Figure,
    post_nr: int,
    name: str,
    target_display_dpi: int = 100,
):
    path = get_figures_folder(post_nr)

    # classic file formats
    fig.savefig(path / f"{name}.png", dpi=600)
    fig.savefig(path / f"{name}.jpg", dpi=600)

    # save as webp, intended for web display, with quality optimized for target file size
    save_fig_as_webp(fig, path / f"{name}.webp", target_display_dpi)


# =================================================================================================
#  webP optimized saving
# =================================================================================================
def save_fig_as_webp(
    fig: Figure,
    filename: Path,
    target_display_dpi: int = 100,
    max_bits_per_displayed_pixel: float = 2.5,
):
    # --- prep --------------------------------------------
    t_start = time.time_ns()
    wh_inch = fig.get_size_inches()
    w_px, h_px = target_display_dpi * wh_inch[0], target_display_dpi * wh_inch[1]
    max_file_size_bytes = int((w_px * h_px) * max_bits_per_displayed_pixel / 8)
    if filename.exists():
        filename.unlink(missing_ok=True)

    # --- try lossless ------------------------------------
    settings = [
        WebpSettings(name=f"{dpi}dpi_lossless", dpi=dpi, pil_kwargs=dict(lossless=True))
        for dpi in range(300, 901, 1)  # 300, 301, ..., 900
    ]
    opt_setting, file_size = bisect_webp_settings(fig, filename, settings, max_file_size_bytes)

    # --- try lossy ---------------------------------------
    if file_size > max_file_size_bytes:
        # only try lossy if lossless exceeds max file size
        settings = [
            WebpSettings(name=f"{int(dpi)}dpi_q{int(q)}", dpi=int(dpi), pil_kwargs=dict(quality=int(q), method=6))
            for q, dpi in zip(
                np.linspace(40, 100, 601),
                np.linspace(200, 800, 601),
            )
        ]
        opt_setting, file_size = bisect_webp_settings(fig, filename, settings, max_file_size_bytes)

    # --- finalize ----------------------------------------

    # make sure we saved with optimal setting
    _save_as_webp_with_settings(fig, filename, opt_setting)

    # print result
    filesize_bytes = filename.stat().st_size
    bits_per_pixel = (filesize_bytes * 8) / (w_px * h_px)
    t_elapsed_sec = (time.time_ns() - t_start) / 1_000_000_000
    print(
        f"Saved '{filename.name}' with setting '{opt_setting.name}', resulting in {filesize_bytes:_} bytes ({bits_per_pixel:.2f} bits/pixel @ {target_display_dpi}dpi)  ({t_elapsed_sec:.1f}sec)"
    )


# -------------------------------------------------------------------------
#  WebP helpers
# -------------------------------------------------------------------------
def bisect_webp_settings(
    fig: Figure,
    filename: Path,
    settings: list[WebpSettings],
    max_file_size_bytes,
) -> tuple[WebpSettings, int]:
    """
    Bisect over a list of settings (sorted from low to high quality) to find the highest quality settings that fit
    within the max file size.

    If lowest quality settings already exceed the max file size, this one is selected.

    Returns final (settings, filesize_bytes) tuple.
    """

    # --- init --------------------------------------------
    i_min, i_max = 0, len(settings) - 1

    # check lowest quality settings
    size_min = _save_as_webp_with_settings(fig, filename, settings[i_min])
    if size_min > max_file_size_bytes:
        # lowest quality settings exceed max file size, return them
        return settings[i_min], size_min

    # check highest quality settings
    size_max = _save_as_webp_with_settings(fig, filename, settings[i_max])
    if size_max <= max_file_size_bytes:
        # highest quality settings already fit, return them
        return settings[i_max], size_max

    # --- bisect ------------------------------------------
    while i_min < i_max - 1:
        # bisect with invariant size_min <= max_file_size_bytes < size_max
        # until i_max = i_min + 1
        i_mid = (i_min + i_max) // 2
        size_mid = _save_as_webp_with_settings(fig, filename, settings[i_mid])

        if size_mid > max_file_size_bytes:
            i_max = i_mid
        else:
            i_min = i_mid

    # --- return ------------------------------------------
    return settings[i_min], size_min


@dataclass
class WebpSettings:
    name: str
    dpi: float
    pil_kwargs: dict


def _save_as_webp_with_settings(
    fig: Figure,
    filename: Path,
    settings: WebpSettings,
) -> int:
    # save using these settings
    fig.savefig(filename, dpi=settings.dpi, pil_kwargs=settings.pil_kwargs)

    # return number of bytes saved
    return filename.stat().st_size
