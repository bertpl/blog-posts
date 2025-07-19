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

    # We just save as lossless .webp, in very high resolution (significantly higher than needed)
    # We append the expected display pixel width to the file name (e.g "figure_800px.webp",
    # with 800=width inches * target_display_dpi), so next tools in the toolchain can use this info
    # to optimize the images appropriately for web display.
    w_inch = fig.get_size_inches()[0]
    w_display_px = int(w_inch * target_display_dpi)
    fig.savefig(
        path / f"{name}_{w_display_px}px.webp",
        dpi=10 * target_display_dpi,
        pil_kwargs=dict(
            lossless=True,  # lossless, no quality loss
            quality=100,  # highest compression effort
            method=1,  # higher methods are slower but not significantly smaller
        ),
    )
