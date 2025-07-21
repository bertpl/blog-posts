import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Circle
from matplotlib.pyplot import Axes, Figure


def show_vectors_2d(v: torch.Tensor) -> tuple[Figure, Axes]:
    """
    Show vectors in 2D, which assumes v has 2 rows
    """

    # --- init figure ---------------------------
    fig, ax = plt.subplots(figsize=(5, 5))

    # --- plot vectors --------------------------
    n = v.size(dim=1)  # nr of vectors
    for i in range(n):
        x = 0.99 * v[0, i].item()
        y = 0.99 * v[1, i].item()
        ax.arrow(0, 0, x, y, head_width=0.05, head_length=0.1, fc="k", ec="k", length_includes_head=True)

    # --- decorate ------------------------------
    circle = Circle((0, 0), 1, color="k", alpha=0.5, lw=1, ls="--", fill=False, zorder=-1)
    ax.add_patch(circle)

    ax.set_title(f"{n} vectors in 2D")
    ax.grid(True, zorder=-100, alpha=0.2, linestyle="--")

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

    fig.tight_layout()

    # --- done ----------------------------------
    return fig, ax
