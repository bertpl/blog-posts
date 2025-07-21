import math
import time

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


# =================================================================================================
#  Optimizer class
# =================================================================================================
class VectorOptimizer:
    """
    Class to optimize vectors in a given number of dimensions, minimizing the maximum absolute scaled dot product.
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, n_dims: int, n_vectors: int, alpha: float = 10.0, use_float64: bool = False):
        """
        Initialize the optimizer with the number of vectors, dimensions, and alpha parameter.
        """
        self.n_dims = n_dims
        self.n_vectors = n_vectors
        self.alpha = alpha
        self.use_float64 = use_float64
        self.t_elapsed = 0.0  # time elapsed in seconds

        # Initialize vectors to be optimized
        self.v = self._init_vectors(n_dims, n_vectors, use_float64)

        # Initialize optimizer
        self._hist_loss: list[float] = []
        self._hist_lr: list[float] = []
        self._optimizer: torch.optim.Optimizer | None = None

    @property
    def n_steps(self) -> int:
        return len(self._hist_loss)

    @property
    def dtype(self) -> str:
        return "float64" if use_float64 else "float32"

    # -------------------------------------------------------------------------
    #  Public Methods
    # -------------------------------------------------------------------------
    def solve(self, lr_max: float = 1.0, lr_min: float = 1e-9, verbose: bool = True):
        """
        Solves the optimization problem with an adaptive learning rate schedule & stopping criterion.

        Strategy:
         - start with 100 steps of warmup, ramping up from lr_min to lr_max
         - perform 100 steps of optimization at a time as long as loss decreases (with a max. of 10000 steps per lr)
           - as long as loss decreases by a certain fixed amount, keep lr constant
           - if not, reduce learning rate by a factor of 2 with lr_min as a minimum
        - if lr == lr_min, and we stopped making progress, stop optimization

        """

        # print what we will do
        t_start = time.time_ns()
        if verbose:
            print(
                f"Optimizing {self.n_vectors:>5} vectors in {self.n_dims:>5} dimensions "
                + f"with alpha={self.alpha:.1f} [{self.dtype}]."
            )

        # configure optimizer
        self._optimizer = torch.optim.Adam([self.v], lr=lr_min)

        # warmup phase
        self._optimize_multiple_steps(n_steps=100, lr_initial=lr_min, lr_final=lr_max, verbose=verbose)

        # main optimization loop
        progress_threshold_abs = 1e-24  # threshold for progress in relative loss (if less, we stop for this lr)
        lr_last = lr_max
        lr_values = self._lr_values(lr_max, lr_min)  # decreasing learning rates
        if verbose:
            lr_values = tqdm(lr_values)

        for lr in lr_values:
            n_steps = 100
            for i in range(10):
                # do max. 10x100 steps for any given learning rate

                # do n_steps steps
                self._optimize_multiple_steps(n_steps=n_steps, lr_initial=lr_last, lr_final=lr, verbose=verbose)
                lr_last = lr

                # move on to next learning rate or stop if we progress too slowly
                if self._hist_loss[-1] >= self._hist_loss[-n_steps] - progress_threshold_abs:
                    break

        # print final result
        t_end = time.time_ns()
        t_elapsed_sec = (t_end - t_start) / 1_000_000_000  # convert to seconds
        self.t_elapsed = t_elapsed_sec
        if verbose:
            max_abs_sdp = max_abs_scaled_dot_product(self.v)
            print()
            print(
                f"Optimization finished in {len(self._hist_lr):>5} steps, {t_elapsed_sec:.1f} seconds. "
                + f"Max. abs. scaled dot product: {max_abs_sdp:.6e}."
            )
            print()

    # -------------------------------------------------------------------------
    #  Internal Optimization Methods
    # -------------------------------------------------------------------------
    def _optimize_multiple_steps(self, n_steps: int, lr_initial: float, lr_final: float, verbose: bool = False):
        """
        Perform n_steps of optimization, with exponential learning rate schedule, starting at lr_initial and ending at
        lr_final.
        """

        for i in range(n_steps):
            lr = lr_initial * ((lr_final / lr_initial) ** (i / (n_steps - 1)))
            self._optimize_one_step(lr)

        if verbose:
            loss_before = self._hist_loss[-(n_steps - 1)]
            loss_after = self._hist_loss[-1]
            print(
                f"   Optimized for {n_steps:>3} steps.  lr: {lr_initial:.3e} -> {lr_final:.3e}, "
                + f" loss: {loss_before:.9e} -> {loss_after:.9e}."
            )

    def _optimize_one_step(self, lr: float):
        # set learning rate in solver
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

        # single optimization step
        self._optimizer.zero_grad()  # clear previous gradients
        loss = self._compute_loss(self.v)
        loss.backward()  # compute gradients
        self._optimizer.step()  # update

        # remember loss and learning rate
        self._hist_loss.append(loss.item())
        self._hist_lr.append(lr)

    # -------------------------------------------------------------------------
    #  Internal Helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _init_vectors(n_dims: int, n_vectors: int, use_float64: bool) -> torch.Tensor:
        """
        Initialize a tensor of vectors with random values, to be optimized.
        """

        # initialize vectors of size (n_dims, n_vectors) (1 column = 1 vector) of appropriate dtype
        v = torch.randn(n_dims, n_vectors, dtype=torch.float64 if use_float64 else torch.float32)

        # set to appropriate device
        if not use_float64 and torch.backends.mps.is_available():
            v = v.to(torch.device("mps"))
        elif torch.cuda.is_available():
            v = v.to(torch.device("cuda"))
        else:
            v = v.to(torch.device("cpu"))

        # normalize, set required_grad=True and return
        v / v.norm(p=2, dim=0, keepdim=True)  # normalize vectors to have 2-norm = 1
        v.requires_grad_(True)  # these are optimizable parameters
        return v

    @staticmethod
    def _lr_values(lr_max: float, lr_min: float) -> list[float]:
        # Generate a list of decreasing learning rates for the optimization process
        n = math.ceil(math.log2(lr_max / lr_min) + 1)  # number of steps to go from lr_max to lr_min
        return [lr_max / (lr_max / lr_min) ** (i / (n - 1)) for i in range(n)]

    def _compute_loss(self, v: torch.Tensor) -> torch.Tensor:
        # PART 1 - off-diagonal scaled dot products --> should be close to 0
        v_norm = v / v.norm(p=2, dim=0, keepdim=True)  # divide by 2-norm, column-wise
        off_diag_scaled_dot_products = torch.triu(
            v_norm.T @ v_norm,  # matrix of ALL scaled dot products
            diagonal=1,
        ).view(-1)  # upper triangle, excluding diagonal, vectorized

        # take the smoothed absolute maximum of the off-diagonal elements,
        #   so we have a sufficiently stable loss function
        loss = smooth_abs_max(off_diag_scaled_dot_products, self.alpha)

        # PART 2 - vector norms --> should be close to 1
        loss += (torch.pow(v.norm(p=2, dim=0) - 1, 2)).sum()

        return loss


# =================================================================================================
#  Helpers
# =================================================================================================
# def init_vectors(n_dims: int, n_vectors: int) -> torch.Tensor:
#     """
#     Initialize a tensor of vectors with random values, to be optimized.
#     """
#     v = torch.randn(n_dims, n_vectors, dtype=torch.float64)  # columns contain vectors
#     # v = v.to(torch.device("mps"))  # use Apple GPU device
#     v / v.norm(p=2, dim=0, keepdim=True)  # normalize vectors to have 2-norm = 1
#     v.requires_grad_(True)  # these are optimizable parameters
#
#     return v


def smooth_abs_max(x: torch.Tensor, alpha: float = 10.0) -> torch.Tensor:
    """
    Computes smooth approximation of the absolute maximum of a tensor.

      smooth_abs_max = smooth_max([x; -x])

    """
    # return (1 / alpha) * torch.log((torch.exp(alpha * x) + torch.exp(-alpha * x)).sum())
    return smooth_max(torch.cat((x, -x), dim=-1), alpha)  # smooth max of both x and -x


def smooth_max(x: torch.Tensor, alpha: float = 10.0) -> torch.Tensor:
    """
    Computes smooth approximation of the maximum of a tensor, by computing a weighted sum of all elements,
    weighted by a softmax function.

        smooth_max(v) = (1/sum_i(exp(alpha * v[i]))) * sum_i(exp(alpha * v[i]) * v[i])

    """
    return (F.softmax(alpha * x, dim=-1) * x).sum()


# def compute_loss(v: torch.Tensor, alpha: float) -> torch.Tensor:
#     """
#     Compute the loss for a given epsilon, with only |scaled dot products| > epsilon
#       contributing to the loss.
#     """
#
#     # part 1 - off-diagonal scaled dot products
#     v_norm = v / v.norm(p=2, dim=0, keepdim=True)  # divide by 2-norm, column-wise
#     off_diag_scaled_dot_products = torch.triu(
#         v_norm.T @ v_norm,  # matrix of ALL scaled dot products
#         diagonal=1,
#     ).view(-1)  # upper triangle, excluding diagonal, vectorized
#
#     # take the smoothed absolute maximum of the off-diagonal elements,
#     #   so we have a sufficiently stable loss function
#     loss = smooth_abs_max(off_diag_scaled_dot_products, alpha)
#
#     # part 2 - vector norms
#     loss += (torch.pow(v.norm(p=2, dim=0) - 1, 2)).sum()
#
#     return loss


def scaled_dot_products(v: torch.Tensor) -> torch.Tensor:
    """
    Compute the scaled dot products of the vectors i,j with j>i.
    """
    v_norm = v / v.norm(p=2, dim=0, keepdim=True)  # divide by 2-norm, column-wise
    off_diag_scaled_dot_products = torch.triu(
        v_norm.T @ v_norm,  # matrix of ALL scaled dot products
        diagonal=1,
    ).view(-1)  # upper triangle, excluding diagonal, vectorized

    return off_diag_scaled_dot_products


def max_abs_scaled_dot_product(v: torch.Tensor) -> float:
    return torch.max(torch.abs(scaled_dot_products(v))).item()


# def _set_adam_lr(optimizer: torch.optim.Optimizer, lr: float):
#     for param_group in optimizer.param_groups:
#         param_group["lr"] = lr


# def optimize_vectors(
#     v: torch.Tensor,
#     alpha: float,
#     n_steps: int = 1000,
#     min_lr: float = 1e-3,
#     max_lr: float = 1e-3,
#     verbose: bool = True,
# ) -> torch.Tensor:
#     """
#     Optimize the vectors using gradient descent (Adam optimizer) where we want to minimize the maximum
#     absolute scaled dot product of the vectors, while keeping their norms close to 1.
#     """
#
#     n_dims = v.size(dim=0)
#     n_vectors = v.size(dim=1)
#
#     optimizer = torch.optim.Adam([v], lr=min_lr)
#
#     if verbose:
#         steps_range = tqdm(list(range(n_steps)), desc=f"Optimizing {n_vectors} vectors in {n_dims} dims")
#     else:
#         steps_range = range(n_steps)
#
#     for step in steps_range:
#         # update learning rate
#         lr = min_lr * ((max_lr / min_lr) ** (step / (n_steps - 1)))
#         _set_adam_lr(optimizer, lr)
#
#         # perform one step
#         optimizer.zero_grad()  # clear previous gradients
#         loss = compute_loss(v, alpha)  # compute loss
#         loss.backward()  # compute gradients
#         optimizer.step()  # update parameters
#
#         if verbose and (step % 100 == 0):
#             print(f"Step {step:>5}, lr: {lr:.9f}, Loss: {loss.item():.9f}")
#
#     return v
#
#
# def optimize_vectors_auto(v: torch.Tensor, alpha: float, verbose: bool = True) -> torch.Tensor:
#     """
#     Optimize the vectors using gradient descent (Adam optimizer) where we want to minimize the maximum
#     absolute scaled dot product of the vectors, while keeping their norms close to 1.
#     """
#
#     n_dims = v.size(dim=0)
#     n_vectors = v.size(dim=1)
#
#     n_steps = int(3 * math.log2(n_dims * n_vectors)) * 1_000
#     min_lr = 1.0
#     max_lr = 1e-9
#
#     return optimize_vectors(v, alpha, n_steps, min_lr, max_lr, verbose)
