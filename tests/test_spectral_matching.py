import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.optimize
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helper_fcns import (  # noqa: E402
    F_root,
    _direct_reference_scores,
    reference_changepoint,
    reference_half_spectrum,
    support_bracket,
)
import wsindy as wsindy_module  # noqa: E402
from wsindy import WSINDy  # noqa: E402


def lightweight_model(U, tau=1e-10, tau_hat=2, X=None):
    model = object.__new__(WSINDy)
    model.U = U
    model.X = X if X is not None else [
        torch.linspace(0, 1, n, dtype=torch.float64) for n in U.shape
    ]
    model.tau = tau
    model.tau_hat = tau_hat
    model.verbosity = False
    model.eqn_type = "ode" if U.ndim == 1 else "pde"
    return model


@pytest.mark.parametrize("N", [5, 6, 127, 128, 301])
def test_rfft_reconstructs_reference_fftshift_half_spectrum(N):
    generator = torch.Generator().manual_seed(4 + N)
    U = torch.randn((N, 7), generator=generator, dtype=torch.float64)

    full = torch.abs(torch.fft.fftshift(torch.fft.fft(U, dim=0), dim=0))
    full = full.mean(dim=1)[: int(np.ceil(N / 2))]
    reconstructed = reference_half_spectrum(U, 0)

    assert reconstructed.shape == full.shape
    assert torch.allclose(reconstructed, full, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("kind", ["random", "piecewise", "tie", "near_tie"])
def test_prefix_scores_match_direct_reference_objective(kind):
    generator = torch.Generator().manual_seed(17)
    if kind == "random":
        spectrum = torch.exp(torch.randn(96, generator=generator, dtype=torch.float64))
    elif kind == "piecewise":
        spectrum = torch.cat((0.2 * torch.ones(35), 2.0 * torch.ones(61))).double()
    elif kind == "tie":
        spectrum = torch.ones(96, dtype=torch.float64)
    else:
        spectrum = torch.ones(96, dtype=torch.float64)
        spectrum += 1e-10 * torch.randn(96, generator=generator, dtype=torch.float64)

    H = torch.cumsum(spectrum, dim=0)
    kc, split, optimized = reference_changepoint(H)
    direct = _direct_reference_scores(H)
    direct_winner = int(torch.argmin(direct).item())
    expected_split = direct_winner + 1

    assert split == expected_split
    assert kc == max(H.numel() - expected_split - 3, 1)
    assert torch.allclose(optimized, direct, rtol=1e-8, atol=1e-10)


def test_published_bracket_changes_sign_and_old_guess_is_midpoint():
    N, kc, tau_hat, tau = 256, 24, 2, 1e-10
    lower, upper = support_bracket(kc, N, tau_hat, tau)
    old_guess = lower * (1 + np.sqrt(1 - (8 / np.sqrt(3)) * np.log(tau))) / 2

    assert old_guess == pytest.approx((lower + upper) / 2)
    assert F_root(lower, kc, N, tau_hat, tau) >= 0
    assert F_root(upper, kc, N, tau_hat, tau) <= 0

    root = scipy.optimize.brentq(F_root, lower, upper,
                                 args=(kc, N, tau_hat, tau))
    assert root == pytest.approx(16.7577329062, rel=1e-10)
    assert abs(F_root(root, kc, N, tau_hat, tau)) < 1e-6


def test_tau_hat_default_and_root_propagation():
    signature = inspect.signature(WSINDy.__init__)
    assert signature.parameters["tau_hat"].default == 2
    assert "init_kc_guess" not in signature.parameters

    N, kc, tau = 256, 24, 1e-10
    roots = []
    for tau_hat in (1, 2):
        lower, upper = support_bracket(kc, N, tau_hat, tau)
        roots.append(scipy.optimize.brentq(
            F_root, lower, upper, args=(kc, N, tau_hat, tau)))
    assert roots[0] != pytest.approx(roots[1])
    assert roots[1] > roots[0]


def test_deterministic_noisy_ks_reference_result():
    U_clean = np.loadtxt(ROOT / "data" / "KS.txt", delimiter=",")
    rng = np.random.default_rng(12345)
    sigma = 0.5 * np.sqrt(np.mean(U_clean**2))
    U = torch.tensor(U_clean + sigma * rng.normal(size=U_clean.shape),
                     dtype=torch.float64)

    expected_k = (22, 16)
    expected_root = (18.0247860633, 27.1357550034)
    expected_m = (19, 28)
    axes = [
        torch.linspace(0, 32*np.pi, U.shape[0], dtype=torch.float64),
        torch.linspace(0, 150, U.shape[1], dtype=torch.float64),
    ]
    model = lightweight_model(U, X=axes)

    for d in range(2):
        spectrum = reference_half_spectrum(U, d)
        kc, _, _ = reference_changepoint(torch.cumsum(spectrum, dim=0))
        lower, upper = support_bracket(kc, U.shape[d], 2, 1e-10)
        root = scipy.optimize.brentq(
            F_root, lower, upper, args=(kc, U.shape[d], 2, 1e-10))

        assert kc == expected_k[d]
        assert root == pytest.approx(expected_root[d], rel=1e-10)
        assert model.spectral_matching(d) == expected_m[d]


@pytest.mark.parametrize(
    "tau,tau_hat,message",
    [(0, 2, "tau must"), (1, 2, "tau must"), (1e-10, 0, "tau_hat")],
)
def test_invalid_hyperparameters_raise(tau, tau_hat, message):
    U = torch.randn(32, dtype=torch.float64)
    model = lightweight_model(U, tau=tau, tau_hat=tau_hat)
    with pytest.raises(ValueError, match=message):
        model.spectral_matching(0)


def test_nonfinite_data_and_nonuniform_axis_raise():
    U = torch.randn(32, dtype=torch.float64)
    U[3] = torch.nan
    with pytest.raises(ValueError, match="finite data"):
        lightweight_model(U).spectral_matching(0)

    U = torch.randn(32, dtype=torch.float64)
    axis = torch.linspace(0, 1, 32, dtype=torch.float64)
    axis[12] += 1e-3
    with pytest.raises(ValueError, match="uniformly spaced"):
        lightweight_model(U, X=[axis]).spectral_matching(0)


def test_degenerate_spectrum_and_failed_existence_condition_fall_back():
    U = torch.zeros(32, dtype=torch.float64)
    model = lightweight_model(U)
    with pytest.warns(RuntimeWarning, match="degenerate spectrum"):
        assert model.spectral_matching(0) == 15

    generator = torch.Generator().manual_seed(8)
    U = torch.randn(64, generator=generator, dtype=torch.float64)
    model = lightweight_model(U, tau_hat=1e6)
    with pytest.warns(RuntimeWarning, match="root-existence condition"):
        assert model.spectral_matching(0) == 31


def test_spectral_matching_has_no_injected_helper_arguments():
    assert list(inspect.signature(WSINDy.spectral_matching).parameters) == ["self", "d"]


def test_too_short_or_incompatible_axes_raise():
    U = torch.randn(4, dtype=torch.float64)
    with pytest.raises(ValueError, match="at least five"):
        lightweight_model(U).spectral_matching(0)

    U = torch.randn(32, dtype=torch.float64)
    with pytest.raises(ValueError, match="coordinate axes"):
        lightweight_model(U, X=[]).spectral_matching(0)


def test_complex_data_raises():
    U = torch.randn(32, dtype=torch.float64).to(torch.complex128)
    U += 1j * torch.randn(32, dtype=torch.float64)
    with pytest.raises(ValueError, match="real-valued"):
        lightweight_model(U).spectral_matching(0)


@pytest.mark.parametrize("axis_kind,message", [
    ("wrong_length", "32 entries"),
    ("nonfinite", "real and finite"),
    ("decreasing", "strictly increasing"),
])
def test_invalid_coordinate_axis_raises(axis_kind, message):
    U = torch.randn(32, dtype=torch.float64)
    if axis_kind == "wrong_length":
        axis = torch.linspace(0, 1, 31, dtype=torch.float64)
    elif axis_kind == "nonfinite":
        axis = torch.linspace(0, 1, 32, dtype=torch.float64)
        axis[5] = torch.inf
    else:
        axis = torch.linspace(1, 0, 32, dtype=torch.float64)
    with pytest.raises(ValueError, match=message):
        lightweight_model(U, X=[axis]).spectral_matching(0)


def test_undefined_relative_weights_fall_back(monkeypatch):
    monkeypatch.setattr(
        wsindy_module, "reference_half_spectrum",
        lambda *args: torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64),
    )
    U = torch.ones(33, dtype=torch.float64)
    with pytest.warns(RuntimeWarning, match="undefined relative spectral weights"):
        assert lightweight_model(U).spectral_matching(0) == 16


def test_no_finite_changepoint_score_falls_back(monkeypatch):
    def fail_changepoint(_):
        raise ValueError("No finite reference changepoint score was found.")

    monkeypatch.setattr(wsindy_module, "reference_changepoint", fail_changepoint)
    U = torch.randn(64, dtype=torch.float64)
    with pytest.warns(RuntimeWarning, match="No finite reference changepoint score"):
        assert lightweight_model(U).spectral_matching(0) == 31


def test_missing_root_sign_change_falls_back(monkeypatch):
    monkeypatch.setattr(wsindy_module, "reference_changepoint", lambda _: (8, 1, None))
    monkeypatch.setattr(wsindy_module, "F_root", lambda *args: -1.0)
    U = torch.randn(64, dtype=torch.float64)
    with pytest.warns(RuntimeWarning, match="bracket does not change sign"):
        assert lightweight_model(U).spectral_matching(0) == 31


def test_root_solver_failure_falls_back(monkeypatch):
    monkeypatch.setattr(wsindy_module, "reference_changepoint", lambda _: (8, 1, None))

    def fail_solver(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(wsindy_module.scipy.optimize, "brentq", fail_solver)
    U = torch.randn(64, dtype=torch.float64)
    with pytest.warns(RuntimeWarning, match="support-radius solve failed"):
        assert lightweight_model(U).spectral_matching(0) == 31


def test_nonconverged_and_out_of_range_roots_fall_back(monkeypatch):
    class Result:
        converged = False

    monkeypatch.setattr(wsindy_module, "reference_changepoint", lambda _: (8, 1, None))
    monkeypatch.setattr(
        wsindy_module.scipy.optimize, "brentq",
        lambda *args, **kwargs: (10.0, Result()),
    )
    U = torch.randn(64, dtype=torch.float64)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        assert lightweight_model(U).spectral_matching(0) == 31

    Result.converged = True
    monkeypatch.setattr(wsindy_module, "support_bracket", lambda *args: (2.0, 3.0))
    monkeypatch.setattr(wsindy_module, "F_root", lambda *args: 0.0)
    monkeypatch.setattr(
        wsindy_module.scipy.optimize, "brentq",
        lambda *args, **kwargs: (32.0, Result()),
    )
    with pytest.warns(RuntimeWarning, match="outside the valid range"):
        assert lightweight_model(U).spectral_matching(0) == 31
