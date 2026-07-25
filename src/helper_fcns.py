# HELPER FUNCTIONS
import ast
import torch
import scipy
import numpy as np
import itertools
import symengine as sp

import torch.linalg as la
from scipy.signal import convolve
from scipy.special import factorial
import matplotlib.pyplot as plt
from tqdm import tqdm

import warnings

# Reconstruct the half-spectrum used by the reference MATLAB changepoint
# routine from the nonnegative-frequency FFT of real-valued data.
def reference_half_spectrum(U, d):
  N = U.shape[d]
  U64 = U.detach().to(dtype=torch.float64)
  spectrum = torch.abs(torch.fft.rfft(U64, n=N, dim=d))
  mean_dims = tuple(dim for dim in range(spectrum.ndim) if dim != d)
  if mean_dims:
    spectrum = spectrum.mean(dim=mean_dims)

  # fftshift(fft(U)) starts at the negative Nyquist mode. For real data,
  # conjugate symmetry makes its retained negative-frequency half exactly the
  # following reversal of rfft magnitudes.
  if N % 2 == 0:
    return torch.flip(spectrum[1:], dims=(0,))  # -N/2, ..., -1
  return torch.flip(spectrum, dims=(0,))        # -(N-1)/2, ..., 0


# Direct O(N^2) implementation of the reference relative-error objective.
# This is used to verify or fall back from the optimized prefix-sum formula.
def _direct_reference_scores(H, candidates=None):
  N = H.numel()
  x = torch.arange(N, dtype=H.dtype, device=H.device)
  if candidates is None:
    candidates = torch.arange(1, N-1, dtype=torch.long, device=H.device)

  scores = []
  for candidate in candidates:
    j = int(candidate.item())
    slope1 = (H[j] - H[0]) / j
    line1 = H[0] + slope1*x[:j+1]

    slope2 = (H[-1] - H[j]) / (N-1-j)
    line2 = H[-1] + slope2*(x[j:] - (N-1))

    error1 = ((line1 - H[:j+1]) / H[:j+1])**2
    error2 = ((line2 - H[j:]) / H[j:])**2
    scores.append(error1.sum() + error2.sum())
  return torch.stack(scores)


# Evaluate the reference MATLAB changepoint objective in O(N) time using
# prefix sums. Returns (critical wavenumber, split index, scores).
def reference_changepoint(H):
  if H.ndim != 1 or H.numel() < 3:
    raise ValueError("Cumulative spectrum must be one-dimensional with at least three entries.")
  if not torch.isfinite(H).all().item() or not torch.all(H > 0).item():
    raise ValueError("Cumulative spectrum has undefined relative weights.")
  if not torch.all(H[1:] >= H[:-1]).item():
    raise ValueError("Cumulative spectrum must be nondecreasing.")

  H = H.to(dtype=torch.float64)
  N = H.numel()
  # The reference Fourier coordinates are an affine scaling of these indices;
  # endpoint-interpolating lines have identical relative residuals either way.
  x = torch.arange(N, dtype=H.dtype, device=H.device)
  candidates = torch.arange(1, N-1, dtype=torch.long, device=H.device)
  j = candidates.to(dtype=H.dtype)

  inv_H = 1/H
  inv_H2 = inv_H**2
  P0 = torch.cumsum(inv_H2, dim=0)
  P1 = torch.cumsum(x*inv_H2, dim=0)
  P2 = torch.cumsum((x**2)*inv_H2, dim=0)
  Q0 = torch.cumsum(inv_H, dim=0)
  Q1 = torch.cumsum(x*inv_H, dim=0)

  slope1 = (H[candidates] - H[0]) / j
  intercept1 = torch.full_like(slope1, H[0])
  slope2 = (H[-1] - H[candidates]) / (N-1-j)
  intercept2 = H[-1] - slope2*(N-1)

  def expanded_error(a, b, S0, S1, S2, T0, T1, count):
    terms = torch.stack((
      a**2*S0,
      2*a*b*S1,
      b**2*S2,
      -2*a*T0,
      -2*b*T1,
      count,
    ))
    return terms.sum(dim=0), torch.abs(terms).sum(dim=0)

  count1 = j + 1
  error1, magnitude1 = expanded_error(
    intercept1, slope1,
    P0[candidates], P1[candidates], P2[candidates],
    Q0[candidates], Q1[candidates], count1)

  previous = candidates - 1
  count2 = N - j
  error2, magnitude2 = expanded_error(
    intercept2, slope2,
    P0[-1]-P0[previous], P1[-1]-P1[previous], P2[-1]-P2[previous],
    Q0[-1]-Q0[previous], Q1[-1]-Q1[previous], count2)

  scores = error1 + error2
  eps = torch.finfo(H.dtype).eps
  roundoff_bound = 64*eps*(magnitude1 + magnitude2)

  # A materially negative or nonfinite expanded score indicates catastrophic
  # cancellation. Fall back to the direct reference computation in that case.
  inconsistent = (~torch.isfinite(scores)) | (scores < -roundoff_bound)
  if inconsistent.any().item():
    scores = _direct_reference_scores(H, candidates)
  else:
    scores = torch.clamp_min(scores, 0.)

    # Verify the provisional winner directly. If the expanded score is not
    # consistent with the direct residual, recompute every candidate.
    winner = int(torch.argmin(scores).item())
    direct_winner = _direct_reference_scores(H, candidates[winner:winner+1])[0]
    tolerance = 8*roundoff_bound[winner] + 64*eps*(1 + torch.abs(direct_winner))
    if (not torch.isfinite(direct_winner).item()
        or torch.abs(scores[winner] - direct_winner) > tolerance):
      scores = _direct_reference_scores(H, candidates)
    else:
      scores[winner] = direct_winner

      # Resolve candidates whose prefix-sum uncertainty overlaps the winning
      # score. This preserves the reference routine's first-minimum behavior
      # for exact or near ties without giving up O(N) work in the usual case.
      possible = torch.where(scores - roundoff_bound <= direct_winner + tolerance)[0]
      if possible.numel() > 1:
        scores[possible] = _direct_reference_scores(H, candidates[possible])

  if not torch.isfinite(scores).all().item():
    raise ValueError("No finite reference changepoint score was found.")

  winner = int(torch.argmin(scores).item())
  split_index = int(candidates[winner].item())
  critical_wavenumber = max(N - split_index - 3, 1)
  return critical_wavenumber, split_index, scores


def support_bracket(k, N, tau_hat, tau):
  lower = (np.sqrt(3)/np.pi) * (N/2) * (tau_hat/k)
  upper = lower * np.sqrt(1 - (8/np.sqrt(3))*np.log(tau))
  return lower,upper

# Function used in spectral matching
def F_root(m,k,N,tau_hat,tau):
  log_term = np.log((2*m-1)/m**2)
  mid_term = (2*np.pi*k*m)**2 - 3*(tau_hat*N)**2
  last_term = 2*(tau_hat*N)**2 * np.log(tau)
  return log_term * mid_term - last_term

def _spectral_fallback(self, d, reason, m_max):
    warnings.warn(f"spectral_matching (axis {d}): {reason}; using fallback "
                  f"support radius m={m_max}. Consider passing m explicitly.",
                  RuntimeWarning, stacklevel=2)
    return m_max

# Compute test function degrees given support radii
def compute_degrees(d, m, alpha, tau=1.e-10):
  alpha_bar = max(tuple(item[d] for item in alpha))
  log_tau_term = np.ceil(np.log(tau)/np.log((2*m-1)/m**2))
  p = int(max(log_tau_term, alpha_bar + 1))
  return p

# Indices of query points along a single axis
def subsample(s, m, x):
  if (2*m + 1) > x.shape[0]:
    raise ValueError('Error: m produces non-compact support.')
  return list(range(m, x.shape[0]-m, s))

# Return subsampled tensor to normal dimensions
def reshape_subsampled_tensor(Uk, mask, s, m, X):
  xk = [len(subsample(s[i], m[i], X[i])) for i in range(len(X))]
  return Uk.reshape(tuple(xk))

# Carefully compute n-choose-k with non-integer k
def my_nchoosek(n, k):
  n_factorial = scipy.special.factorial(n)
  k_factorial = scipy.special.factorial(np.ceil(k))
  nk_term = scipy.special.factorial(n-np.floor(k))
  return n_factorial / (nk_term * k_factorial)

# Calculate symbolic derivatives of test fcns (Dth derivative at degree p)
def D_phibar(x, D, x_sym, phi_bar):
  D_phi = sp.diff(phi_bar, x_sym, D)
  if abs(x) < 1.:
    return float(D_phi.subs(x_sym, x))
  else:
    return 0.

# FFT-base convolution with separable kernel
def separable_convolve(u, kernels, jacobian=1.):
  conv = jacobian * u.clone()
  for i, Ki in enumerate(kernels):
    shape = u.ndim * [1]
    shape[i] = -1
    conv = convolve(conv, Ki.reshape(shape), mode='valid')
  return conv

# Compute a weak polynomial term, <D_phi, u^power>
def compute_weak_poly(u, kernels, spacing, power=1., yu=1., yxyt=1., jacobian=1.):
  weak_poly = torch.from_numpy(separable_convolve((yu*u)**power, kernels, jacobian=jacobian))
  weak_poly *= yxyt * np.prod(spacing)
  return weak_poly

# Weak multivariable polynomial term, <D_phi, u1^p1 * ... * un^pn>
def compute_weak_multipoly(u, kernels, spacing, power=[1.], yu=[1.], yxyt=1., jacobian=1.):
  assert type(u) == type(power) == type(yu) == list, "Must provide a list."
  monomial = 1
  for i,ui in enumerate(u):
    monomial *= (yu[i]*ui)**power[i]
  weak_poly = torch.from_numpy(separable_convolve(monomial, kernels, jacobian=jacobian))
  weak_poly *= yxyt * np.prod(spacing)
  return weak_poly

# Compute a weak trig term, <D_phi, cos(freq*u + phase)>
def compute_weak_trig(u, kernels, spacing, freq=1., phase=0., yxyt=1., jacobian=1.):
  trig = torch.cos(freq*u + phase)
  weak_trig = torch.from_numpy(separable_convolve(trig, kernels, jacobian=jacobian))
  weak_trig *= yxyt * np.prod(spacing)
  return weak_trig

# Computes loss for a given candidate threshold
def loss(w_n, w_LS, G):
  LS_num = la.norm(G @ (w_n - w_LS)).item()
  LS_denom = la.norm(G @ w_LS).item()
  LS_term = LS_num / LS_denom
  zero_norm = sum(w_n != 0).item()/w_n.shape[0]
  loss_n = LS_term + zero_norm
  return loss_n

# Prints the symbolic ODE or PDE
def symbolic_eqn(lhs_name, rhs_names, w):
  nonzero_inds = w.nonzero().flatten()
  nonzero_coeffs = w[nonzero_inds].tolist()
  nonzero_terms = [rhs_names[i] for i in nonzero_inds]

  pde = []
  for coeff, term in zip(nonzero_coeffs, nonzero_terms):
    if coeff >= 0.:
      pde.append(f"+ {coeff:.2f}{term}")
    else:
      pde.append(f"- {abs(coeff):.2f}{term}")
  pde = lhs_name + " = " + " ".join(pde)
  return pde

# Returns residuals and R^2
def compute_residuals(G, w, b):
  r = b - G@w
  R2 = 1 - (r**2).sum() / ((b - b.mean())**2).sum()
  return r,R2

# Add noise
def add_noise(U, sigma_NR):
  U_rms = (torch.sqrt((U**2).mean())).item();
  sigma = sigma_NR * U_rms
  epsilon = torch.normal(mean=0, std=sigma, size=U.shape, dtype=torch.float64)
  return U + epsilon

# Add noise to a list of states
def add_noise_to_states(states, noise):
  if noise == 0:
    return states
  return [add_noise(ui, noise) for ui in states]

# Augmented libraries
def composite_term(columns, coeffs, name, model, lib_info):
  [G, powers, derivs, rhs_names] = lib_info
  if model.rescale:
    mu = model.compute_scale_matrix(powers, derivs)
  else:
    mu = torch.ones(len(powers))

  term = 0
  for i,column in enumerate(sorted(columns, reverse=True)):
    term += coeffs[i] * (1/mu[column]) * G[column]
    G.pop(column)
    powers.pop(column)
    derivs.pop(column)
    rhs_names.pop(column)

  G.append(term)
  powers.append(None)
  derivs.append(None)
  rhs_names.append(name)
  return G, powers, derivs, rhs_names

# Convert a SciPy ODE solution to a list of torch tensors
def convert_to_torch(sol):
  return [torch.tensor(ui, dtype=torch.float64) for ui in sol.y]

# Plot (u1,...,ud)(t) time series data for ODE systems
def plot_states_ode(t, states, names, title, ls='-', alpha=1):
  plt.figure(figsize=(8,3))
  for ui,name in zip(states,names):
    plt.plot(t, ui, ls, alpha=alpha, label='$' + name + '(t)$')
  plt.xlabel('$t$')
  plt.title(title)
  plt.grid(True, alpha=0.3, color='silver')
  plt.legend(loc='upper left')
  plt.show()

# For plotting hyperparameter sweeps
def parse_m_labels(m_labels):
  try:
    m_values = [ast.literal_eval(label) for label in m_labels]
  except (SyntaxError, ValueError):
    m_values = None
  if (m_values is not None) and all(len(set(np.atleast_1d(mi)))==1 for mi in m_values):
    m_ticks = np.array([np.atleast_1d(mi)[0] for mi in m_values])
    order = np.argsort(m_ticks)
    return m_ticks[order], None, order
  return np.arange(len(m_labels)), m_labels, np.arange(len(m_labels))

# For plotting hyperparameter sweeps
def sweep_grid(results, metric, rescale, m_labels, Lambdas, order):
  Z = np.full((len(m_labels), len(Lambdas)), np.nan)
  for i in range(len(results['Lambda'])):
    if results['rescale'][i] == rescale:
      Z[m_labels.index(results['m'][i]), Lambdas.index(results['Lambda'][i])] = results[metric][i]
  return Z[order]

# For plotting hyperparameter sweeps
def pad_ticks(ticks, positive=False):
  if len(ticks) > 1:
    left, right = 2*ticks[0]-ticks[1], 2*ticks[-1]-ticks[-2]
  else:
    step = ticks[0]/2 if positive else 1
    left, right = ticks[0]-step, ticks[0]+step
  if positive and left <= 0:
    left = ticks[0]/2
  return np.r_[left, ticks, right]