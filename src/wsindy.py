from helper_fcns import *
import csv
import warnings
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D

class WSINDy:
  def __init__(self, U, alpha, beta, X, V=[], names=None, m=None, p=None, s=None, jacobian = 1.,
               tau=1e-10, tau_hat=2, verbosity=True, rescale=True, eqn_type='pde'):
    self.U = U # state variable
    self.V = V # auxiliary variables
    self.alpha = alpha # derivative multi-indices
    self.beta = beta # monomial multi-indices
    self.X = X if type(X) == list else [X] # coordinate axes
    self.eqn_type = eqn_type # either 'ode' or 'pde'

    self.jacobian = jacobian # dx*...*dt
    self.tau = tau # test function support tolerance
    self.tau_hat = tau_hat # for spectral matching
    self.verbosity = verbosity # prints plots and info
    self.rescale = rescale # scale-invariant preconditioning

    if eqn_type not in ('ode', 'pde'):
      raise ValueError("eqn_type must be 'ode' or 'pde'.")
    if eqn_type == 'ode':
      if self.U.dim() != 1:
        raise ValueError("ODE data must be one-dimensional in time.")
      if len(self.X) != 1:
        raise ValueError("ODE data requires a single time axis.")
    elif self.U.dim() < 2:
      raise ValueError("PDE data requires at least one spatial axis.")

    self.beta_max = max([bj[0] for bj in beta])
    self.dX = [Xi.diff()[0] for Xi in self.X]

    self.names = names if names is not None else ['u']+['v'+str(i+1) for i in range(len(V))]
    self.m = m if m is not None else self.compute_m() # test function support radii
    self.p = p if p is not None else self.compute_p() # test function degrees
    self.s = s if s is not None else [U.shape[i]//100 for i in range(U.ndim)] # subsampling rates
    
    if rescale:
      [self.yx, self.yt] = self.compute_spatial_scales()
      self.yu = self.compute_u_scale(self.U, self.beta_max)
      self.aux_scales = []
      for i,Vi in enumerate(V):
        beta_max = max([bj[i+1] for bj in self.beta])
        self.aux_scales.append(self.compute_u_scale(Vi, beta_max))
    else:
      [self.yx, self.yt, self.yu, self.aux_scales] = 4*[None]

    [self.mask, self.flat_mask] = self.compute_query_points()
    [self.axes, self.kernels] = self.build_axes()
    self.test_fcns = self.build_test_fcns()
    self.derivative_names = self.get_derivative_names()
  
  ##########################################################################################################
  # AUTOMATIC HYPERPARAMETER SELECTION
  ##########################################################################################################

  def compute_m(self):
    return [self.spectral_matching(d) for d in range(self.U.dim())]

  def compute_p(self):
    p = [compute_degrees(d, md, self.alpha, tau=self.tau) for d,md in enumerate(self.m)]
    return p

  def _validate_spectral_inputs(self, d):
    if not (0 < self.tau < 1):
      raise ValueError("tau must lie strictly between 0 and 1.")
    if not np.isfinite(self.tau_hat) or self.tau_hat <= 0:
      raise ValueError("tau_hat must be positive and finite.")
    if torch.is_complex(self.U):
      raise ValueError("Spectral matching requires real-valued data.")
    if not torch.isfinite(self.U).all().item():
      raise ValueError("Spectral matching requires finite data.")

    Nd = self.U.shape[d]
    if Nd < 5:
      raise ValueError("Spectral matching requires at least five samples per axis.")
    if len(self.X) != self.U.dim():
      raise ValueError("The number of coordinate axes must match the data dimensions.")

    axis = self.X[d]
    if not isinstance(axis, torch.Tensor):
      axis = torch.as_tensor(axis)
    if axis.ndim != 1 or axis.numel() != Nd:
      raise ValueError(f"Coordinate axis {d} must be one-dimensional with {Nd} entries.")
    if torch.is_complex(axis) or not torch.isfinite(axis).all().item():
      raise ValueError(f"Coordinate axis {d} must be real and finite.")

    axis64 = axis.detach().to(dtype=torch.float64)
    spacing = torch.diff(axis64)
    if not torch.all(spacing > 0).item():
      raise ValueError(f"Coordinate axis {d} must be strictly increasing.")
    atol = 1e-12*max(abs(spacing[0].item()), 1.)
    if not torch.allclose(spacing, torch.full_like(spacing, spacing[0]),
                          rtol=1e-5, atol=atol):
      raise ValueError(f"Coordinate axis {d} must be uniformly spaced for FFT matching.")
    return Nd

  def _spectral_fallback(self, d, reason, m_max):
    warnings.warn(f"spectral_matching (axis {d}): {reason}; using fallback "
                  f"support radius m={m_max}. Consider passing m explicitly.",
                  RuntimeWarning, stacklevel=2)
    return m_max

  # Determine a test-function support radius by matching its Fourier decay to
  # the signal/noise changepoint, following Appendix A and findcorners.m.
  def spectral_matching(self, d):
    Nd = self._validate_spectral_inputs(d)
    m_max = (Nd-1) // 2

    spectrum = reference_half_spectrum(self.U, d)
    if (not torch.isfinite(spectrum).all().item()
        or spectrum.numel() < 3
        or spectrum.sum().item() <= 0):
      return self._spectral_fallback(d, "zero or degenerate spectrum", m_max)

    Hd = torch.cumsum(spectrum, dim=0)
    if not torch.all(Hd > 0).item():
      return self._spectral_fallback(d, "undefined relative spectral weights", m_max)

    try:
      kc, _, _ = reference_changepoint(Hd)
    except (ValueError, RuntimeError, FloatingPointError) as error:
      return self._spectral_fallback(d, f"failed changepoint fit ({error})", m_max)

    existence_value = (np.sqrt(3)/np.pi) * (self.tau_hat/kc)
    if not ((4/Nd) <= existence_value <= 1):
      return self._spectral_fallback(
        d, "the Appendix A root-existence condition is not satisfied", m_max)

    lower,upper = support_bracket(kc, Nd, self.tau_hat, self.tau)
    F_lower = F_root(lower, kc, Nd, self.tau_hat, self.tau)
    F_upper = F_root(upper, kc, Nd, self.tau_hat, self.tau)
    if (not np.isfinite(F_lower) or not np.isfinite(F_upper)
        or F_lower < 0 or F_upper > 0):
      return self._spectral_fallback(d, "support-radius bracket does not change sign", m_max)

    try:
      md_fit,result = scipy.optimize.brentq(
        F_root, lower, upper, args=(kc,Nd,self.tau_hat,self.tau),
        full_output=True, disp=False)
    except (ValueError, RuntimeError, OverflowError, FloatingPointError) as error:
      return self._spectral_fallback(d, f"support-radius solve failed ({error})", m_max)

    scale = max(abs(F_lower), abs(F_upper), 1.)
    residual = abs(F_root(md_fit, kc, Nd, self.tau_hat, self.tau))
    if not result.converged or not np.isfinite(md_fit) or residual > 1e-8*scale:
      return self._spectral_fallback(d, "support-radius solve did not converge", m_max)

    md = int(np.ceil(md_fit))
    if not (2 <= md <= m_max):
      return self._spectral_fallback(d, "support-radius root is outside the valid range", m_max)

    if self.verbosity:
      if Nd % 2 == 0:
        modes = torch.arange(Nd//2, 0, -1)
      else:
        modes = torch.arange(Nd//2, -1, -1)
      modes = modes.numpy()
      spectrum_np = spectrum.detach().cpu().numpy()
      label = r'$|\mathcal{F}[u]|(k)$' if self.eqn_type == 'ode' else r'$|\mathcal{F}_{x_d}[u]|(k)$'

      plt.figure(figsize=(7,2))
      plt.plot(modes, spectrum_np, '.-')
      plt.axvline(kc, ls='--', color='r', label=r'Estimated changepoint, $\hat{k}_d$')
      plt.xlabel('Wavenumber, $k$')
      plt.ylabel(label)
      plt.title(fr'Spectral matching: $\hat{{m}}_d={md}$ (axis: $d=${d})')
      plt.legend(loc='upper right', framealpha=0.8, fontsize=11)
      plt.yscale('log')
      plt.grid(True, alpha=0.3, color='silver')
      plt.show()
    return md
  
  # # Determines support radius md in [mx,...,mt] of test fcn by estimating signal-dominated Fourier modes
  # def spectral_matching(self, d, changepoint, F_root):
  #   Nd = self.U.shape[d]
  #   Uhat_d = abs(torch.fft.rfft(self.U, n=Nd, dim=d))
  #   dims = [dim for dim in range(Uhat_d.ndimension()) if dim != d]
  #   Uhat_d = Uhat_d.mean(dim = dims) if dims else Uhat_d

  #   Hd = torch.cumsum(Uhat_d, dim=0) # Cumulative sum
  #   Hd = (Hd/Hd.max()).numpy() # Normalize for curve fitting

  #   # Estimate wavenumber kc at which spectrum becomes noise-dominated
  #   freqs = torch.arange(0, np.floor(Nd/2)+1, 1).numpy()
  #   params = scipy.optimize.curve_fit(changepoint, freqs, Hd, p0=self.init_kc_guess)[0]
  #   kc = int(params[0]) if np.isfinite(params[0]) else 0
  #   degenerate = kc < 1 # Flags a non-physical wavenumber
  #   kc = max(kc, 1)

  #   # Solve root-finding problem
  #   guess = (np.sqrt(3)*Nd*self.tau_hat)/(2*np.pi*kc)
  #   guess = guess * (1 + np.sqrt(1 - (8/np.sqrt(3))*np.log(self.tau)))/2
  #   md_fit = scipy.optimize.root(F_root, guess, args=(kc,Nd,self.tau_hat,self.tau)).x[0]
  #   md = int(md_fit) if np.isfinite(md_fit) else None

  #   # Clamp support radius to the valid range
  #   m_max = (Nd-1) // 2
  #   if md is None or not (2 <= md <= m_max):
  #   md = int(np.clip(md_fit, 2, m_max)) if np.isfinite(md_fit) else m_max
  #   degenerate = True
  #   if degenerate:
  #   warnings.warn(f"spectral_matching (axis {d}): degenerate change-point fit; "
  #   f"using fallback support radius m={md}. Consider passing m explicitly.")

  #   if self.verbosity:
  #   plt.figure(figsize=(7,2))
  #   Uhat_d = Uhat_d.numpy()
  #   label = r'$|\mathcal{F}[u]|(k)$' if self.eqn_type == 'ode' else r'$|\mathcal{F}_{x_d}[u]|(k)$'
  #   plt.plot(freqs, Uhat_d, '.-')
  #   plt.axvline(kc, ls='--', color='r', label=r'Estimated changepoint, $\hat{k}_d$')
  #   plt.xlabel('Wavenumber, $k$')
  #   plt.ylabel(label)
  #   plt.title(fr'Spectral matching: $\hat{{m}}_d={md}$ (axis: $d=${d})')
  #   plt.legend(loc = 'upper right', framealpha=0.8, fontsize=11)
  #   plt.yscale('log')
  #   plt.grid(True, alpha=0.3, color='silver')
  #   plt.show()
  #   return md

  # Compute mask of query point indices, U[mask] = U[xk,...,tk]
  def compute_query_points(self):
    subsamples = [subsample(self.s[i], self.m[i], self.X[i]) for i in range(self.U.ndim)]
    cartesian_prod = itertools.product(*subsamples)
    mask = tuple(map(torch.tensor, zip(*cartesian_prod))) # for tensors
    flat_mask = tuple([mask[i]-self.m[i] for i in range(self.U.ndim)]) # flattens tensors

    if self.verbosity:
      if self.eqn_type == 'ode':
        t = self.X[0]
        tk = t[subsamples[0]]

        plt.figure(figsize=(7,2))
        plt.plot(t, self.U, '.-', zorder=1)
        plt.scatter(tk, self.U[mask], color='r', marker='.', s=14, label='Query points', zorder=2)
        plt.xlabel('$t$')
        plt.ylabel('$' + self.names[0] + '(t)$')
        plt.legend(loc='best', framealpha=0.8)
        plt.grid(True, alpha=0.3, color='silver')
        plt.show()
      else:
        [x1, x2] = [self.X[0], self.X[1]]
        [X1, X2] = np.meshgrid(self.X[0], self.X[1])
        [xk1, xk2] = [x1[subsamples[0]], x2[subsamples[1]]]
        [XK1, XK2] = [Xi.flatten() for Xi in np.meshgrid(xk1, xk2)]
        slice0 = (slice(None),slice(None)) + (self.U.dim()-2)*(0,)

        plt.figure(figsize=(6,6))
        plt.pcolormesh(X1, X2, self.U[slice0].T, cmap='coolwarm')
        plt.scatter(XK1, XK2, color='black', marker='.', s=2, label='Query Points')
        labels = ['x', 't'] if self.U.ndim==2 else ['x_1', 'x_2', ', \\dots']
        plt.xlabel('$' + labels[0] + '$')
        plt.ylabel('$' + labels[1] + '$')
        plt.title('$' + self.names[0] + '(' + labels[0] + ',' + labels[1:] + ')$')
        plt.legend(loc='upper right')
        plt.show()
    return mask, flat_mask
  
  ##########################################################################################################
  # SCALE-INVARIANT PRECONDITIONING
  ##########################################################################################################

  # Compute scale for a state variable, yu
  def compute_u_scale(self, u, beta_max):
    U_2 = la.norm(u).item()
    U_beta = la.norm(u**beta_max).item()
    yu = (U_2 / U_beta)**(1 / max(1,beta_max))
    return yu

  # Compute spatio-temporal scales yx, yt (yx = [] for ODEs)
  def compute_spatial_scales(self):
    D = len(self.dX) - 1
    max_x = []
    for d in range(len(self.alpha[0]) - 1):
      max_d = max(tuple(ai[d] for ai in self.alpha))
      max_x.append(max(1,max_d))
    max_t = max(1,max(tuple(ai[-1] for ai in self.alpha)))

    # Ansatz given in the paper
    yx = [(1/(self.m[d]*self.dX[d]) * (my_nchoosek(self.p[d], max_x[d]/2)
          * factorial(max_x[d]))**(1/max_x[d])).item() for d in range(D)]
    yt = (1 / (self.m[-1] * self.dX[-1]) * (my_nchoosek(self.p[-1], max_t/2)
          * factorial(max_t))**(1/max_t)).item()
    return yx,yt

  # Compute scale matrix diagonal, M = diag(mu)
  def compute_scale_matrix(self, powers, derivs):
    D = self.U.dim() - 1
    [yx, yt] = [self.yx, self.yt]
    yu = [self.yu] + self.aux_scales

    num_terms = len(powers)
    mu = torch.zeros(num_terms, dtype=torch.float64)
    for j in range(num_terms):
      aj = derivs[j]
      bj = powers[j]
      if (aj==None) or (bj==None):
        mu[j] = 1.
      else:
        yx_exps = [yx[d]**(self.alpha[0][d] - self.alpha[aj][d]) for d in range(D)]
        yx_term = np.prod(yx_exps)
        yt_term = yt**(self.alpha[0][-1] - self.alpha[aj][-1])

        yu_term = [yu[n]**(bj[n]) for n in range(len(yu))]
        yu_term = np.prod(yu_term)

        mu[j] = (yu_term * yx_term * yt_term)/self.yu
    return mu

  ##########################################################################################################
  # TEST FUNCTION CONSTRUCTION
  ##########################################################################################################

  # Compute test function and its derivatives along d-th axis
  def get_weight_fcns(self, d):
    [x,m,p] = [self.X[d], self.m[d], self.p[d]]
    dx = (x[1] - x[0]).item()

    if (m > (len(x)-1)/2) or (m <= 1):
      raise ValueError('Error: invalid test function support.')

    if self.rescale:
      scale_d = (self.yx + [self.yt])[d]
      dx *= scale_d

    # Initialize grid of discretized test fcn values
    test_fcns_d = torch.zeros(len(self.alpha), 2*m+1, dtype=torch.float64)
    n_grid = torch.linspace(-1, 1, 2*m+1, dtype=torch.float64)
    multi_index_d = tuple(ai[d] for ai in self.alpha)

    # Compute symbolic derivatives
    x_sym = sp.Symbol('x')
    phi_bar = (1 - x_sym**2)**p
    vec = np.vectorize(D_phibar)

    for i in range(len(self.alpha)):
      if (i > 0) and (multi_index_d[i-1] == multi_index_d[i]):
        test_fcns_d[i,:] += test_fcns_d[i-1,:]
      else:
        num_derivs = multi_index_d[i]
        A_i = torch.from_numpy(vec(n_grid, num_derivs, x_sym, phi_bar))
        test_fcns_d[i,:] += (1/((m*dx)**num_derivs)) * A_i

    if self.verbosity:
      plt.figure(figsize=(7,2))
      for i in range(len(test_fcns_d[:,0])):
        plt.plot(m*dx*n_grid, test_fcns_d[i,:], '--.', label=f'$i={i}$')
      plt.title(f'Test function components (axis: $d={d}$)')
      if self.eqn_type == 'ode':
        plt.xlabel('$t_k - t$')
        plt.ylabel(r'$\left(\frac{d}{dt}\right)^{{\alpha_i}}\phi(t_k - t)$')
      else:
        plt.xlabel('$x_k - x$')
        plt.ylabel(fr'$D^{{\alpha_i}}\phi(x_k - x)$')
      plt.grid(True, alpha=0.3, color='silver')
      plt.legend(loc='upper right', framealpha=0.8)
      plt.show()
    return test_fcns_d

  # Build lists of vectorized test function components
  def build_axes(self):
    #axes = [self.get_weight_fcns(d) for d in range(len(self.m))]
    axes = [self.get_weight_fcns(d) for d in range(self.U.ndim)]
    kernels = [[axes[d][i,:] for d in range(self.U.ndim)] for i in range(len(self.alpha))]
    return axes, kernels

  # Build tensorized test functions (Nx x ... x Nt)
  def build_test_fcns(self):
    num_derivs = len(self.axes[0])
    D = len(self.axes)
    test_fcns = []
    for s in range(num_derivs):
      axes = [self.axes[d][s,:] for d in range(D)]
      ijk = [chr(105 + i) for i in range(D)]
      einsum = ','.join(ijk) + '->' + ''.join(ijk)
      D_phi = torch.einsum(einsum, *axes)
      test_fcns.append(D_phi)
    return test_fcns
  
  ##########################################################################################################
  # BUILD WSINDY LINEAR SYSTEM (Gw = b)
  ##########################################################################################################

  # Weak time derivative
  def build_lhs(self, lhs_name):
    kernel = self.kernels[0]
    if self.rescale:
      yxyt = np.prod(self.yx + [self.yt])
      lhs = compute_weak_poly(self.U, kernel, self.dX, yu=self.yu, yxyt=yxyt, jacobian=self.jacobian)
    else:
      lhs = compute_weak_poly(self.U, kernel, self.dX, jacobian=self.jacobian)
    b = lhs[self.flat_mask]
    self.lhs_name = lhs_name
    self.lhs = b
    return

  # Computes default monomial library terms
  def create_default_library(self):
    [G, powers, derivs, rhs_names] = [],[],[],[]
    state = [self.U] + self.V
    if self.rescale:
      yu = [self.yu] + self.aux_scales
      yxyt = np.prod(self.yx + [self.yt])
    else:
      yu = len(state) * [1.]
      yxyt = 1.

    for i,ai in enumerate(tqdm(self.alpha[1:], disable=not self.verbosity), start=1):
      kernel = self.kernels[i]
      for j,bj in enumerate(self.beta):
        if all(bjd==0 for bjd in bj) and any(aid!=0 for aid in ai):
          continue # Avoid derivatives of constant terms
        else:
          assert (len(bj)-1) == len(self.V), "Inconsistent number of powers!"
          derivs.append(i)
          powers.append(bj)
          term = compute_weak_multipoly(state, kernel, self.dX, power=bj, yu=yu, yxyt=yxyt, jacobian=self.jacobian)
          name = self.format_monomial(bj) + self.derivative_names[i]
          G.append(term[self.flat_mask])
          rhs_names.append(name)
    return G, powers, derivs, rhs_names
  
  def set_library(self, G, powers, derivs, rhs_names):
    G = torch.stack(G, dim=1)
    if G.shape[0] != len(self.mask[0]):
      raise ValueError("Library has inconsistent dimensions.")
    self.rhs_names = rhs_names
    self.library = G
    if self.rescale:
      self.mu = self.compute_scale_matrix(powers, derivs)
    return
  
  ##########################################################################################################
  # SPARSE REGRESSION
  ##########################################################################################################

  # Full MSTLS optimization routine, scans through Lambdas
  def MSTLS(self, Lambda=None, Lambdas=10**((3/99)*torch.arange(0,100)-3)):
  #def MSTLS(self, Lambda=None, Lambdas=10**((4/49)*torch.arange(0,50)-4)):
    w_LS = la.lstsq(self.library, self.lhs, driver='gelsd').solution

    if Lambda is not None:
      Lambda_star = Lambda
    else:
      loss_history = []
      for Lambda_n in Lambdas:
        [_, loss_n] = self.MSTLS_iterate(Lambda_n.item(), w_LS.clone())
        loss_history.append(loss_n)

      # Find minimizer (smallest minimzer, if not unique)
      index = loss_history.index(min(loss_history))
      Lambda_star = Lambdas[index].item()

    [w_star, loss_star] = self.MSTLS_iterate(Lambda_star, w_LS.clone())
    if self.rescale:
      w_star = self.mu * w_star
    self.Lambda = Lambda_star
    self.loss = loss_star
    self.coeffs = w_star
    return w_star

  # Modified sequential thresholding least squares (MSTLS) routine
  def MSTLS_iterate(self, Lambda_n, w_LS):
    G = self.library
    b = self.lhs
    max_its = G.shape[1]

    bounds = la.norm(b) / la.norm(G,dim=0)
    lower = Lambda_n * torch.maximum(bounds, torch.ones(bounds.shape[0]))
    upper = (1/Lambda_n) * torch.minimum(bounds, torch.ones(bounds.shape[0]))

    iteration = 0
    w_n = w_LS.clone()
    nonzero_inds = torch.tensor([])
    while iteration <= max_its:
      ib_inds = torch.where((abs(w_n) >= lower) & (abs(w_n) <= upper))[0]
      oob_inds = torch.where((abs(w_n) < lower) | (abs(w_n) > upper))[0]

      if (torch.equal(nonzero_inds, ib_inds)) or (ib_inds.shape[0]==0):
        break
      w_n[ib_inds] = la.lstsq(G[:,ib_inds], b, driver='gelsd').solution
      w_n[oob_inds] = 0
      nonzero_inds = ib_inds
      iteration += 1

    loss_n = loss(w_n, w_LS, G)
    return w_n, loss_n
  
  ##########################################################################################################
  # OUTPUT AND FORMATTING
  ##########################################################################################################
  
  # Returns symbolic derivatives
  def get_derivative_names(self):
    D = self.U.dim() - 1
    derivative_names = []
    for elem in self.alpha:
      if all(value == 0 for value in elem):
        derivative_names.append('')
      else:
        # (0+1)-D, (1+1)-D, (2+1)-D, (3+1)-D case-handling
        if D == 0:
          derivative_names.append('_{'+'t'*elem[0]+'}') # for ODEs
        elif D == 1:
          derivative_names.append('_{'+'t'*elem[1]+'x'*elem[0]+'}')
        elif D == 2:
          derivative_names.append('_{'+'t'*elem[2]+'x'*elem[0]+'y'*elem[1]+'}')
        elif D == 3:
          derivative_names.append('_{'+'t'*elem[3]+'x'*elem[0]+'y'*elem[1]+'z'*elem[2]+'}')
        else:
          raise ValueError("Whoah! Spatial dimension can only be: 0, 1, 2, or 3.")
    return derivative_names

  # Fancy monomial formatting
  def format_monomial(self, bj):
    terms = []
    for d in range(len(bj)):
      if bj[d] == 0:
        continue
      if bj[d] == 1:
        terms.append(self.names[d])
      else:
        terms.append(f'{self.names[d]}^{bj[d]}')
    return '(' + ' '.join(terms) + ')' if terms else '(1)'

  # Prints a report of the WSINDy run
  def print_report(self):
    print('\nHYPER-PARAMETERS')
    print(f'm = {self.m}')
    print(f'p = {self.p}')
    print(f's = {self.s}')
    if self.rescale:
      if self.eqn_type == 'ode':
        print(f'yt = {self.yt:.3f}')
      else:
        scales = self.yx + [self.yt]
        print('[yx, yt] = ' + str([float(f" {yi:.3f}") for yi in scales]))
      print(f'yu = {self.yu:.3f}')
      aux_scales = [float(f'{yu:.3f}') for yu in self.aux_scales]
      print(f'Aux. scales = {aux_scales}\n')
    else:
      print('Not rescaled.\n')

    print('LIBRARY')
    print(f'Number of query points = {self.library.shape[0]}')
    print(f'Number of candidate terms = {self.library.shape[1]}')
    print(f'Condition number = {la.cond(self.library):.2e}\n')

    print('RESULTS')
    eq_label = self.eqn_type.upper()
    eqn = symbolic_eqn(self.lhs_name, self.rhs_names, self.coeffs)
    num_terms = self.coeffs.count_nonzero().item()
    if self.rescale:
      [r, R2] = compute_residuals(self.library, self.coeffs/self.mu, self.lhs)
    else:
      [r, R2] = compute_residuals(self.library, self.coeffs, self.lhs)
    print(f'{eq_label}: {eqn}')
    print(f'Nonzero terms = {num_terms}')
    print(f'Relative L2 error = {la.norm(r)/la.norm(self.lhs):.3f}')
    print(f'R^2 = {R2:.3f}')
    print(f'Lambda = {self.Lambda:.2e}')
    print(f'Loss = {self.loss:.3f}')
    return
  
  ##########################################################################################################
  # HYPERPARAMETER SWEEP
  ##########################################################################################################

  # Sweep hyperparameters (m, Lambda) with rescale = True/False
  def hyperparameter_sweep(self, lhs_name=None, m_values=None, Lambdas=None, rescales=(True, False),
                           library_fcn=None, true_coeffs=None, csv_path=None, plot=True, noise=None):
    if lhs_name is None:
      lhs_name = getattr(self, 'lhs_name', self.names[0] + self.derivative_names[0])
    if m_values is None:
      m_values = []
      #for factor in np.linspace(0.5, 1.5, 40):
      for factor in np.linspace(0.5, 1.5, 60):
        m_new = [int(round(factor*mi)) for mi in self.m]
        m_new = [min(max(mi,2), (self.U.shape[i]-1)//2) for i,mi in enumerate(m_new)]
        if m_new not in m_values:
          m_values.append(m_new)
    else:
      m_values = [(self.U.ndim*[mi] if type(mi)==int else list(mi)) for mi in m_values]
    if Lambdas is None:
      #Lambdas = 10**((4/49)*torch.arange(0,50)-4)
      #Lambdas = 10**((4/199)*torch.arange(0,200)-4)
      Lambdas = 10**((3/199)*torch.arange(0,200)-3)
    Lambdas = [float(Li) for Li in Lambdas]

    keys = ['m','p','Lambda','rescale','model','terms','coeffs','sparsity','loss','R2','L2','rel_L2','cond_G','noise']
    if true_coeffs is not None:
      keys.extend(['coeff_error','support_error','rel_sparsity'])
    results = {key: [] for key in keys}

    s = self.U.ndim * [1]
    for m in tqdm(m_values):
      for rescale in rescales:
        model = WSINDy(self.U, self.alpha, self.beta, self.X, V=self.V, names=self.names, m=m, s=s, jacobian=self.jacobian, tau=self.tau,
                       tau_hat=self.tau_hat, verbosity=False, rescale=rescale, eqn_type=self.eqn_type)
        if library_fcn is None:
          [G, powers, derivs, rhs_names] = model.create_default_library()
        else:
          [G, powers, derivs, rhs_names] = library_fcn(model)
        model.build_lhs(lhs_name)
        model.set_library(G, powers, derivs, rhs_names)
        cond_G = la.cond(model.library).item()

        for Lambda in Lambdas:
          w = model.MSTLS(Lambda=Lambda)
          w_tilde = w/model.mu if rescale else w
          [r, R2] = compute_residuals(model.library, w_tilde, model.lhs)
          nonzero = w.nonzero().flatten().tolist()

          results['m'].append(str(m))
          results['p'].append(str(model.p))
          results['Lambda'].append(Lambda)
          results['rescale'].append(rescale)
          results['model'].append(symbolic_eqn(lhs_name, rhs_names, w))
          results['terms'].append([rhs_names[j] for j in nonzero])
          results['coeffs'].append([w[j].item() for j in nonzero])
          results['sparsity'].append(len(nonzero))
          results['loss'].append(model.loss)
          results['R2'].append(R2.item())
          results['L2'].append(la.norm(r).item())
          results['rel_L2'].append((la.norm(r)/la.norm(model.lhs)).item())
          results['cond_G'].append(cond_G)

          if noise is not None:
            results['noise'].append(noise)
          else:
            results['noise'].append('unknown')

          # Max relative coefficient error
          if true_coeffs is not None:
            learned = dict(zip(results['terms'][-1], results['coeffs'][-1]))
            error = max(abs(learned.get(name,0.) - wj)/abs(wj) for name,wj in true_coeffs.items())
            results['coeff_error'].append(error)
            true_support = {name for name,wj in true_coeffs.items() if wj != 0}
            results['support_error'].append(len(set(learned) ^ true_support))
            results['rel_sparsity'].append(len(nonzero) - len(true_support))

    if csv_path is not None:
      with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(keys)
        for i in range(len(results['Lambda'])):
          writer.writerow([results[key][i] for key in keys])
    if plot:
      self.plot_sweep(results, noise=noise)
    return results

  # Heatmaps of hyperparameter sweep results over (m, Lambda)
  def plot_sweep(self, results, metrics=None, noise=None):
    if metrics is None:
      metrics = ('loss','L2','sparsity')
      metrics += tuple(metric for metric in ('coeff_error','support_error','rel_sparsity') if metric in results)

    m_labels = list(dict.fromkeys(results['m']))
    Lambdas = list(dict.fromkeys(results['Lambda']))
    rescales = list(dict.fromkeys(results['rescale']))
    [m_ticks, tick_labels, order] = parse_m_labels(m_labels)
    titles = {'loss': r'MSTLS loss, $\log_{10}\mathcal{L}(\mathbf{w}^{\lambda})$',
              'R2': r'Equation fit, $R^2$',
              'L2': r'$L^2$ error, $\log_{10}\|\mathbf{Gw}^{\lambda}-\mathbf{b}\|_2$',
              'sparsity': r'Nonzero terms, $\|\mathbf{w}^{\lambda}\|_0$',
              'coeff_error': r'Coeff. error, $\log_{10}E_{\infty}$',
              'support_error': 'Support error (# indices)',
              'rel_sparsity': r'Relative sparsity, $\|\mathbf{w}^{\lambda}\|_0-\|\mathbf{w}^{\star}\|_0$'}

    for rescale in rescales:
      nrows = 2 if len(metrics) > 4 else 1
      ncols = int(np.ceil(len(metrics)/nrows))
      fig, ax = plt.subplots(nrows, ncols, figsize=(4*ncols,4*nrows), sharex=True, sharey=True)
      ax = np.atleast_1d(ax).flatten()

      support_Z = sweep_grid(results, 'support_error', rescale, m_labels, Lambdas, order) \
                  if 'support_error' in results else None
      correct = support_Z == 0 if support_Z is not None else None
      coeff_Z = sweep_grid(results, 'coeff_error', rescale, m_labels, Lambdas, order) \
                if 'coeff_error' in results else None
      finite_coeffs = np.isfinite(coeff_Z) & correct if coeff_Z is not None and correct is not None else None
      best = None
      if finite_coeffs is not None and finite_coeffs.any():
        best = np.unravel_index(np.nanargmin(np.where(finite_coeffs, coeff_Z, np.nan)), coeff_Z.shape)

      for j,metric in enumerate(metrics):
        Z = sweep_grid(results, metric, rescale, m_labels, Lambdas, order)
        cbar_ticks = None

        if metric == 'loss':
          Z = np.log10(np.maximum(Z, 1e-16))
          pcm = ax[j].pcolormesh(Lambdas, m_ticks, Z, cmap='coolwarm')
        elif metric == 'R2':
          pcm = ax[j].pcolormesh(Lambdas, m_ticks, Z, cmap='Reds_r', vmin=0, vmax=1)
        elif metric == 'L2':
          Z = np.log10(np.maximum(Z, 1e-16))
          pcm = ax[j].pcolormesh(Lambdas, m_ticks, Z, cmap='coolwarm')
        elif metric in ('sparsity','support_error'):
          vmax = int(np.nanmax(Z))
          cbar_ticks = np.arange(vmax + 1)
          bounds = np.arange(-0.5, vmax + 1.5)
          norm = BoundaryNorm(bounds, plt.get_cmap('bone_r').N)
          pcm = ax[j].pcolormesh(Lambdas, m_ticks, Z, cmap='bone_r', norm=norm)
        elif metric == 'rel_sparsity':
          vmax = int(np.nanmax(np.abs(Z)))
          cbar_ticks = np.arange(-vmax, vmax + 1)
          bounds = np.arange(-vmax - 0.5, vmax + 1.5)
          norm = BoundaryNorm(bounds, plt.get_cmap('coolwarm').N)
          pcm = ax[j].pcolormesh(Lambdas, m_ticks, Z, cmap='PiYG', norm=norm)
        elif metric == 'coeff_error':
          Z = np.log10(np.maximum(Z, 1e-16))
          pcm = ax[j].pcolormesh(Lambdas, m_ticks, Z, cmap='coolwarm')
        else:
          pcm = ax[j].pcolormesh(Lambdas, m_ticks, Z, cmap='Reds')

        if correct is not None and correct.any():
          ax[j].contour(pad_ticks(Lambdas, positive=True), pad_ticks(m_ticks),
                        np.pad(correct.astype(float), 1), levels=[0.5],
                        colors='darkgoldenrod', linestyles='--', linewidths=1, zorder=3)
        if best is not None:
          ax[j].plot(Lambdas[best[1]], m_ticks[best[0]], marker='*', color='darkgoldenrod',
                     markersize=7, linestyle='none', zorder=4)

        ax[j].set_xscale('log')
        if tick_labels is not None:
          ax[j].set_yticks(m_ticks)
          ax[j].set_yticklabels(tick_labels, fontsize=7)
        ax[j].set_xlabel(r'$\lambda$')
        ax[j].set_ylabel('$m$')
        ax[j].set_title(titles.get(metric, metric))
        fig.colorbar(pcm, ax=ax[j], ticks=cbar_ticks)
      for extra_ax in ax[len(metrics):]:
        extra_ax.set_visible(False)
      fig.suptitle(f'Rescaled: {rescale}, Noise: {noise}')
      legend_handles = ([Line2D([0],[0], color='darkgoldenrod', linestyle='--', label='Correct support')]
                        if support_Z is not None else [])
      if best is not None:
        legend_handles.append(Line2D([0],[0], color='darkgoldenrod', marker='*', linestyle='none',
                                     markersize=7, label='Min. coeff. error (in support)'))
      if legend_handles:
        fig.legend(handles=legend_handles, loc='upper right', ncol=len(legend_handles))
      fig.tight_layout(rect=(0,0,1,0.9) if legend_handles else None)
      plt.show()
    return
