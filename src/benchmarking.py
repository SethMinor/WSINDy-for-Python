# WSINDy validation testing using systems from the 'dysts' library
#
# Automated ODE benchmarking: for each dysts system we extract the exact
# polynomial right-hand side (ground truth), generate a trajectory, run WSINDy
# equation-by-equation, and score recovery of the true terms/coefficients.
#
# v1 scope: polynomial, autonomous systems. The eligibility check returns a
# status flag ('polynomial'/'nonpolynomial'/'forced'/'delay') so that
# non-polynomial and forced systems can be handled by a future tier without a
# rewrite.

import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import inspect
import itertools
import time

import numpy as np
import pandas as pd
import torch
torch.set_default_dtype(torch.float64)

import sympy as smp

import dysts.flows as flows
from dysts.systems import get_attractor_list
from dysts.base import DynSysDelay

from wsindy import *

# ===========================================================================
# Ground-truth extraction
# ===========================================================================

# Mirrors WSINDy.format_monomial (src/wsindy.py): builds the term name for a
# monomial multi-index `bj` given the local variable `names`. Kept standalone so
# the benchmark can name terms without instantiating a model.
def format_monomial(bj, names):
    terms = []
    for d in range(len(bj)):
        if bj[d] == 0:
            continue
        if bj[d] == 1:
            terms.append(names[d])
        else:
            terms.append(f'{names[d]}^{bj[d]}')
    return '(' + ' '.join(terms) + ')' if terms else '(1)'


# All monomial multi-indices (length n) of total degree <= `degree`, in the
# graded order WSINDy's default library uses: constant, then degree 1, then
# degree 2, ... (matches itertools.combinations_with_replacement ordering).
def build_beta(n, degree):
    beta = []
    for d in range(degree + 1):
        for combo in itertools.combinations_with_replacement(range(n), d):
            powers = [0] * n
            for idx in combo:
                powers[idx] += 1
            beta.append(powers)
    return beta


# Extract the exact RHS of a dysts system as monomial coefficients.
#
# Returns a dict:
#   status      : 'polynomial' | 'nonpolynomial' | 'forced' | 'delay'
#   state_names : state-variable names in physical order (from _rhs signature)
#   exprs       : list of sympy expressions (one per equation), or None
#   coeff_dicts : list of {monomial-tuple (physical order): float coeff}, or None
#   max_degree  : highest total monomial degree across all equations, or None
#
# Non-polynomial systems (transcendental terms, |x|, etc.) are detected either
# by a numpy-ufunc failure when fed sympy symbols, or by a PolynomialError.
# Explicit t-dependence -> 'forced'. Delay systems -> 'delay'.
def get_ground_truth(name):
    model = getattr(flows, name)()
    base = {'status': None, 'state_names': None, 'exprs': None,
            'coeff_dicts': None, 'max_degree': None}

    if isinstance(model, DynSysDelay):
        base['status'] = 'delay'
        return base

    f = model._rhs.py_func   # pure-Python (un-jitted) RHS accepts sympy symbols
    pnames = list(inspect.signature(f).parameters)
    ti = pnames.index('t')
    state_names = pnames[:ti]
    param_names = pnames[ti + 1:]
    n = len(state_names)
    base['state_names'] = list(state_names)

    syms = smp.symbols(state_names)
    if not isinstance(syms, (list, tuple)):
        syms = (syms,)
    syms = list(syms)
    t_sym = smp.Symbol('t')
    params = {p: model.params[p] for p in param_names}

    try:
        out = f(*syms, t_sym, **params)
    except Exception:
        # np.sin/np.cos/etc. reject sympy symbols -> transcendental
        base['status'] = 'nonpolynomial'
        return base

    exprs = [smp.expand(smp.sympify(e)) for e in out]
    base['exprs'] = exprs

    if any(t_sym in e.free_symbols for e in exprs):
        base['status'] = 'forced'
        return base

    try:
        polys = [smp.Poly(e, *syms) for e in exprs]
    except smp.PolynomialError:
        base['status'] = 'nonpolynomial'
        return base

    coeff_dicts = [dict(zip(p.monoms(), [float(c) for c in p.coeffs()])) for p in polys]
    base['coeff_dicts'] = coeff_dicts
    base['max_degree'] = max((sum(mon) for cd in coeff_dicts for mon in cd), default=0)
    base['status'] = 'polynomial'
    return base


# Build the true_coeffs dict {WSINDy-term-name: coeff} for equation i, using the
# local variable ordering WSINDy assigns that equation (LHS var first, then the
# others). Only nonzero true terms are included, matching how
# WSINDy.hyperparameter_sweep consumes true_coeffs.
def true_coeffs_for_eq(gt, i, beta):
    n = len(gt['state_names'])
    others = [j for j in range(n) if j != i]
    local_names = [gt['state_names'][i]] + [gt['state_names'][j] for j in others]
    cd = gt['coeff_dicts'][i]

    true_coeffs = {}
    for bj in beta:
        # map local powers -> physical monomial tuple
        phys = [0] * n
        phys[i] = bj[0]
        for k, j in enumerate(others):
            phys[j] = bj[k + 1]
        coeff = cd.get(tuple(phys), 0.0)
        if coeff != 0.0:
            true_coeffs[format_monomial(bj, local_names)] = coeff
    return true_coeffs


# ===========================================================================
# Data loading
# ===========================================================================

# Generate a trajectory for a dysts system.
#   returns (t, states, names) where states is a list of 1-D torch tensors (one
#   per state variable) and t is a 1-D torch tensor of sample times.
# standardize=False keeps the data on the same scale as the true equations, so
# ground-truth coefficients match. noise is applied via the WSINDy helper.
def load_system(name, n_points=5000, pts_per_period=100, noise=0.0, seed=0, names=None):
    model = getattr(flows, name)()
    t, traj = model.make_trajectory(n_points, pts_per_period=pts_per_period,
                                    return_times=True, standardize=False,
                                    random_seed=seed)
    t = torch.tensor(np.asarray(t), dtype=torch.float64)
    states = [torch.tensor(np.asarray(traj[:, d]), dtype=torch.float64)
              for d in range(traj.shape[1])]
    if noise:
        torch.manual_seed(seed)
        states = add_noise_to_states(states, noise)
    if names is None:
        f = model._rhs.py_func
        pnames = list(inspect.signature(f).parameters)
        names = pnames[:pnames.index('t')]
    return t, states, names


# ===========================================================================
# WSINDy fitting (headless; lifted from examples/wsindy_ode_examples.ipynb)
# ===========================================================================

ODE_ALPHA = [[1], [0]]   # LHS = d/dt; RHS library uses the 0th derivative

# Fit a single ODE: variable Ui on the LHS, auxiliary states V, in local `names`.
def fit_wsindy_ode(Ui, V, names, t, beta, m=None, s=None, Lambda=None,
                   rescale=True, verbosity=False):
    model = WSINDy(Ui, ODE_ALPHA, beta, t, V=V, names=names, m=m, s=s,
                   verbosity=verbosity, rescale=rescale, eqn_type='ode')
    G, powers, derivs, rhs_names = model.create_default_library()
    model.build_lhs(names[0] + model.derivative_names[0])
    model.set_library(G, powers, derivs, rhs_names)
    model.MSTLS(Lambda=Lambda)
    return model


# ===========================================================================
# Benchmarking a single system
# ===========================================================================

# Score one WSINDy fit against known true coefficients (single-fit metrics,
# mirroring WSINDy.hyperparameter_sweep). model.coeffs are in physical units.
def _score_fit(model, true_coeffs, rescale):
    w = model.coeffs
    nonzero = w.nonzero().flatten().tolist()
    learned = {model.rhs_names[j]: w[j].item() for j in nonzero}

    true_support = {name for name, wj in true_coeffs.items() if wj != 0}
    support_error = len(set(learned) ^ true_support)
    if true_coeffs:
        coeff_error = max(abs(learned.get(name, 0.0) - wj) / abs(wj)
                          for name, wj in true_coeffs.items())
    else:
        coeff_error = float('nan')

    w_tilde = w / model.mu if rescale else w
    r, R2 = compute_residuals(model.library, w_tilde, model.lhs)
    rel_L2 = (la.norm(r) / la.norm(model.lhs)).item()

    return {
        'success': support_error == 0,
        'support_error': support_error,
        'coeff_error': coeff_error,
        'rel_L2': rel_L2,
        'R2': R2.item(),
        'n_true': len(true_support),
        'n_learned': len(learned),
        'cond_G': la.cond(model.library).item(),
        'learned': learned,
        'true_coeffs': true_coeffs,
    }


# Run WSINDy on every equation of one system and score against ground truth.
# Returns a dict with 'status', per-equation records, and a system-level summary.
# For non-polynomial/forced/delay systems (v1 out of scope) status is passed
# through and no fit is attempted.
def benchmark_system(name, n_points=5000, pts_per_period=100, noise=0.0, seed=0,
                     degree=None, m=None, s=None, Lambda=None, rescale=True,
                     verbosity=False):
    gt = get_ground_truth(name)
    result = {'system': name, 'status': gt['status'], 'noise': noise,
              'equations': [], 'success': None, 'error': None}
    if gt['status'] != 'polynomial':
        return result

    n = len(gt['state_names'])
    deg = degree if degree is not None else max(2, gt['max_degree'])
    beta = build_beta(n, deg)
    result['degree'] = deg
    result['dimension'] = n

    try:
        t, states, names = load_system(name, n_points=n_points,
                                       pts_per_period=pts_per_period,
                                       noise=noise, seed=seed,
                                       names=gt['state_names'])
    except Exception as e:
        result['error'] = f'load: {type(e).__name__}: {e}'
        return result

    for i in range(n):
        others = [j for j in range(n) if j != i]
        Ui = states[i]
        V = [states[j] for j in others]
        local_names = [names[i]] + [names[j] for j in others]
        true_coeffs = true_coeffs_for_eq(gt, i, beta)
        rec = {'eq': i, 'variable': names[i], 'true_coeffs': true_coeffs}
        try:
            model = fit_wsindy_ode(Ui, V, local_names, t, beta, m=m, s=s,
                                   Lambda=Lambda, rescale=rescale,
                                   verbosity=verbosity)
            rec.update(_score_fit(model, true_coeffs, rescale))
            rec['equation'] = symbolic_eqn(model.lhs_name, model.rhs_names, model.coeffs)
        except Exception as e:
            rec['error'] = f'{type(e).__name__}: {e}'
            rec['success'] = False
        result['equations'].append(rec)

    scored = [r for r in result['equations'] if 'success' in r]
    result['success'] = bool(scored) and all(r['success'] for r in scored)
    return result


# ===========================================================================
# Running the benchmark over many systems
# ===========================================================================

# Classify every continuous dysts system by ground-truth status.
def classify_systems():
    status = {}
    for name in get_attractor_list():
        try:
            status[name] = get_ground_truth(name)['status']
        except Exception as e:
            status[name] = f'error:{type(e).__name__}'
    return status


# Systems eligible for a given benchmark tier. v1 tier: 'poly_autonomous'.
# Future tiers (non-polynomial, forced) can be added here without touching the
# rest of the pipeline.
def eligible_systems(tier='poly_autonomous', status=None):
    status = status if status is not None else classify_systems()
    if tier == 'poly_autonomous':
        return [n for n, s in status.items() if s == 'polynomial']
    raise ValueError(f'unknown tier: {tier!r}')


def _summary_rows(name, res, noise, elapsed):
    n_ok = sum(1 for r in res['equations'] if r.get('success'))
    coeff_errs = [r['coeff_error'] for r in res['equations']
                  if isinstance(r.get('coeff_error'), float) and np.isfinite(r['coeff_error'])]
    sys_row = {
        'system': name, 'status': res['status'],
        'dimension': res.get('dimension'), 'degree': res.get('degree'),
        'noise': noise, 'system_success': res['success'],
        'n_eq': len(res['equations']), 'n_recovered': n_ok,
        'max_coeff_error': max(coeff_errs) if coeff_errs else float('nan'),
        'time_s': elapsed, 'error': res.get('error'),
    }
    eq_rows = [{
        'system': name, 'dimension': res.get('dimension'),
        'degree': res.get('degree'), 'noise': noise,
        'eq': r['eq'], 'variable': r['variable'], 'success': r.get('success'),
        'support_error': r.get('support_error'), 'coeff_error': r.get('coeff_error'),
        'rel_L2': r.get('rel_L2'), 'R2': r.get('R2'),
        'n_true': r.get('n_true'), 'n_learned': r.get('n_learned'),
        'cond_G': r.get('cond_G'), 'equation': r.get('equation'),
        'error': r.get('error'),
    } for r in res['equations']]
    return sys_row, eq_rows


def _print_summary(sys_df):
    n = len(sys_df)
    n_ok = int((sys_df['system_success'] == True).sum())
    print(f'\nSystems fully recovered: {n_ok}/{n} ({100 * n_ok / max(n, 1):.0f}%)')
    dims = sys_df.dropna(subset=['dimension'])
    if len(dims):
        print('By dimension:')
        for dim, grp in dims.groupby('dimension'):
            ok = int((grp['system_success'] == True).sum())
            print(f'  {int(dim)}D: {ok}/{len(grp)} recovered')


# Run the benchmark over a list of systems (default: all poly-autonomous).
# Returns (systems_df, equations_df); optionally writes <csv_prefix>_*.csv.
def run_benchmark(system_list=None, noise=0.0, n_points=5000, pts_per_period=100,
                  seed=0, degree=None, m=None, s=None, Lambda=None, rescale=True,
                  csv_prefix=None, verbose=True):
    if system_list is None:
        if verbose:
            print('Classifying dysts systems...')
        system_list = eligible_systems('poly_autonomous')
        if verbose:
            print(f'{len(system_list)} polynomial-autonomous systems.')

    sys_rows, eq_rows = [], []
    for name in tqdm(system_list, disable=not verbose):
        t0 = time.time()
        try:
            res = benchmark_system(name, n_points=n_points, pts_per_period=pts_per_period,
                                   noise=noise, seed=seed, degree=degree, m=m, s=s,
                                   Lambda=Lambda, rescale=rescale)
        except Exception as e:
            res = {'system': name, 'status': 'benchmark_error', 'equations': [],
                   'success': False, 'error': f'{type(e).__name__}: {e}'}
        sys_row, rows = _summary_rows(name, res, noise, time.time() - t0)
        sys_rows.append(sys_row)
        eq_rows.extend(rows)

    sys_df = pd.DataFrame(sys_rows)
    eq_df = pd.DataFrame(eq_rows)
    if csv_prefix:
        sys_df.to_csv(f'{csv_prefix}_systems.csv', index=False)
        eq_df.to_csv(f'{csv_prefix}_equations.csv', index=False)
    if verbose:
        _print_summary(sys_df)
    return sys_df, eq_df


# ===========================================================================
# Regression-test wrapper (staged: fast pass/fail over a curated subset)
# ===========================================================================

# Curated polynomial-autonomous systems that recover cleanly (correct support,
# coeff error < 1e-12) and reasonably fast in single-fit mode on clean data.
# Chosen from the full benchmark run (data/benchmark_noise0_*.csv) to span 3D and
# 4D; used as a fast check that changes to the WSINDy code have not broken
# equation recovery. ~2-3 min to run all eight.
REGRESSION_SYSTEMS = [
    'Lorenz',       # 3D, canonical
    'Halvorsen',    # 3D, quadratic in every equation
    'SprottP',      # 3D
    'GenesioTesi',  # 3D
    'Hadley',       # 3D
    'RabinovichFabrikant',  # 3D, cubic terms
    'HenonHeiles',  # 4D
    'HyperWang',    # 4D
]


# Assert that every curated system fully recovers (correct support) with
# coefficient error below coeff_tol. Raises AssertionError listing failures, so
# it works directly as a pytest/CI check. Returns True on success.
def test_wsindy_recovery(systems=None, noise=0.0, coeff_tol=1e-6,
                         n_points=5000, pts_per_period=100, seed=0, verbose=True):
    systems = systems if systems is not None else REGRESSION_SYSTEMS
    failures = []
    for name in systems:
        res = benchmark_system(name, n_points=n_points, pts_per_period=pts_per_period,
                               noise=noise, seed=seed)
        why = None
        if res['status'] != 'polynomial':
            why = f'status={res["status"]}'
        elif not res['success']:
            missed = [r['variable'] for r in res['equations'] if not r.get('success')]
            why = f'support not recovered for {missed}'
        else:
            max_ce = max(r['coeff_error'] for r in res['equations'])
            if not (max_ce <= coeff_tol):
                why = f'coeff_error {max_ce:.2e} > {coeff_tol:.0e}'
        if why:
            failures.append((name, why))
        if verbose:
            tag = 'FAIL' if why else 'ok'
            extra = f'  ({why})' if why else ''
            print(f'  {name:22s} {tag}{extra}')
    assert not failures, ('WSINDy regression failures: '
                          + '; '.join(f'{n}: {w}' for n, w in failures))
    if verbose:
        print(f'All {len(systems)} regression systems recovered '
              f'(noise={noise:g}, coeff_tol={coeff_tol:g}).')
    return True


if __name__ == '__main__':
    test_wsindy_recovery()
