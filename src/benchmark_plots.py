# Aggregate plots for the dysts WSINDy benchmark.
#
# Design notes (from the dataviz method): forms chosen by the data's job
# (magnitude -> bars, distribution -> histogram, change-over-parameter -> line);
# a fixed-order colorblind-safe palette (Okabe-Ito); recovered/missed use
# reserved good/bad status colors with hatching for redundant (non-color)
# encoding; single y-axis; recessive grid.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette (fixed order, never cycled past its length)
OKABE_ITO = ['#0072B2', '#E69F00', '#009E73', '#D55E00',
             '#56B4E9', '#CC79A7', '#F0E442', '#000000']
GOOD, BAD = '#009E73', '#D55E00'   # reserved status colors: recovered / missed


def load_results(prefix):
    """Load <prefix>_systems.csv and <prefix>_equations.csv -> (sys_df, eq_df)."""
    return pd.read_csv(f'{prefix}_systems.csv'), pd.read_csv(f'{prefix}_equations.csv')


def plot_success_by_dimension(sys_df, ax=None):
    """Stacked bars: recovered vs missed systems per state-space dimension."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    df = sys_df.dropna(subset=['dimension']).copy()
    df['dimension'] = df['dimension'].astype(int)
    grp = df.groupby('dimension')['system_success'].agg(
        recovered=lambda s: int((s == True).sum()), total='count')
    grp['missed'] = grp['total'] - grp['recovered']
    dims = grp.index.astype(str)

    ax.bar(dims, grp['recovered'], width=0.6, color=GOOD, label='Recovered', zorder=2)
    ax.bar(dims, grp['missed'], width=0.6, bottom=grp['recovered'], color=BAD,
           hatch='///', edgecolor='white', linewidth=0, label='Missed', zorder=2)
    for x, (rec, tot) in enumerate(zip(grp['recovered'], grp['total'])):
        ax.text(x, tot + 0.15, f'{rec}/{tot}', ha='center', va='bottom', fontsize=9)
    ax.set_xlabel('State-space dimension')
    ax.set_ylabel('Number of systems')
    ax.set_title('WSINDy recovery by dimension')
    ax.margins(y=0.15)
    ax.grid(True, axis='y', alpha=0.3, color='silver', zorder=0)
    ax.legend(loc='upper right', framealpha=0.9)
    return ax


def plot_coeff_error_hist(eq_df, ax=None, floor=1e-16):
    """Histogram of log10 coefficient error over recovered equations."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    ok = eq_df[(eq_df['success'] == True) & eq_df['coeff_error'].notna()]
    vals = np.log10(np.maximum(ok['coeff_error'].to_numpy(dtype=float), floor))
    ax.hist(vals, bins=25, color=OKABE_ITO[0], edgecolor='white', zorder=2)
    ax.axvline(np.median(vals), ls='--', color=OKABE_ITO[3],
               label=f'median = 1e{np.median(vals):.1f}')
    ax.set_xlabel(r'$\log_{10}$ max relative coefficient error')
    ax.set_ylabel('Recovered equations')
    ax.set_title('Coefficient-error distribution (recovered equations)')
    ax.grid(True, axis='y', alpha=0.3, color='silver', zorder=0)
    ax.legend(loc='upper right', framealpha=0.9)
    return ax


def plot_success_vs_noise(sys_dfs, ax=None):
    """Line: fraction of systems fully recovered vs noise level.
    sys_dfs: a single concatenated DataFrame with a 'noise' column, or a dict
    {noise: sys_df}."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    if isinstance(sys_dfs, dict):
        sys_dfs = pd.concat([d.assign(noise=n) for n, d in sys_dfs.items()])
    frac = (sys_dfs.assign(ok=sys_dfs['system_success'] == True)
            .groupby('noise')['ok'].mean())
    ax.plot(frac.index * 100, frac.values * 100, '-o', color=OKABE_ITO[0],
            markersize=7, linewidth=2, zorder=2)
    ax.set_xlabel('Noise level (%)')
    ax.set_ylabel('Systems fully recovered (%)')
    ax.set_title('WSINDy recovery vs. noise')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, color='silver', zorder=0)
    return ax


def results_table(sys_df, sort_by='max_coeff_error'):
    """Tidy per-system summary sorted for quick scanning."""
    cols = ['system', 'dimension', 'degree', 'system_success',
            'n_eq', 'n_recovered', 'max_coeff_error', 'time_s']
    poly = sys_df[sys_df['status'] == 'polynomial'][cols].copy()
    return poly.sort_values([sort_by], ascending=False).reset_index(drop=True)
