"""
Normal probability plot — Python equivalent of MATLAB normplot().

How it works (step by step):
  1. Sort the data in ascending order.
  2. Assign Hazen plotting positions: p_i = (i - 0.5) / n  for i = 1..n
     These estimate the cumulative probability of each sorted point.
  3. x-axis = sorted data values.
  4. y-axis = norm.ppf(p_i)  (probit transform), but tick labels are shown
              as probabilities (1%, 5%, 25%, 50%, 75%, 95%, 99%) so the scale
              matches MATLAB's non-linear normal probability axis.
  5. Reference line drawn through Q1 and Q3 of the data (same as MATLAB).

A straight-line alignment = data is Gaussian.
No test statistic is computed — this is a visual check only.

Numerical example with n=5, data = [0.3, -0.1, 0.5, -0.4, 0.0]:
  sorted    = [-0.4,  -0.1,   0.0,   0.3,   0.5]
  p_i       = [ 0.10,  0.30,  0.50,  0.70,  0.90]
  probit_y  = [-1.28, -0.52,  0.00,  0.52,  1.28]
  Q1 = -0.25,  Q3 = 0.40
  z25 = norm.ppf(0.25) = -0.6745,  z75 = norm.ppf(0.75) = +0.6745
  ref_slope = (0.40 - (-0.25)) / (0.6745 - (-0.6745)) = 0.482
  ref_intercept = -0.25 - 0.482 * (-0.6745) = 0.075
  -> ref line: probit_y = (x - 0.075) / 0.482
"""

import numpy as np
from scipy.stats import norm as norm_dist


def plot_normplot_panel(ax, data, panel_title):
    """Draw a MATLAB-style normal probability plot on the given axes.

    Parameters
    ----------
    ax          : matplotlib Axes to draw on
    data        : 1-D array of values to test (e.g. filter innovations)
    panel_title : string shown as the axes title
    """
    n           = len(data)
    sorted_data = np.sort(data)

    # Hazen plotting positions: p_i = (i - 0.5) / n
    plotting_positions = (np.arange(1, n + 1) - 0.5) / n
    probit_y           = norm_dist.ppf(plotting_positions)

    # Reference line through Q1 and Q3 (robust to tail outliers, same as MATLAB)
    q25, q75       = np.percentile(data, [25, 75])
    z25, z75       = norm_dist.ppf([0.25, 0.75])   # -0.6745, +0.6745
    ref_slope      = (q75 - q25) / (z75 - z25)
    ref_intercept  = q25 - ref_slope * z25
    line_x         = np.linspace(sorted_data[0], sorted_data[-1], 200)
    line_y         = (line_x - ref_intercept) / ref_slope

    # Plot data as + markers (matches MATLAB normplot style) and reference line
    ax.plot(sorted_data, probit_y, '+', markersize=6, color='steelblue', label='data')
    ax.plot(line_x, line_y, 'r-', lw=1.5, label='normal ref')

    # Y-axis ticks: probability percentages on the normal probability scale
    prob_tick_values = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ax.set_yticks(norm_dist.ppf(prob_tick_values))
    ax.set_yticklabels([f'{int(p * 100)}%' for p in prob_tick_values], fontsize=7)

    ax.set_xlabel('Data')
    ax.set_ylabel('Probability')
    ax.set_title(panel_title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
