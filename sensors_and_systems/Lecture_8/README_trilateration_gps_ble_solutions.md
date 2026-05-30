# Trilateration, GPS, and BLE RSSI — Exercise Solutions

This folder contains a Jupyter notebook that solves the four exercises from the prompt in Python.

## Files

- `trilateration_gps_ble_solutions.ipynb`: executable notebook with formulas, code, numerical checks, and plots.
- `README_trilateration_gps_ble_solutions.md`: this explanation and summary.

The notebook uses `$...$` and `$$...$$` math delimiters so that formulas render in VS Code Markdown/KaTeX.

## How to run

Install the Python packages below if needed:

```bash
pip install numpy scipy matplotlib nbformat nbclient
```

Then open `trilateration_gps_ble_solutions.ipynb` in VS Code or Jupyter and run all cells.

## Exercise 1 — GPS time of flight

Let

$$
r = R_E + h.
$$

The shortest satellite-receiver distance occurs when the satellite is directly overhead:

$$
d_{\min} = R_s - r = R_s - R_E - h.
$$

The longest visible distance occurs at the tangent line of sight:

$$
d_{\max} = \sqrt{R_s^2 - R_E^2} + \sqrt{r^2 - R_E^2}.
$$

Equivalently, with

$$
\beta = \cos^{-1}\left({R_E \over R_s}\right), \quad \phi = \sin^{-1}\left({R_E \over r}\right), \quad \psi = \cos^{-1}\left({R_E \over r}\right),
$$

we get

$$
d_{\max} = R_s {\sin(\psi + \beta) \over \sin \phi}.
$$

The time of flight is

$$
t = {d \over c}.
$$

For a nominal GPS radius $R_s = 26{,}600$ km and a receiver on the surface, the notebook gives approximately $67.48$ ms minimum and $86.15$ ms maximum.

## Exercise 2 — Flight from Paris CDG to Rio Galeão

The great-circle distance is computed with the haversine formula:

$$
D = 2R_E\sin^{-1}\left(\sqrt{\sin^2\left({\Delta\varphi \over 2}\right) + \cos\varphi_1\cos\varphi_2\sin^2\left({\Delta\lambda \over 2}\right)}\right).
$$

Using CDG at approximately $(49.0097^\circ, 2.5479^\circ)$ and GIG at approximately $(-22.8100^\circ, -43.2506^\circ)$ gives

$$
D \approx 9184 \text{ km}.
$$

For a scheduled flight time of $11.5$ h,

$$
v = {D \over 11.5 \text{ h}} \approx 799 \text{ km/h}.
$$

The Earth rotates, using a 24-hour day as in the exercise answer,

$$
360^\circ {11.5 \over 24} = 172.5^\circ.
$$

## Exercise 3 — Blewitt design matrix

The ideal pseudorange model is

$$
P_j = \rho_j(x,y,z) + c\tau - c\tau_j,
$$

where

$$
\rho_j = \sqrt{(x^j-x)^2 + (y^j-y)^2 + (z^j-z)^2}.
$$

At provisional receiver coordinates $(x_0,y_0,z_0)$, row $j$ of the design matrix is

$$A_j = \begin{bmatrix}
{x_0-x^j \over \rho_j} & {y_0-y^j \over \rho_j} & {z_0-z^j \over \rho_j} & c
\end{bmatrix}.$$

The notebook verifies this by comparing the analytic matrix against central finite differences. The maximum difference is tiny, so the expression on p. 17 is correct. The important detail is that each row uses its own satellite range $\rho_j$ evaluated at the provisional receiver position.

## Exercise 4 — BLE RSSI multilateration

The log-distance RSSI model is

$$
RSSI_i = A_i - 10\gamma\log_{10}(d_i).
$$

The inverse distance estimate is

$$
\hat d_i = 10^{(A_i - RSSI_i)/(10\gamma)}.
$$

Given beacons $b_i = (x_i,y_i)$ and candidate receiver position $p=(x,y)$,

$$
r_i(p)=\sqrt{(x-x_i)^2+(y-y_i)^2}.
$$

The optimization problem is

$$
\min_{x,y}\sum_i \left(r_i(x,y)-\hat d_i\right)^2.
$$

The exercise specifies $\gamma=1.74$ and $f=2.45$ GHz. Because no measured beacon power $A_i$ is provided, the notebook computes a default value from free-space path loss at $1$ m:

$$
A = -10\gamma\log_{10}\left({4\pi f \over c}\right).
$$

The simulated room is $5 \times 10$ m with beacons at all four corners. The notebook includes a single-position example plus a Monte Carlo noise test.

## Conclusions

The GPS geometry and Blewitt design matrix results are deterministic under the stated simplifications. The Paris-to-Rio flight computation agrees with the supplied answer after rounding. The BLE RSSI solution works well with exact simulated RSSI, but meter-level errors appear when realistic dB-level noise is added. This is expected because logarithmic RSSI-to-distance conversion amplifies signal fluctuations, especially at larger distances. Reliable indoor BLE positioning therefore depends heavily on calibration of $\gamma$ and $A$, beacon geometry, weak-signal filtering, and smoothing/averaging of repeated measurements.
