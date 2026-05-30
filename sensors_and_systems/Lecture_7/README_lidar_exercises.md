# LiDAR Exercise Solutions

This folder contains a Python/Jupyter solution for the seven LiDAR exercises.

## Files

- `lidar_exercises_solution.ipynb` — complete executed Jupyter notebook with formulas, code, plots, and conclusions.
- `README_lidar_exercises.md` — this explanation.

## Main formulas used

### 1. LiDAR received-to-emitted power ratio

For a single scatterer, the simplified LiDAR equation is

$$
\frac{P_R}{P_T} = \frac{\rho A_s D_r^2 \eta_{\mathrm{atm}} \eta_{\mathrm{sys}}}{R^4 \beta_t^2 \Omega}
$$

where $\beta_t = \tan(\phi)$. Using the exercise values:

$$
\begin{aligned}
R &= 150\,\mathrm{m} \\
\phi &= 3\,\mathrm{mrad} \\
D_r &= 0.15\,\mathrm{m} \\
A_s &= 0.17\,\mathrm{m}^2 \\
\rho &= 0.2 \\
\Omega &= 0.5\,\mathrm{sr} \\
\eta_{\mathrm{atm}} &= 0.9 \\
\eta_{\mathrm{sys}} &= 0.9
\end{aligned}
$$

gives

$$
\frac{P_R}{P_T} \approx 2.7 \times 10^{-7}
$$

This is the same order as the expected answer in the exercise statement, approximately $2.6 \times 10^{-7}$; the small difference is due to rounding and using $\tan(\phi)$ rather than $\phi$ directly.

### 2. Best straight-line fit

The line is represented implicitly:

$$
a x + b y + c = 0, \qquad a^2 + b^2 = 1
$$

The optimization problem is

$$
\begin{aligned}
\text{minimize} \quad & \sum_i (a x_i + b y_i + c)^2 \\
\text{subject to} \quad & a^2 + b^2 = 1
\end{aligned}
$$

Because $(a,b)$ is a unit normal vector, each residual is a perpendicular distance. This handles vertical and non-vertical lines. The notebook solves it with total least squares using SVD/PCA.

### 3. Line-circle intersection

The line through two points is written as

$$
\mathbf{p}(t) = \mathbf{p}_1 + t(\mathbf{p}_2 - \mathbf{p}_1)
$$

The circle is

$$
\lVert \mathbf{p}(t) - \mathbf{c} \rVert^2 = r^2
$$

Substitution gives a quadratic in $t$. The discriminant determines whether there are zero, one, or two intersections.

### 4–7. 2D LiDAR simulation

A LiDAR beam is modeled as a ray:

$$
\mathbf{p}(t) = \mathbf{o} + t[\cos(\theta), \sin(\theta)], \qquad t \geq 0
$$

For each beam, the simulator computes intersections with the circular robot and the wall segment, then keeps the closest positive intersection. Measurement noise is added as

$$
\begin{aligned}
r_{\mathrm{noisy}} &= r + \mathcal{N}(0, \sigma_r^2) \\
\theta_{\mathrm{noisy}} &= \theta + \mathcal{N}(0, \sigma_\theta^2)
\end{aligned}
$$

The robot is modeled as a circle in top view, equivalent to a horizontal slice of a cylinder.

## Results and conclusions

1. The LiDAR power ratio is very small: about $2.7 \times 10^{-7}$. This demonstrates the strong effect of range, beam divergence, receiver aperture, target reflectance, and transmission efficiencies.
2. The total least-squares line fit works for arbitrary line orientations, including vertical walls.
3. The line-circle intersection function correctly returns two intersections, one tangent point, or no points depending on line geometry.
4. The robot-only point cloud forms an arc on the visible side of the cylindrical robot.
5. With a wall behind the robot, the robot occludes central wall returns; side beams still reach the wall.
6. The wall cluster fit recovers a line close to the true wall at $x = 5\,\mathrm{m}$; the notebook reports the fitted equation and RMSE.
7. Moving the robot changes the robot returns and the wall occlusion gap, while the wall fit remains stable when enough wall points are visible.

## How to run

Open `lidar_exercises_solution.ipynb` in JupyterLab, VS Code, or classic Jupyter Notebook and run all cells. The notebook only uses standard scientific Python packages:

```text
numpy
matplotlib
```

No external data files are required to run the solution.

## Sources used

- `LiDAR Remote Sensing Principles_26_01_07_15_40_06-1(1).pdf` for the LiDAR equation and scattering assumptions.
- `Lidar-Based_Relative_Position_Estimation_and_Track-1(1).pdf` by Wąsik et al. for the circular-robot LiDAR detection context.
- `slidesRanging(1).pdf` for the course framing around ranging sensors, point clouds, clustering, and object recognition.
