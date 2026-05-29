"""LiDAR exercises 4--6 solution: ray/circle intersections, point-cloud simulation,
and least-squares wall fitting.

Run:
    python lidar_exercises_4_6_solution.py

Outputs:
    ex4_robot_point_cloud.png
    ex5_ex6_robot_wall_fit.png
    ex7_robot_positions.png
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


@dataclass
class ScanResult:
    angles: np.ndarray       # measured beam angles [rad]
    ranges: np.ndarray       # measured ranges [m]
    points: np.ndarray       # measured Cartesian points, shape (N, 2)
    labels: np.ndarray       # 'robot', 'wall', or 'max_range'
    true_ranges: np.ndarray  # noiseless ranges before adding range noise


def line_circle_intersections(
    line_point: Iterable[float],
    line_direction: Iterable[float],
    circle_center: Iterable[float],
    radius: float,
    *,
    as_ray: bool = False,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return line/circle intersection points.

    The line is p(t) = line_point + t * line_direction. If as_ray=True, only
    intersections with t >= 0 are returned. The function returns a (0,2), (1,2),
    or (2,2) array.
    """
    p0 = np.asarray(line_point, dtype=float)
    v = np.asarray(line_direction, dtype=float)
    c = np.asarray(circle_center, dtype=float)

    norm_v = np.linalg.norm(v)
    if norm_v < eps:
        raise ValueError("line_direction must be nonzero")
    v = v / norm_v

    # Solve ||p0 + t v - c||^2 = radius^2.
    q = p0 - c
    A = 1.0
    B = 2.0 * np.dot(v, q)
    C = np.dot(q, q) - radius**2
    disc = B**2 - 4.0 * A * C

    if disc < -eps:
        return np.empty((0, 2))
    if abs(disc) <= eps:
        ts = np.array([-B / (2.0 * A)])
    else:
        root = math.sqrt(disc)
        ts = np.array([(-B - root) / (2.0 * A), (-B + root) / (2.0 * A)])

    if as_ray:
        ts = ts[ts >= -eps]
    pts = p0 + ts[:, None] * v
    return pts


def ray_circle_nearest(
    origin: np.ndarray,
    direction: np.ndarray,
    circle_center: np.ndarray,
    radius: float,
    eps: float = 1e-9,
) -> Optional[float]:
    """Nearest positive ray/circle distance t, or None if no hit."""
    pts = line_circle_intersections(origin, direction, circle_center, radius, as_ray=True)
    if len(pts) == 0:
        return None
    # direction is assumed unit-length in simulate_scan, so distance = projection t.
    ts = (pts - origin) @ direction
    ts = ts[ts > eps]
    return None if len(ts) == 0 else float(np.min(ts))


def ray_segment_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    eps: float = 1e-9,
) -> Optional[float]:
    """Nearest positive ray/segment distance t, or None if no hit."""
    seg = p2 - p1
    M = np.column_stack((direction, -seg))
    det = np.linalg.det(M)
    if abs(det) < eps:
        return None
    rhs = p1 - origin
    t, s = np.linalg.solve(M, rhs)
    if t >= eps and -eps <= s <= 1.0 + eps:
        return float(t)
    return None


def simulate_scan(
    robot_center: Tuple[float, float] = (3.0, 0.7),
    robot_radius: float = 0.60,
    wall: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    *,
    n_beams: int = 720,
    fov: float = 2.0 * math.pi,
    max_range: float = 7.0,
    range_noise_std: float = 0.025,
    angle_noise_std: float = math.radians(0.05),
    seed: int = 7,
) -> ScanResult:
    """Simulate a 2D scanning LiDAR in the x-y plane.

    The cylindrical robot is represented by a circle in top view. The wall is an
    optional finite line segment. Each ray returns the nearest hit among the robot
    and wall; otherwise it returns max_range.
    """
    rng = np.random.default_rng(seed)
    origin = np.array([0.0, 0.0])
    center = np.asarray(robot_center, dtype=float)

    start = -fov / 2.0
    angles = np.linspace(start, start + fov, n_beams, endpoint=False)
    ranges = np.full(n_beams, max_range, dtype=float)
    labels = np.array(["max_range"] * n_beams, dtype=object)

    for i, theta in enumerate(angles):
        u = np.array([math.cos(theta), math.sin(theta)])
        candidates = []

        tr = ray_circle_nearest(origin, u, center, robot_radius)
        if tr is not None and tr <= max_range:
            candidates.append((tr, "robot"))

        if wall is not None:
            p1 = np.asarray(wall[0], dtype=float)
            p2 = np.asarray(wall[1], dtype=float)
            tw = ray_segment_intersection(origin, u, p1, p2)
            if tw is not None and tw <= max_range:
                candidates.append((tw, "wall"))

        if candidates:
            ranges[i], labels[i] = min(candidates, key=lambda z: z[0])

    true_ranges = ranges.copy()
    measured_ranges = ranges + rng.normal(0.0, range_noise_std, size=n_beams)
    measured_ranges = np.clip(measured_ranges, 0.0, max_range)
    measured_angles = angles + rng.normal(0.0, angle_noise_std, size=n_beams)
    points = np.column_stack((measured_ranges * np.cos(measured_angles),
                              measured_ranges * np.sin(measured_angles)))

    return ScanResult(measured_angles, measured_ranges, points, labels, true_ranges)


def fit_line_normal(points: np.ndarray) -> Tuple[float, float, float]:
    """Least-squares line fit in normal form a*x + b*y + c = 0.

    Constraint: a^2 + b^2 = 1. This is the orthogonal least-squares formulation:
        minimize sum_i (a*x_i + b*y_i + c)^2 subject to a^2 + b^2 = 1.

    Returns (a, b, c), with a^2 + b^2 = 1.
    """
    if len(points) < 2:
        raise ValueError("At least two points are required")

    centroid = points.mean(axis=0)
    X = points - centroid
    # Normal vector is the right singular vector corresponding to smallest singular value.
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    normal = vh[-1, :]
    normal = normal / np.linalg.norm(normal)
    c = -float(normal @ centroid)

    # Make the sign deterministic: prefer positive x coefficient when possible.
    if normal[0] < 0 or (abs(normal[0]) < 1e-12 and normal[1] < 0):
        normal = -normal
        c = -c
    return float(normal[0]), float(normal[1]), float(c)


def fit_circle_center_known_radius(
    points: np.ndarray,
    radius: float,
    max_iter: int = 25,
    tol: float = 1e-10,
) -> np.ndarray:
    """Fit a circle center with known radius using Gauss-Newton least squares.

    This mirrors Wasik et al.'s objective: minimize sum_i (||p_i - c|| - R)^2.
    """
    if len(points) < 3:
        raise ValueError("At least three points are required for circle fitting")

    # Algebraic initial estimate for unknown radius, then refine with known radius.
    x = points[:, 0]
    y = points[:, 1]
    A = np.column_stack((x, y, np.ones_like(x)))
    b = -(x**2 + y**2)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    center = np.array([-sol[0] / 2.0, -sol[1] / 2.0])

    for _ in range(max_iter):
        diff = points - center
        d = np.linalg.norm(diff, axis=1)
        d = np.maximum(d, 1e-12)
        residual = d - radius
        J = -diff / d[:, None]
        step, *_ = np.linalg.lstsq(J, -residual, rcond=None)
        center = center + step
        if np.linalg.norm(step) < tol:
            break
    return center


def cluster_points_by_distance(points: np.ndarray, threshold: float = 0.18) -> list[np.ndarray]:
    """Simple nearest-neighbor clustering in scan order."""
    if len(points) == 0:
        return []
    clusters = [[points[0]]]
    for p_prev, p in zip(points[:-1], points[1:]):
        if np.linalg.norm(p - p_prev) > threshold:
            clusters.append([p])
        else:
            clusters[-1].append(p)
    return [np.asarray(c) for c in clusters]


def plot_scan(
    scan: ScanResult,
    filename: str,
    *,
    title: str,
    robot_center: Tuple[float, float],
    robot_radius: float,
    wall: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    fitted_wall_line: Optional[Tuple[float, float, float]] = None,
    fitted_robot_center: Optional[Tuple[float, float]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    finite = scan.labels != "max_range"
    for label in ["robot", "wall"]:
        mask = scan.labels == label
        if np.any(mask):
            ax.scatter(scan.points[mask, 0], scan.points[mask, 1], s=12, label=f"LiDAR returns: {label}")

    # Plot max range points faintly only if desired; here omitted to keep the point cloud clear.
    ax.scatter([0], [0], marker="x", s=70, label="LiDAR")
    ax.add_patch(Circle(robot_center, robot_radius, fill=False, linewidth=2, label="true robot outline"))

    if fitted_robot_center is not None:
        ax.add_patch(Circle(fitted_robot_center, robot_radius, fill=False, linestyle="--", linewidth=2,
                            label="fitted robot circle"))

    if wall is not None:
        p1 = np.asarray(wall[0], dtype=float)
        p2 = np.asarray(wall[1], dtype=float)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=2, label="true wall")

    if fitted_wall_line is not None:
        a, b, c = fitted_wall_line
        if abs(b) > 1e-3:
            xs = np.linspace(-0.5, 6.5, 200)
            ys = -(a * xs + c) / b
        else:
            ys = np.linspace(-4.5, 4.5, 200)
            xs = np.full_like(ys, -c / a)
        ax.plot(xs, ys, linestyle="--", linewidth=2, label="least-squares wall fit")

    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)


def demo() -> None:
    robot_radius = 0.60  # Wasik et al. use R = 60 cm for the circular robot model.

    # Exercise 4: robot only.
    scan4 = simulate_scan(robot_center=(3.0, 0.7), robot_radius=robot_radius, wall=None,
                          n_beams=720, max_range=7.0, seed=10)
    robot_pts4 = scan4.points[scan4.labels == "robot"]
    fitted_center4 = fit_circle_center_known_radius(robot_pts4, robot_radius)
    plot_scan(scan4, "/mnt/data/ex4_robot_point_cloud.png",
              title="Exercise 4: simulated LiDAR returns from a cylindrical robot",
              robot_center=(3.0, 0.7), robot_radius=robot_radius,
              fitted_robot_center=tuple(fitted_center4))

    # Exercises 5 and 6: robot in front of a wall, then fit wall cluster.
    wall = ((5.0, -4.0), (5.0, 4.0))
    scan56 = simulate_scan(robot_center=(3.0, 0.7), robot_radius=robot_radius, wall=wall,
                           n_beams=720, max_range=7.0, seed=11)
    wall_pts = scan56.points[scan56.labels == "wall"]
    fitted_wall = fit_line_normal(wall_pts)
    plot_scan(scan56, "/mnt/data/ex5_ex6_robot_wall_fit.png",
              title="Exercises 5--6: robot in front of wall and fitted wall line",
              robot_center=(3.0, 0.7), robot_radius=robot_radius,
              wall=wall, fitted_wall_line=fitted_wall)

    # Exercise 7 optional: move the robot and update point clouds.
    positions = [(2.4, -1.0), (3.0, 0.7), (3.7, 1.4)]
    fig, ax = plt.subplots(figsize=(8, 6))
    p1 = np.asarray(wall[0], dtype=float)
    p2 = np.asarray(wall[1], dtype=float)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=2, label="wall")
    ax.scatter([0], [0], marker="x", s=70, label="LiDAR")
    for k, pos in enumerate(positions, start=1):
        scan = simulate_scan(robot_center=pos, robot_radius=robot_radius, wall=wall,
                             n_beams=720, max_range=7.0, seed=20 + k)
        finite = scan.labels != "max_range"
        ax.scatter(scan.points[finite, 0], scan.points[finite, 1], s=8, label=f"scan {k}")
        ax.add_patch(Circle(pos, robot_radius, fill=False, linewidth=1.5))
    ax.set_title("Exercise 7: updated point clouds for different robot positions")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig("/mnt/data/ex7_robot_positions.png", dpi=160)
    plt.close(fig)

    print("Exercise 4 fitted robot center [m]:", fitted_center4)
    print("Exercise 5/6 fitted wall line a*x + b*y + c = 0:", fitted_wall)
    print("Wall points used:", len(wall_pts))


if __name__ == "__main__":
    demo()
