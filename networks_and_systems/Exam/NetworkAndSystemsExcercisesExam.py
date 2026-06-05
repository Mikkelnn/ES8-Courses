#!/usr/bin/env python3
"""Python conversion of the Maple worksheet NetworkAndSystemsExcercisesExam.mw.

The original Maple worksheet contains one exercise from lectures 4 and 5.
It defines the constants below, evaluates three calculations, and creates the
same plots that were stored in the worksheet.

Run normally to open the figures:
    python NetworkAndSystemsExcercisesExam.py

Save all figures without opening plot windows:
    python NetworkAndSystemsExcercisesExam.py --no-show --save-dir plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Constants copied from the Maple worksheet.
BW = 1e6
nSensors = 8
fs = 1000
Overhead = 100
sampleBytes = 100


def data(t: np.ndarray | float) -> np.ndarray | float:
    """Original Maple definition: Data(t) := 100*t."""
    return sampleBytes * t


def sent_data(t: np.ndarray | float) -> np.ndarray | float:
    """Original Maple definition: SentData(t) := 125*t - 100."""
    return 125 * t - 100


def aggregate_bandwidth(samples_per_packet: np.ndarray | float) -> np.ndarray | float:
    """Bandwidth used when each packet contains the requested sample count.

    This matches the worksheet expression:
        ((x*100 + Overhead) * nSensors * fs) / x
    """
    x = samples_per_packet
    return ((x * sampleBytes + Overhead) * nSensors * fs) / x


def packetized_data(t: np.ndarray | float, n: int) -> np.ndarray | float:
    """Worksheet F[n] step function for n samples per packet."""
    return np.floor(t / n) * (n * sampleBytes + Overhead) * nSensors


def max_rate(t: np.ndarray | float) -> np.ndarray | float:
    """Original Maple definition: maxRate(t) := (BW/1000)*t."""
    return (BW / 1000) * t


def solve_values() -> tuple[float, float, float]:
    """Evaluate the three Maple solve(...) calls algebraically."""
    # solve(BW = (x + Overhead) * nSensors * fs, x)
    payload_per_sensor_sample = BW / (nSensors * fs) - Overhead

    # solve(Data(t) = SentData(t), t)
    crossover_time = Overhead / (125 - sampleBytes)

    # solve(BW = ((x*100 + Overhead) * nSensors * fs) / x, x)
    denominator = BW - sampleBytes * nSensors * fs
    if denominator == 0:
        raise ZeroDivisionError("No finite packet size satisfies the bandwidth equation.")
    samples_per_packet = Overhead * nSensors * fs / denominator

    return payload_per_sensor_sample, crossover_time, samples_per_packet


def save_or_keep_figure(
    fig: plt.Figure,
    save_dir: Path | None,
    filename: str,
) -> None:
    """Save a plot when --save-dir is supplied."""
    if save_dir is not None:
        fig.savefig(save_dir / filename, dpi=150, bbox_inches="tight")


def create_plots(save_dir: Path | None = None) -> None:
    """Create the 18 plots represented in the Maple worksheet."""
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Maple: plot([Data(t), SentData(t)], t = 0 .. 10)
    t = np.linspace(0, 10, 501)
    fig = plt.figure()
    plt.plot(t, data(t), label="Data(t) = 100t")
    plt.plot(t, sent_data(t), label="SentData(t) = 125t - 100")
    plt.xlabel("t")
    plt.ylabel("Data")
    plt.title("Data and sent data")
    plt.grid(True)
    plt.legend()
    save_or_keep_figure(fig, save_dir, "01_data_vs_sent_data.png")

    # Maple: plot([1e6, ((x*100 + Overhead)*8*1000)/x], x = 1 .. 10)
    x = np.linspace(1, 10, 501)
    fig = plt.figure()
    plt.plot(x, np.full_like(x, BW), label="BW")
    plt.plot(x, aggregate_bandwidth(x), label="Aggregate bandwidth")
    plt.xlabel("Samples per packet (x)")
    plt.ylabel("Bandwidth")
    plt.title("Bandwidth requirement versus packet size")
    plt.grid(True)
    plt.legend()
    save_or_keep_figure(fig, save_dir, "02_bandwidth_vs_packet_size.png")

    # Maple loop: plot([maxRate(t), F[n]], t = 0 .. 50), for n = 1 .. 8
    t = np.linspace(0, 50, 2501)
    for n in range(1, nSensors + 1):
        fig = plt.figure()
        plt.plot(t, max_rate(t), label="maxRate(t)")
        plt.plot(t, packetized_data(t, n), label=f"F[{n}](t)")
        plt.xlabel("t")
        plt.ylabel("Data")
        plt.title(f"Aggregate packetized data, n = {n}")
        plt.grid(True)
        plt.legend()
        save_or_keep_figure(fig, save_dir, f"03_aggregate_packetized_n_{n}.png")

    # Maple loop: plot([maxRate(t), F[n]/8], t = 0 .. 50), for n = 1 .. 8
    for n in range(1, nSensors + 1):
        fig = plt.figure()
        plt.plot(t, max_rate(t), label="maxRate(t)")
        plt.plot(t, packetized_data(t, n) / nSensors, label=f"F[{n}](t) / {nSensors}")
        plt.xlabel("t")
        plt.ylabel("Data")
        plt.title(f"Per-sensor packetized data, n = {n}")
        plt.grid(True)
        plt.legend()
        save_or_keep_figure(fig, save_dir, f"04_per_sensor_packetized_n_{n}.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-dir",
        type=Path,
        help="Optional directory where all generated figures are saved as PNG files.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open plot windows. Useful together with --save-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    payload, crossover, packet_size = solve_values()
    print(f"Payload per sensor sample: {payload:g}")
    print(f"Data(t) and SentData(t) cross at t = {crossover:g}")
    print(f"Samples per packet satisfying BW: {packet_size:g}")

    create_plots(args.save_dir)
    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
