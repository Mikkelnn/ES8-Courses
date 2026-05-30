#!/usr/bin/env python3

import sys
import serial
import serial.tools.list_ports
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from collections import deque

BAUD = 115200
MAX_POINTS = 500

FIELDS = [
    't_ms', 'accel',
    'bench_v', 'bench_p',
    'ekf_a', 'ekf_v', 'ekf_p', 'ekf_stdv', 'ekf_stdp',
    'ekfb_a', 'ekfb_v', 'ekfb_p', 'ekfb_b', 'ekfb_stdv', 'ekfb_stdp',
]


def find_port():
    if len(sys.argv) > 1:
        return sys.argv[1]

    # Check for Arduino-like devices
    for p in serial.tools.list_ports.comports():
        dev = p.device.lower()
        desc = (p.description or '').lower()

        # Linux: /dev/ttyACM*, /dev/ttyUSB*, /dev/ttyAMA*
        if any(x in dev for x in ['acm', 'usb', 'ama']):
            return p.device

        # Windows: COM*
        if dev.startswith('com') and dev[3:].isdigit():
            return p.device

        # macOS: /dev/tty.usbserial*, /dev/tty.SLAB_USBtoUART, Arduino
        if any(x in dev for x in ['usbserial', 'usbmodem']):
            return p.device

        # Check description for Arduino mentions
        if 'arduino' in desc or 'usb' in desc:
            return p.device

    return '/dev/ttyACM0'


def parse_line(line):
    data = {}
    for token in line.split():
        if ':' in token:
            k, _, v = token.partition(':')
            try:
                data[k] = float(v)
            except ValueError:
                pass
    return data


buf = {k: deque(maxlen=MAX_POINTS) for k in FIELDS}

port = find_port()
print(f"Connecting to {port} @ {BAUD} baud …")
ser = serial.Serial(port, BAUD, timeout=1)
print("Connected. Waiting for data.")

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle('Extended Kalman Filter (3-state & 4-state) — Live', fontsize=13, fontweight='bold')
ax_a, ax_v, ax_p, ax_b = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

# Pre-init line objects for faster updates
line_accel, = ax_a.plot([], [], color='tab:orange', lw=1.5, ls='--', label='raw')
line_ekf_a, = ax_a.plot([], [], color='tab:blue', lw=1.6, label='EKF [a,v,p]')
line_ekfb_a, = ax_a.plot([], [], color='tab:red', lw=1.6, label='EKF [a,v,p,b]')
ax_a.set_title('Acceleration')
ax_a.set_xlabel('t [ms]')
ax_a.set_ylabel('a [m/s²]')
ax_a.legend(fontsize=7)
ax_a.grid(alpha=0.3)

line_bench_v, = ax_v.plot([], [], color='tab:orange', lw=1.5, ls='--', label='bench')
line_ekf_v, = ax_v.plot([], [], color='tab:blue', lw=1.6, label='EKF [a,v,p]')
line_ekfb_v, = ax_v.plot([], [], color='tab:red', lw=1.6, label='EKF [a,v,p,b]')
ax_v.set_title('Velocity')
ax_v.set_xlabel('t [ms]')
ax_v.set_ylabel('v [m/s]')
ax_v.legend(fontsize=7)
ax_v.grid(alpha=0.3)

line_bench_p, = ax_p.plot([], [], color='tab:orange', lw=1.5, ls='--', label='bench')
line_ekf_p, = ax_p.plot([], [], color='tab:blue', lw=1.6, label='EKF [a,v,p]')
line_ekfb_p, = ax_p.plot([], [], color='tab:red', lw=1.6, label='EKF [a,v,p,b]')
ax_p.set_title('Position')
ax_p.set_xlabel('t [ms]')
ax_p.set_ylabel('p [m]')
ax_p.legend(fontsize=7)
ax_p.grid(alpha=0.3)

line_ekfb_b, = ax_b.plot([], [], color='tab:red', lw=1.6, label='EKF [a,v,p,b]')
ax_b.axhline(0, color='k', lw=0.6, linestyle='--')
ax_b.set_title('Estimated Bias')
ax_b.set_xlabel('t [ms]')
ax_b.set_ylabel('b [m/s²]')
ax_b.legend(fontsize=7)
ax_b.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])


def animate(_):
    # Read available serial data
    while ser.in_waiting:
        try:
            raw = ser.readline().decode('utf-8', errors='ignore').strip()
        except Exception:
            break
        if not raw or raw.startswith('#'):
            continue
        d = parse_line(raw)
        # Only append if t_ms present (minimal completeness check)
        if 't_ms' in d:
            for k in FIELDS:
                buf[k].append(d.get(k, float('nan')))

    t = np.array(list(buf['t_ms']))
    n = len(t)
    if n < 2:
        return []

    def g(k):
        v = np.array(list(buf[k]))
        return v

    # Get all data
    accel = g('accel')
    ekf_a = g('ekf_a')
    ekfb_a = g('ekfb_a')

    bench_v = g('bench_v')
    ekf_v = g('ekf_v')
    ekf_sv = g('ekf_stdv')
    ekfb_v = g('ekfb_v')
    ekfb_sv = g('ekfb_stdv')

    bench_p = g('bench_p')
    ekf_p = g('ekf_p')
    ekf_sp = g('ekf_stdp')
    ekfb_p = g('ekfb_p')
    ekfb_sp = g('ekfb_stdp')

    ekfb_b = g('ekfb_b')

    # Update acceleration lines
    line_accel.set_data(t, accel)
    line_ekf_a.set_data(t, ekf_a)
    line_ekfb_a.set_data(t, ekfb_a)
    ax_a.relim()
    ax_a.autoscale_view()

    # Update velocity lines and bands
    line_bench_v.set_data(t, bench_v)
    line_ekf_v.set_data(t, ekf_v)
    line_ekfb_v.set_data(t, ekfb_v)

    # Remove old fill_between and create new ones
    for patch in ax_v.patches[:]:
        patch.remove()
    ekf_hi = ekf_v + 2 * ekf_sv
    ekf_lo = ekf_v - 2 * ekf_sv
    ax_v.fill_between(t, ekf_lo, ekf_hi, alpha=0.15, color='tab:blue', edgecolor='none')
    ekfb_hi = ekfb_v + 2 * ekfb_sv
    ekfb_lo = ekfb_v - 2 * ekfb_sv
    ax_v.fill_between(t, ekfb_lo, ekfb_hi, alpha=0.15, color='tab:red', edgecolor='none')

    ax_v.relim()
    ax_v.autoscale_view()

    # Update position lines and bands
    line_bench_p.set_data(t, bench_p)
    line_ekf_p.set_data(t, ekf_p)
    line_ekfb_p.set_data(t, ekfb_p)

    # Remove old fill_between and create new ones
    for patch in ax_p.patches[:]:
        patch.remove()
    ekf_hi = ekf_p + 2 * ekf_sp
    ekf_lo = ekf_p - 2 * ekf_sp
    ax_p.fill_between(t, ekf_lo, ekf_hi, alpha=0.15, color='tab:blue', edgecolor='none')
    ekfb_hi = ekfb_p + 2 * ekfb_sp
    ekfb_lo = ekfb_p - 2 * ekfb_sp
    ax_p.fill_between(t, ekfb_lo, ekfb_hi, alpha=0.15, color='tab:red', edgecolor='none')

    ax_p.relim()
    ax_p.autoscale_view()

    # Update bias line
    line_ekfb_b.set_data(t, ekfb_b)
    ax_b.relim()
    ax_b.autoscale_view()

    return [line_accel, line_ekf_a, line_ekfb_a,
            line_bench_v, line_ekf_v, line_ekfb_v,
            line_bench_p, line_ekf_p, line_ekfb_p,
            line_ekfb_b]


ani = animation.FuncAnimation(fig, animate, interval=50, cache_frame_data=False, blit=True)

try:
    plt.show()
finally:
    ser.close()
    print("Serial closed.")
