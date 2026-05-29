#!/usr/bin/env python3

import sys
import serial
import serial.tools.list_ports
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

BAUD = 115200
MAX_POINTS = 500

FIELDS = [
    't_ms', 'accel',
    'bench_v', 'bench_p',
    'kf_a', 'kf_v', 'kf_p', 'kf_stdv', 'kf_stdp',
    'kfb_a', 'kfb_v', 'kfb_p', 'kfb_b', 'kfb_stdv', 'kfb_stdp',
]


def find_port():
    if len(sys.argv) > 1:
        return sys.argv[1]
    for p in serial.tools.list_ports.comports():
        if 'ACM' in p.device or 'USB' in p.device:
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
fig.suptitle('Kalman Filter — Live', fontsize=13, fontweight='bold')
ax_a, ax_v, ax_p, ax_b = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]


def _bands(ax, t, vals, stds, color):
    hi = [v + 2 * s for v, s in zip(vals, stds)]
    lo = [v - 2 * s for v, s in zip(vals, stds)]
    ax.fill_between(t, lo, hi, alpha=0.15, color=color)


def animate(_):
    while ser.in_waiting:
        try:
            raw = ser.readline().decode('utf-8', errors='ignore').strip()
        except Exception:
            break
        if not raw or raw.startswith('#'):
            continue
        d = parse_line(raw)
        if 't_ms' not in d:
            continue
        for k in FIELDS:
            if k in d:
                buf[k].append(d[k])

    t = list(buf['t_ms'])
    n = len(t)
    if n < 2:
        return

    def g(k):
        v = list(buf[k])
        return v[:n]  # trim to current t length; [] if empty

    # ── Acceleration ──────────────────────────────────────────────
    ax_a.cla()
    ax_a.set_title('Acceleration')
    ax_a.set_xlabel('t [ms]'); ax_a.set_ylabel('a [m/s²]')
    ax_a.plot(t, g('accel'),  color='tab:orange', lw=1.5, ls='--', label='raw')
    ax_a.plot(t, g('kf_a'),   color='tab:blue',   lw=1.6, label='kalman')
    ax_a.plot(t, g('kfb_a'),  color='tab:red',    lw=1.6, label='kalman_bias')
    ax_a.legend(fontsize=7); ax_a.grid(alpha=0.3)

    # ── Velocity ──────────────────────────────────────────────────
    ax_v.cla()
    ax_v.set_title('Velocity')
    ax_v.set_xlabel('t [ms]'); ax_v.set_ylabel('v [m/s]')
    ax_v.plot(t, g('bench_v'), color='tab:orange', lw=1.5, ls='--', label='bench')
    kf_v, kf_sv   = g('kf_v'),  g('kf_stdv')
    kfb_v, kfb_sv = g('kfb_v'), g('kfb_stdv')
    ax_v.plot(t, kf_v,  color='tab:blue', lw=1.6, label='kalman')
    ax_v.plot(t, kfb_v, color='tab:red',  lw=1.6, label='kalman_bias')
    if kf_v and kf_sv:
        _bands(ax_v, t, kf_v,  kf_sv,  'tab:blue')
    if kfb_v and kfb_sv:
        _bands(ax_v, t, kfb_v, kfb_sv, 'tab:red')
    ax_v.legend(fontsize=7); ax_v.grid(alpha=0.3)

    # ── Position ──────────────────────────────────────────────────
    ax_p.cla()
    ax_p.set_title('Position')
    ax_p.set_xlabel('t [ms]'); ax_p.set_ylabel('p [m]')
    ax_p.plot(t, g('bench_p'), color='tab:orange', lw=1.5, ls='--', label='bench')
    kf_p, kf_sp   = g('kf_p'),  g('kf_stdp')
    kfb_p, kfb_sp = g('kfb_p'), g('kfb_stdp')
    ax_p.plot(t, kf_p,  color='tab:blue', lw=1.6, label='kalman')
    ax_p.plot(t, kfb_p, color='tab:red',  lw=1.6, label='kalman_bias')
    if kf_p and kf_sp:
        _bands(ax_p, t, kf_p,  kf_sp,  'tab:blue')
    if kfb_p and kfb_sp:
        _bands(ax_p, t, kfb_p, kfb_sp, 'tab:red')
    ax_p.legend(fontsize=7); ax_p.grid(alpha=0.3)

    # ── Bias ──────────────────────────────────────────────────────
    ax_b.cla()
    ax_b.set_title('Estimated Bias')
    ax_b.set_xlabel('t [ms]'); ax_b.set_ylabel('b [m/s²]')
    ax_b.plot(t, g('kfb_b'), color='tab:red', lw=1.6, label='kfb_b')
    ax_b.axhline(0, color='k', lw=0.6, linestyle='--')
    ax_b.legend(fontsize=7); ax_b.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])


ani = animation.FuncAnimation(fig, animate, interval=80, cache_frame_data=False)
plt.tight_layout(rect=[0, 0, 1, 0.96])

try:
    plt.show()
finally:
    ser.close()
    print("Serial closed.")
