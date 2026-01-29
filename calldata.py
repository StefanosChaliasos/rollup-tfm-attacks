#!/usr/bin/env python3
"""
Script to perform the calldata analysis (Appendix B -- Compression Analysis)

To get the figures run:

    python calldata.py --graph

To get the amount of gas required to fill one blob run:

    python calldata.py --blob 1 --zero-ratio 0.03
"""
import argparse
import os
import random
import sys
import hashlib
import brotli
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import matplotlib

matplotlib.rcParams['text.usetex'] = True
matplotlib.use('Agg')
sns.set(font_scale=2.3)
sns.set_style("whitegrid")

width = 4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}
plt.rc('font', **font)

BLOB_SIZE = 4096 * 32  # 131072 bytes per blob
# Old gas cost: 4 per zero byte, 16 per non-zero byte
# Pectra gas cost: 10 per zero byte, 40 per non-zero byte
ZERO_GAS_COST = 10
NON_ZERO_GAS_COST = 40

FIGURES_DIR = 'figures'
if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)

def compute_gas_cost(data: bytes) -> int:
    return sum(ZERO_GAS_COST if b == 0 else NON_ZERO_GAS_COST for b in data)

# Parse size argument (decimal or 0x... hex)
def parse_size(size_str: str) -> int:
    if size_str.lower().startswith('0x'):
        return int(size_str, 16)
    return int(size_str)

# Generate data with given zero-ratio
def generate_data(size: int, zero_ratio: float) -> bytes:
    if zero_ratio <= 0.0:
        return os.urandom(size)
    if zero_ratio >= 1.0:
        return bytes([0]) * size
    data = bytearray(size)
    for i in range(size):
        if random.random() < zero_ratio:
            data[i] = 0
        else:
            b = os.urandom(1)[0]
            data[i] = b if b != 0 else 1
    return bytes(data)

def run_graph():
    zero_ratios = [i * 0.05 for i in range(21)]
    qualities  = [1, 3, 5, 7, 9, 11]
    reps       = 10

    # 1) Compute compression stats per quality
    reduction_means = {q: [] for q in qualities}
    for zr in zero_ratios:
       for q in qualities:
           tot = 0.0
           for _ in range(reps):
               data = generate_data(BLOB_SIZE, zr)
               comp = brotli.compress(data, quality=q)
               tot += (1 - len(comp) / BLOB_SIZE) * 100
           reduction_means[q].append(tot / reps)

    x_vals = [zr * 100 for zr in zero_ratios]

    # Figure 1: Compression vs Zero %
    fig1, ax1 = plt.subplots()
    for q in qualities:
       ax1.plot(x_vals, reduction_means[q], label=f'q={q}', zorder=3)
    ax1.set_xlabel('Zero ratio (\\%)')
    ax1.set_ylabel('Mean Brotli reduction (\\%)')
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.set_xticks(range(0, 101, 10))
    ax1.set_axisbelow(True); ax1.grid(True, zorder=0)
    ax1.legend(loc='upper left', frameon=True)
    fig1.set_size_inches(12, 6)
    fig1.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'compression_quality.pdf')
    fig1.savefig(fig_path, bbox_inches='tight')
    print(f'Figure saved to {fig_path}')

    # 2) Compute gas cost curve
    # For a single blob payload of BLOB_SIZE bytes:
    # gas = zeros*NON_ZERO_GAS_COST + nonzeros*ZERO_GAS_COST
    #     + zr*BLOB_SIZE*NON_ZERO_GAS_COST + (1-zr)*BLOB_SIZE*ZERO_GAS_COST
    gas_costs = []
    for zr in zero_ratios:
        z = zr * BLOB_SIZE
        nz = (1 - zr) * BLOB_SIZE
        gas_costs.append(z * ZERO_GAS_COST + nz * NON_ZERO_GAS_COST)

    # Figure 2: Gas cost vs Zero %
    fig2, ax2 = plt.subplots()
    ax2.plot(x_vals, gas_costs, color='tab:red', zorder=3)
    ax2.set_xlabel('Zero ratio (\\%)')
    ax2.set_ylabel('Calldata gas cost')
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 5.5*(10**6))
    ax2.set_xticks(range(0, 101, 10))
    ax2.set_yticks([i * 0.5 * (10**6) for i in range(12)])  # 0, 0.5, 1, ..., 5.5 million
    ax2.set_axisbelow(True); ax2.grid(True, zorder=0)
    fig2.set_size_inches(12, 6)
    fig2.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'gas_vs_zero.pdf')
    fig2.savefig(fig_path, bbox_inches='tight')
    print(f'Figure saved to {fig_path}')

# Run table mode: for blobs 1-9, zero_ratio=0.3, 10 reps, min reduction -> gas
def run_table():
    reps = 10
    zero_ratio = 0.03
    print("\\begin{tabular}{c r}")
    print("\\toprule")
    print("Nr Blobs & Calldata Gas (avg min) & Min Reduction (\\%)\\\\")
    print("\\midrule")
    for blobs in range(1, 10):
        comp_size = 0
        comp = 0
        for _ in range(reps):
            data = generate_data(blobs * BLOB_SIZE, zero_ratio)
            temp_comp = brotli.compress(data, quality=11)
            temp_comp_size = len(temp_comp)
            if comp_size == 0 or comp_size > temp_comp_size:
                comp_size = temp_comp_size
                comp = temp_comp
        gas = compute_gas_cost(comp)
        print(f"{blobs} & {gas} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")

# Main entrypoint
def main():
    parser = argparse.ArgumentParser(
        description='Generate bytes with tunable zero ratio and report stats.'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--size', help='Total bytes to produce (decimal or 0x-prefixed hex)')
    group.add_argument('--blob', type=int, help=f'Number of {BLOB_SIZE}-byte blobs')
    group.add_argument('--graph', action='store_true',
                        help='Run graph mode: vary zero-ratio and plot mean compression & gas')
    group.add_argument('--table', action='store_true',
                        help='Run table mode: LaTeX table for blobs 1–9')
    parser.add_argument('--zero-ratio', type=float, default=0.0,
                        help='Fraction of bytes set to zero (0.0–1.0, default 0.0)')
    parser.add_argument('--full', action='store_true',
                        help='Print only the full generated bytes (hex) and exit')
    args = parser.parse_args()

    if args.graph:
        run_graph()
        return
    if args.table:
        run_table()
        return

    # Determine total size
    if args.size:
        total_size = parse_size(args.size)
    else:
        total_size = args.blob * BLOB_SIZE

    data = generate_data(total_size, args.zero_ratio)

    if args.full:
        sys.stdout.write(data.hex())
        return

    # Print stats
    print(f"First 16 bytes: {data[:16].hex()}...")
    sha = hashlib.sha256(data).hexdigest()
    print(f"SHA-256 hash:   0x{sha}")
    zero_count = data.count(0)
    zero_pct = zero_count / total_size * 100
    print(f"Zero bytes:     {zero_count} ({zero_pct:.2f}%)")
    gas = compute_gas_cost(data)
    print(f"Gas cost:       {gas}")
    comp = brotli.compress(data, quality=11)
    comp_size = len(comp)
    reduction = (1 - comp_size / total_size) * 100
    print(f"Brotli size:    {comp_size} bytes")
    print(f"Reduction:      {reduction:.2f}%")

if __name__ == '__main__':
    main()
