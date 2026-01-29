#!/usr/bin/env python3
"""
dos.py: Simulate Denial-of-Service blob-fee attack cost across rollups
and plot per-block attack cost over 10 L1 blocks in Ether with one line per rollup,
including cumulative cost up to 30 mins per rollup in the Legend.

Note that here we model the worst case scenario for the attacker.

Commands to run:
    - Default (label at 30 mins):
    python3 dos.py --finality-delay 1 --blocks 250 --label 150

    - Finality delay 20 blocks (label at 10 hours):
    python dos.py --finality-delay 20 --blocks 4500 --label 3000
    
    - Finality delay 64 blocks (label at 24 hours): 
    python dos.py --finality-delay 64 --blocks 14400 --label 7200
"""
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import matplotlib
import os
import argparse

matplotlib.rcParams['text.usetex'] = True
matplotlib.use('Agg')
sns.set(font_scale=2.3)
sns.set_style("whitegrid")

width = 4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}
plt.rc('font', **font)

# Constants
FIGURES_DIR = 'figures'
if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)

L1_BLOCK_TIME = 12.0    # seconds per L1 block
TARGET_BLOBS = 6
L1_MAX_BLOBS = 9
U_BLOB = 1.15           # blob fee increase multiplier per L1 block if over target
#OLD_L2_GAS_PER_BLOB = 2_080_000  # gas units to fill one blob as calldata
L2_GAS_PER_BLOB = 5_124_000  # gas units to fill one blob as calldata
ERG_PER_BYTE = 10
L2_GAS_PER_BLOB_ERA = 2**17 * ERG_PER_BYTE # cost for one blob in era
GAS_PER_TX = 21_000       # gas units per transaction
BASE_FEE = 1.0            # Gwei L1 base fee per gas unit
DELTA = 0.2               # Gwei L2 priority fee per gas unit
DELTA_ERA = 0
GAS_PER_BLOB = 2**17
INITIAL_BLOB_FEE = 1.0  # initial blob fee in wei
#INITIAL_BLOB_FEE = 100000000 # what if 0.1 Gwei
LINEA_MIN = 0.09 # Gwei
LINEA_MAX = 10   # Gwei

# Rollup parameters: block_time (s), batch_blob_limit, commit_cost (L1 gas), tx_fill, L2 gas price (Gwei)
ROLLUPS = {
    "Scroll":             {"bt": 3.0,  "batch_limit": 6, "block_limit": 6, "commit": 75_840,  "tx_fill": 1, "fee": 0.039},
    "Linea":              {"bt": 2.0,  "batch_limit": 6, "block_limit": 1, "commit": 400_030, "tx_fill": 2, "fee": 0.083},
    "ZKsync Era":         {"bt": 1.0,  "batch_limit": 1, "block_limit": 1, "commit": 232_524, "tx_fill": 1, "fee": 0.045},
    # The attack does not apply to Arbitrum due to FCFS and the speed limit
    #"Arbitrum":           {"bt": 0.25, "batch_limit": 3, "block_limit": 3, "commit": 168_858, "tx_fill": 1, "fee": 0.014},
    "Base":               {"bt": 2.0,  "batch_limit": 7, "block_limit": 6, "commit": 21_000,  "tx_fill": 1, "fee": 0.002},
    "Base-throttled":     {"bt": 2.0,  "batch_limit": 7, "block_limit": 0.16, "commit": 21_000,  "tx_fill": 70, "fee": 0.002},
    "Optimism":           {"bt": 2.0,  "batch_limit": 6, "block_limit": 2, "commit": 21_000,  "tx_fill": 2, "fee": 0.012},
    "Optimism-throttled": {"bt": 2.0,  "batch_limit": 6, "block_limit": 0.16, "commit": 21_000,  "tx_fill": 70, "fee": 0.012},
}


def simulate_rollup_blocks(params, name, blocks=250, finality_delay=1):
    """
    Simulate attack cost per L1 block for a single rollup over blocks.
    Returns a list of block costs in Gwei.
    """
    rho_blob = INITIAL_BLOB_FEE  # initial blob fee in wei
    gas_price_l1 = BASE_FEE 
    gas_price_l2 = params["fee"] + DELTA
    block_costs = []
    q = 0  # leftover blobs
    batch_limit = params["batch_limit"]
    block_limit = params["block_limit"]
    block_not_set = batch_limit == block_limit
    tx_fill = params["tx_fill"]

    # L1 blob prices FIFO queue
    l1_rho_prices = []
    l1_rho_blob = rho_blob
    # number of blocks to wait for finality
    blocks_passed_since_update = 0
    # Total costs
    total_l1_blob_fees_paid = []
    total_l2_blob_fees_received = []
    for b in range(blocks):
        r = int(L1_BLOCK_TIME / params["bt"])

        total_blobs = r * min(batch_limit, block_limit)
        total_tx = r * tx_fill
        total_batches = r if block_not_set else (q + total_blobs) // batch_limit
        q = (total_blobs + q) % batch_limit


        if name.startswith("ZKsync Era"):
            # They mentioned that priorities fees should be modeled as minimal in their case
            cost_calldata = total_blobs * L2_GAS_PER_BLOB * (params["fee"] + DELTA_ERA)
        else:
            cost_calldata = total_blobs * L2_GAS_PER_BLOB * gas_price_l2
        if name.startswith("Linea"):
            # Use Linea's formula
            # Overall the Min is the current market conditions
            variable_cost = LINEA_MAX if "Max" in name else LINEA_MIN
            cost_blob_fees = total_blobs * GAS_PER_BLOB * variable_cost
            cost_txs = 0
        else:
            cost_blob_fees = total_blobs * GAS_PER_BLOB * rho_blob * 1e-9
            cost_txs = total_tx * GAS_PER_TX * gas_price_l2

        cost_commits = total_batches * params["commit"] * gas_price_l1

        total = cost_blob_fees + cost_calldata + cost_txs + cost_commits
        block_costs.append(total)

        l1_blob_fees_paid = total_blobs * GAS_PER_BLOB * l1_rho_blob * 1e-9
        total_l1_blob_fees_paid.append(l1_blob_fees_paid)
        l2_blob_fees_received = cost_blob_fees
        total_l2_blob_fees_received.append(l2_blob_fees_received)

        blocks_passed_since_update += 1
        if total_blobs > TARGET_BLOBS:
            l1_rho_blob *= U_BLOB
            l1_rho_prices.append(l1_rho_blob)
            if blocks_passed_since_update == finality_delay:
                blocks_passed_since_update = 0
                rho_blob = l1_rho_prices.pop(0)

    return block_costs, total_l1_blob_fees_paid, total_l2_blob_fees_received


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finality-delay", type=int, default=1)
    parser.add_argument("--blocks", type=int, default=250)
    parser.add_argument("--label", type=int, default=150)
    return parser.parse_args()

def main():
    args = parse_args()
    x = list(range(1, args.blocks + 1))
    plt.figure(figsize=(10, 6))

    for name, params in ROLLUPS.items():
        costs_gwei, total_paid, total_received = simulate_rollup_blocks(
            params, name, args.blocks,args.finality_delay
        )
        costs_eth = [c * 1e-9 for c in costs_gwei]
        # Sum all costs from block 1 to 150 (or whatever label is)
        total_cost_up_to_label = sum(costs_eth[:args.label])  
        total_paid_eth_up_to_label = sum(total_paid[:args.label]) * 1e-9
        total_received_eth_up_to_label = sum(total_received[:args.label]) * 1e-9
        label = f"{name} ({total_cost_up_to_label:.3f} ETH)"
        print(
            label, 
            ", Total paid: ", 
            f"{total_paid_eth_up_to_label:.10f}", 
            "ETH, Total received: ", 
            f"{total_received_eth_up_to_label:.10f}", 
            "ETH",
            "Total difference: ",
            f"{total_received_eth_up_to_label - total_paid_eth_up_to_label:.10f}",
            "ETH")
        plt.plot(x, costs_eth, label=label)

    plt.xlabel('L1 Block Number')
    plt.ylabel('Attack Cost per Block (ETH)')
    #plt.title('Per-Block Attack Cost across Rollups')
    plt.yscale('log')
    # Show only major grid lines without dotted minor grid
    plt.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.minorticks_off()
    plt.legend(loc='best', prop={'size': 20}, frameon=True)
    plt.xlim(0, args.blocks)
    fig = plt.gcf()
    fig.set_size_inches(12,6)
    axis = plt.gca()
    fig.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'dos_cost_blocks.pdf')
    plt.savefig(fig_path, bbox_inches='tight')
    print(f'Plot saved to {fig_path}')


if __name__ == '__main__':
    main()
