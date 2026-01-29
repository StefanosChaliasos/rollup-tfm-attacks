"""
Model the amplified finality attack

Notet that we make worst case assumptions for the attacker.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import matplotlib
import math
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
L1_BLOCK_TIME = 12.0            # seconds per L1 block
u = 1.15                        # blob fee multiplier per saturated L1 block
rho0 = 1.0                      # initial blob base fee in wei
eth_per_wei = 1e-18             # conversion factor from wei to ETH

# L2 calldata and tx gas parameters
GAS_PER_BLOB_CALLDATA = 5_124_000  # gas units per blob calldata
GAS_PER_BLOB_CALLDATA_ERA = 2**17 * 10 # gas units per blob calldata in era
GAS_PER_TX = 21_000                # gas units per transaction
L2_PRIORITY_FEE = 0.2  # Gwei
# Convert Gwei to ETH per gas: 1 Gwei = 1e-9 ETH per gas
L2_FEE_PER_GAS_ETH = L2_PRIORITY_FEE * 1e-9

# L1 commit cost: gas per batch commit, assume uses L1 base fee in ETH
# We'll group commit cost per batch in g_commit gas, then multiplied by L1 base fee of 1 wei (converted to ETH)
L1_BASE_FEE_WEI = 1000000000  # wei per gas -- 1Gwei
L1_BASE_FEE_ETH_PER_GAS = L1_BASE_FEE_WEI * eth_per_wei

FIGURES_DIR = 'figures'
if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)

# Rollup configurations: block_time, block_blob_limit, batch_blob_limit, commit_gas
ROLLUPS = {
    "Scroll":             {"bt": 3.0,  "batch_limit": 6, "block_limit": 6, "commit": 75_840,  "tx_fill": 1, "fee": 0.039},
    "ZKsync Era":         {"bt": 1.0,  "batch_limit": 1, "block_limit": 1, "commit": 232_524, "tx_fill": 1, "fee": 0.045},
    # Arbitrum after speed limit has bt=1, block_limit=batch_limit=1.2 (~6.2M gas)
    "Arbitrum":           {"bt": 1.0,  "batch_limit": 1.2, "block_limit": 1.2, "commit": 168_858, "tx_fill": 1, "fee": 0.014},
    "Base":               {"bt": 2.0,  "batch_limit": 7, "block_limit": 6, "commit": 21_000,  "tx_fill": 1, "fee": 0.002},
    "Optimism":           {"bt": 2.0,  "batch_limit": 6, "block_limit": 2, "commit": 21_000,  "tx_fill": 2, "fee": 0.012},
}



def compute_finality_delay(rollup, B_eth, params, cover_cost=False, finality_delay=1):
    # Compute L2 blocks per L1 block
    r = L1_BLOCK_TIME / params["bt"]
    # Compute T_blobs and T_batches per L1 interval
    # T_blobs is the number of blobs per L1 interval
    T_blobs = r * params["block_limit"]
    # T_batches is the number of batches we can create per L1 interval
    T_batches = int(T_blobs // params["batch_limit"])

    # If no amplification (<=1 batch), skip
    if T_batches <= 1:
        return np.nan
    
    # Constant cost per interval: calldata + txs + commit
    if rollup.startswith("ZKsync Era"):
        cost_calldata_eth = T_blobs * GAS_PER_BLOB_CALLDATA_ERA * params["fee"] * 1e-9
    else:
        cost_calldata_eth = T_blobs * GAS_PER_BLOB_CALLDATA * L2_FEE_PER_GAS_ETH
    cost_txs_eth = (r * 1) * GAS_PER_TX * L2_FEE_PER_GAS_ETH  # assume 1 tx per L2 block for simplicity
    cost_commit_eth = T_batches * params["commit"] * L1_BASE_FEE_ETH_PER_GAS
    C_const = cost_calldata_eth + cost_txs_eth + cost_commit_eth

    # Blob cost evolves with u^i
    # c0_blob_eth is the initial blob fee cost in ETH that the attacker pays in the first L1 interval
    c0_blob_eth = T_blobs * (2**17) * rho0 * eth_per_wei  # initial blob fee cost in ETH

    # Find max k such that sum_{i=0..k-1}(c0_blob_eth*u^i + C_const) <= B_eth
    cumulative = 0.0
    k = 0

    blob_fees_received = 0
    blob_fees_paid = 0

    # Let's simulate the attack period in the L2.
    blocks_passed_since_update = 0
    cur_rho = rho0
    l1_rho = rho0
    l1_rho_prices = []

    while True:
        blocks_passed_since_update += 1
        try:
            l1_rho = u**k * rho0
        except:
            pass
        l1_rho_prices.append(l1_rho)
        if blocks_passed_since_update == finality_delay:
            blocks_passed_since_update = 0
            cur_rho = l1_rho_prices.pop(0)
            cost_i = c0_blob_eth * cur_rho + C_const
        else:
            cost_i = c0_blob_eth * cur_rho + C_const
        if cumulative + cost_i > B_eth:
            break
        cumulative += cost_i
        blob_fees_received += c0_blob_eth * cur_rho
        k += 1

    #print("k:", k, "r:", r, "batch_blob_limit:", params["batch_blob_limit"], "T_blobs:", T_blobs, "T_batches:", T_batches)

    # Now let's simulate the attack period in the L1.
    batches_to_submit = 9 // params["batch_limit"]
    backlog_batches_per_k = T_batches - batches_to_submit
    for i in range(k):
        # We can't have 0 blocks
        # In each L1 block, we will post up to batches that include 9 blobs
        # the rest of the space is consumed by other L1 transactions
        # so the price will go up +15%
        blob_fees_paid += batches_to_submit * params["batch_limit"] * (2**17) * rho0 * eth_per_wei * (u**i)
        # If not covering the cost, we will pay the backlog in the current block's price
        if not cover_cost:
            blob_fees_paid += backlog_batches_per_k * params["batch_limit"] * (2**17) * rho0 * eth_per_wei * (u**i)

    # if COVER_COST, we will pay the backlog in the last block's price
    if cover_cost:
        blob_fees_paid += k*backlog_batches_per_k * params["batch_limit"] * (2**17) * rho0 * eth_per_wei * (u**k)
        # The interval would be k + L1 blocks required to clear the backlog
        k_total_intervals = k + ((k*backlog_batches_per_k*params["batch_limit"]) // 9)
    else:
        # We need to wait 1 L1 block each time that we pay the backlog of one L1 block interval
        l1_blocks_required_to_clear_backlog_per_k = math.ceil(backlog_batches_per_k*params["batch_limit"] / 9)
        k_total_intervals = 2*k + k*l1_blocks_required_to_clear_backlog_per_k

    #if B_eth == 1 or B_eth == 10 or B_eth == 100:
    if B_eth == 10:
        L1_attack= 398
        if cover_cost:
            print(f"{rollup}, {B_eth} ETH, {finality_delay} L1 blocks ->", "k (attack):", f"{k},", "k (total):", f"{k_total_intervals},", f"({(k_total_intervals/L1_attack):.2f}x),", "blob_fees_received:", f"{blob_fees_received:.2f},", "blob_fees_paid:", f"{blob_fees_paid:.2f},", "Profit/loss:", f"{(blob_fees_received - blob_fees_paid):.2f}")
        else:
            print(f"{rollup}, {B_eth} ETH, {finality_delay} L1 blocks ->", "k (attack):", f"{k},", "k (total):", f"{k_total_intervals},", f"({(k_total_intervals/L1_attack):.2f}x),", "blob_fees_received:", f"{blob_fees_received:.2f},", "blob_fees_paid:", f"{blob_fees_paid:.2f},", "Profit/loss:", f"{(blob_fees_received - blob_fees_paid):.2f}")
    return k_total_intervals


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cover-cost", default=False, action="store_true")
    parser.add_argument("--finality-delay", type=int, default=1)
    parser.add_argument("--budget", type=int, default=100)
    return parser.parse_args()

def main():
    args = parse_args()
    # Budgets to evaluate (in ETH)
    budgets = np.arange(1, args.budget)
    # Compute and plot
    plt.figure(figsize=(10, 6))
    for name, params in ROLLUPS.items():
        delays = [compute_finality_delay(name, B, params, args.cover_cost, args.finality_delay) for B in budgets]
        if not np.all(np.isnan(delays)):
            plt.plot(budgets, delays, label=name)

    plt.xlabel("Budget (ETH)")
    plt.ylabel("Total Finality Delay (L1 blocks)")
    #plt.title("Amplified Finality-Delay vs Budget for Vulnerable Rollups")
    plt.legend(loc='lower right')
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.xlim(0, args.budget)

    fig = plt.gcf()
    fig.set_size_inches(14,8)
    axis = plt.gca()
    fig.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'l2da.pdf')
    plt.savefig(fig_path, bbox_inches='tight')
    print(f'Plot saved to {fig_path}')

if __name__ == "__main__":
    main()