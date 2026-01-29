import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import matplotlib
import os

FIGURES_DIR = 'figures'
if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)

matplotlib.rcParams['text.usetex'] = True
matplotlib.use('Agg')
sns.set(font_scale=2.3)
sns.set_style("whitegrid")

width = 4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}
plt.rc('font', **font)

# Parameters
u = 1.15  # per-block fee multiplier
b = 9     # blobs per block
rho0 = 1  # initial blob base fee in wei

# Cost in ETH for one block at initial fee
c0_eth = b * (2**17) * rho0 * 1e-18

# Budgets of interest and full range
budgets_interest = [1, 10, 100]
budgets_eth = np.arange(1, 101, 1)

# Calculate disruption for interest budgets
print("DA-Saturation Disruption Summary:")
print(f"{'Budget (ETH)':>12} | {'Attack Blocks (k)':>17} | {'Total Disruption (2k)':>22}")
print("-" * 58)
for B in budgets_interest:
    cumulative = 0.0
    k = 0
    while cumulative + c0_eth * (u**k) <= B:
        cumulative += c0_eth * (u**k)
        k += 1
    print(f"{B:12d} | {k:17d} | {2*k:22d}")

# Re-generate plot and save
blocks_attack = []
blocks_total = []

for B in budgets_eth:
    cumulative = 0.0
    k = 0
    while cumulative + c0_eth * (u**k) <= B:
        cumulative += c0_eth * (u**k)
        k += 1
    blocks_attack.append(k)
    blocks_total.append(2 * k)

plt.figure()
plt.plot(budgets_eth, blocks_attack, label='Attack blocks (k)')
plt.plot(budgets_eth, blocks_total, label='Total disruption blocks (2k)', linestyle='--')
plt.xlabel("Budget (ETH)")
plt.ylabel("Number of L1 Blocks")
#plt.title("DA-Saturation Attack vs Budget with Cool-Down")
plt.ylim(0, max(blocks_total) + 10)
plt.xlim(0, 100)
plt.legend(loc='lower right')

fig = plt.gcf()
fig.set_size_inches(14,8)
axis = plt.gca()
fig.tight_layout()
fig_path = os.path.join(FIGURES_DIR, 'l1da.pdf')
plt.savefig(fig_path, bbox_inches='tight')
print(f'Plot saved to {fig_path}')
