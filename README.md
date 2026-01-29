# Rollup Transaction Fee Mechanism Pricing Measurements

Scripts for analyzing Layer 2 rollup data availability costs, compression efficiency, and DoS attack scenarios under EIP-4844 (Proto-Danksharding).

## Setup

### Prerequisites

- Python 3.7+
- LaTeX distribution (for plot generation):
  - **macOS**: `brew install --cask mactex`
  - **Ubuntu/Debian**: `sudo apt-get install texlive-latex-base texlive-fonts-recommended`

### Installation

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Calldata Compression Analysis

Analyzes calldata compression efficiency and gas costs (Appendix B -- Figure 10).
Takes almost 90 seconds.

```bash
# Generate compression quality and gas cost figures
python calldata.py --graph

# Calculate gas required to fill one blob
python calldata.py --blob 1 --zero-ratio 0.03
```

**Output**: `figures/compression_quality.pdf`, `figures/gas_vs_zero.pdf`

### 2. DoS Attack Cost Simulation

Simulates Denial-of-Service blob-fee attack costs across rollups with configurable finality delays.
The first command reproduce Figure 4, the second Figure 11.a.

```bash
# Default scenario (1-block delay, 30 min label)
python dos.py --finality-delay 1 --blocks 250 --label 150

# 20-block finality delay (10 hour label)
python dos.py --finality-delay 20 --blocks 4500 --label 3000

# 64-block finality delay (24 hour label)
python dos.py --finality-delay 64 --blocks 14400 --label 7200
```

**Output**: `figures/dos_cost_blocks_fd<delay>.pdf` showing per-block attack costs in ETH

### 3. L1 DA Saturation Attack

Models L1 data availability saturation with attack and cooldown periods.
It produces Figure 5.a.

```bash
python l1da.py
```

**Output**:
- Console: DA-Saturation disruption summary table
- `figures/l1da.pdf`: Attack blocks vs budget plot

### 4. L2 Finality Delay Attack

Models amplified finality delay attacks via batch amplification.
First command produces Figure 5.b and the second Figure 11.b.

```bash
python l2da.py
```

**Output**: `figures/l2da.pdf` showing total finality delay vs budget for vulnerable rollups

## Key Parameters

- **Blob size**: 131,072 bytes (4096 field elements × 32 bytes)
- **Target blobs**: 6 per L1 block (EIP-4844)
- **Max blobs**: 9 per L1 block
- **Blob fee multiplier**: 1.15x per block when exceeding target
- **Gas costs** (post-Pectra): 10 per zero byte, 40 per non-zero byte

## Deactivating Environment

```bash
deactivate
```

## Troubleshooting

**LaTeX errors**: Ensure LaTeX is in your PATH. Test with `which latex` (macOS/Linux) or `where latex` (Windows).

**Import errors**: Verify virtual environment is activated (prompt shows `(venv)`).
