# Riemann Zeta Zero Search

Welcome to **RiemannZeroSearch**, a project dedicated to uncovering a counterexample to the Riemann Hypothesis (RH)—one of mathematics' greatest unsolved mysteries. This tool searches for non-trivial zeros of the Riemann zeta function ζ(s) with a real part **σ ≠ 1/2** in the critical strip **0 < σ < 1**.

The project leverages a **two-stage approach**: anomaly detection followed by zero verification. Optimized for **GPU acceleration** yet fully functional on **CPU-only systems**, it’s built to be **hardware-agnostic**, inviting collaboration from enthusiasts worldwide. Initially tested on **NVIDIA RTX 4070 and Tesla P100 GPUs**, it’s ready to scale for global participation.

## Overview

- **Mission**: Identify a zero off the critical line **σ ≠ 1/2**, scanning beyond **t ≈ 3 × 10¹²** up to **t = 10¹⁵** and beyond.
- **Method**:
  - **Anomaly Detection**: Scans regions where **|ζ(s)| < 10⁻⁵** using GPU-accelerated approximations.
  - **Zero Verification**: Confirms zeros with high precision (50 digits) to detect **σ ≠ 1/2**.
- **Interface**: A sleek **CLI GUI powered by rich**, offering real-time stats, system usage, and interactive controls.

## Getting Started

### Step 1: Prerequisites

- **Python**: Version **3.8+**.
- **Hardware**: Any system with Python support; **CUDA-enabled GPUs recommended** for speed.
- **Terminal**: Command Prompt, PowerShell, or any CLI environment.

### Step 2: Clone the Repository

```sh
git clone https://github.com/[your-username]/RiemannZeroSearch.git
cd RiemannZeroSearch
```

### Step 3: Setup Environment

Run the setup script to install dependencies and verify your system:

```sh
python zeta_setup.py
```

#### What It Does:

- Checks Python version (**≥ 3.8**).
- Installs required packages: `numpy`, `mpmath`, `matplotlib`, `torch`, `scikit-learn`, `psutil`, `pandas`, `cupy`, `rich`, `pynvml`, `keyboard`, `pygetwindow`.
- Verifies **CUDA availability** (optional).
- Initializes `searched_regions.db` for tracking.
- Runs a quick test to ensure functionality.

#### Expected Output:

```text
[START] Starting setup...
[OK] Python 3.12.0 detected. Version OK.
[OK] All dependencies are installed.
[INFO] Installed Package Versions: ...
[OK] CUDA is available. Detected 1 GPU(s).
[OK] Database initialized.
[OK] Test completed successfully.
[OK] Setup complete! You can now run the program.
```

### Step 4: Run the Search Loop

```sh
python zeta_loop.py
```

#### What Happens:

- Starts **continuous anomaly detection and verification** at full speed.
- Displays a **live GUI** with runtime, anomalies, counterexamples, CPU/GPU usage, and last scan results.

#### GUI Example:

```text
┌──────────────┬───────────┬─────────────────┐
│ Runtime      │ Anomalies │ Counterexamples │
├──────────────┼───────────┼─────────────────┤
│ 0h 0m 10s    │ 0         │ 0               │
└──────────────┴───────────┴─────────────────┘
┌────────────┬────────────────────────┐
│ CPU Usage  │ GPU Usage              │
├────────────┼────────────────────────┤
│ 25%        │ GPU 0: 50%             │
└────────────┴────────────────────────┘
[Scanning Zeta function space...] [██████████] 100% 00:00
Last Scan: 10000 points, 0 anomalies detected
```

### Interactive Controls

- **F1**: Increase points scanned per iteration.
- **F2**: Decrease points (min 1000).
- **F3**: Reset range to **t = 3.0001753329 × 10¹²** to **10¹⁵**.
- **F4**: Pause/resume the search.
- **F5**: Toggle point increment between 100 and 1000.
- **F6**: Increase zeta approximation precision.
- **Esc**: Exit gracefully.

## Detecting a Zero

### **Anomaly Detection**

Triggers when **|ζ(s)| < 10⁻⁵**, logged to `anomalies_detected.log` and `anomalies.csv`.

### **Counterexample Confirmation** (If **σ ≠ 1/2**):

- **GUI**: "Confirmed counterexample found!" with a beep.
- **Logs**: Saved to `verified_zeros.log` and `searched_regions.db`.
- **CSV**: Recorded in `counterexamples.csv`.

## Files and Functionality

- **zeta_ml_finder.py**: Detects anomalies using GPU-accelerated logarithmic sampling. Outputs `anomalies.csv`, `anomalies_detected.log`.
- **zeta_crunch.py**: Verifies zeros with high precision. Outputs `verified_zeros.log`, `counterexamples.csv`.
- **zeta_loop.py**: Runs the continuous search pipeline with a live CLI GUI.
- **zeta_setup.py**: Sets up the environment, installs dependencies, initializes the database, and runs a test.
- **searched_regions.db**: SQLite database tracking scanned and verified regions.

## Testing the Setup

Verify everything works with a **dry run**:

```sh
python zeta_setup.py  # Setup and initial test
python zeta_ml_finder.py --test  # Inject fake anomaly
python zeta_crunch.py --anomaly_file anomalies.csv --test  # Confirm fake counterexample
```

Expect beeps and log updates confirming the test counterexample **(σ = 0.55, t = 3.1 × 10¹²)**.

## Contributing

Join the hunt!

1. **Clone and Test**: Fork the repo, run locally, and explore.
2. **Enhance**: Suggest improvements (e.g., add `--N` to `zeta_ml_finder.py` for precision control).
3. **Submit**: Open pull requests—optimize zeta calculations or database features.

## Future Vision

- **Volunteer Computing**: Scale globally with a **BOINC-like model** - Date TBD
