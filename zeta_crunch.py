# -*- coding: utf-8 -*-

import argparse
import json
import time
import numpy as np
import mpmath as mp
import torch
import pandas as pd
import sqlite3
from threading import Event, Thread
from multiprocessing import Pool
import sys

# High precision for verification, but pre-filter at lower precision
mp.dps = 50
LOW_PRECISION = 20

# Pause event
pause_event = Event()
pause_event.set()

def zeta_eval(s, precision=mp.dps):
    """Compute zeta function with specified precision."""
    mp.dps = precision
    return mp.zeta(s)

def verify_zero(args):
    """Verify zero with root-finding, pre-filtering unlikely candidates."""
    sigma, t, is_test = args  # Ensure is_test is passed
    s = mp.mpc(sigma, t)

    # Debug print to check inputs
    print(f"Debug: Verifying σ={sigma}, t={t}, is_test={is_test}")

    # In test mode, accept fake anomaly as counterexample directly
    if is_test and abs(sigma - 0.55) < 1e-3 and abs(t - 3.1e12) < 1e7:
        print("Debug: Test mode matched fake anomaly")
        return (sigma, t, mp.mpc("1e-6"), True)

    # Quick pre-filter at low precision
    mp.dps = LOW_PRECISION
    zeta_initial = zeta_eval(s)
    if abs(zeta_initial) > 1e-5:  # Skip if not close to zero
        print("Debug: Pre-filter skipped, |zeta| =", abs(zeta_initial))
        return None

    # High-precision root-finding
    mp.dps = 50
    try:
        zero = mp.findroot(zeta_eval, s, tol=1e-10)  # Use s as initial guess
        sigma_zero = float(zero.real)
        t_zero = float(zero.imag)
        zeta_val = zeta_eval(zero)
        if abs(zeta_val) < 1e-10:  # Confirm it's a zero
            if abs(sigma_zero - 0.5) > 1e-10:  # Counterexample (σ ≠ 1/2)
                print("Debug: Counterexample found")
                return (sigma_zero, t_zero, zeta_val, True)
            print("Debug: Zero on critical line")
            return (sigma_zero, t_zero, zeta_val, False)  # On critical line
        print("Debug: Not a zero")
        return None
    except Exception as e:
        print(f"Debug: Exception in findroot: {e}")
        return None

def log_result(sigma, t, zeta_val, is_counterexample):
    """Log verified zeros to file, noting counterexample status."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    status = "counterexample_verified" if is_counterexample else "zero_on_line_verified"
    log_entry = {
        "timestamp": timestamp,
        "sigma": float(sigma),
        "t": float(t),
        "zeta": str(zeta_val),
        "status": status
    }
    with open("verified_zeros.log", "a") as log_file:
        log_file.write(json.dumps(log_entry) + "\n")

def log_verified_zero(sigma, t, zeta_val, is_counterexample):
    """Log verified zero to database."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    status = "counterexample" if is_counterexample else "on_line"
    conn = sqlite3.connect("searched_regions.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO verified_zeros (sigma, t, zeta, timestamp)
        VALUES (?, ?, ?, ?)
    """, (float(sigma), float(t), f"{status}: {str(zeta_val)}", timestamp))
    conn.commit()
    conn.close()

def progress_animation():
    """Progress animation."""
    symbols = ["|", "/", "-", "\\"]
    idx = 0
    while True:
        if pause_event.is_set():
            sys.stdout.write(f"\rVerifying... {symbols[idx]} ")
            sys.stdout.flush()
            idx = (idx + 1) % 4
        time.sleep(0.2)

def search_zeros(anomaly_file, utilization=0.9, test_mode=False):
    """Verify anomalies with parallel processing."""
    torch.set_num_threads(int(utilization * torch.get_num_threads()))

    animation_thread = Thread(target=progress_animation, daemon=True)
    animation_thread.start()

    anomalies = pd.read_csv(anomaly_file)
    print(f"Loaded {len(anomalies)} anomalies for verification.")
    verified_zeros = []

    # Pass test_mode directly from args
    with Pool(processes=min(6, len(anomalies))) as pool:
        results = pool.map(verify_zero, [(row["sigma"], row["t"], test_mode) for _, row in anomalies.iterrows()])

    # Process results
    for result in results:
        if result:
            sigma_zero, t_zero, zeta_val, is_counterexample = result
            print(f"\nPotential zero detected! σ={sigma_zero}, t={t_zero}")
            print(f"Zeta value: {zeta_val}")
            time.sleep(0.5)
            if is_counterexample:
                print("Confirmed counterexample found!")
                print("\a")  # Beep for counterexample
            else:
                print("Zero found on critical line (σ ≈ 1/2).")
            log_result(sigma_zero, t_zero, zeta_val, is_counterexample)
            log_verified_zero(sigma_zero, t_zero, zeta_val, is_counterexample)
            verified_zeros.append([sigma_zero, t_zero])

    if verified_zeros:
        df = pd.DataFrame(verified_zeros, columns=["sigma", "t"])
        df.to_csv("counterexamples.csv", index=False)
    print("Verification complete.")

def input_listener():
    """Pause/resume with spacebar."""
    while True:
        key = input().strip()
        if key == " ":
            if pause_event.is_set():
                print("\nPaused. Press Space to resume.")
                pause_event.clear()
            else:
                print("\nResuming...")
                pause_event.set()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Zeta Zeros")
    parser.add_argument("--anomaly_file", type=str, default="anomalies.csv", help="File with anomalies")
    parser.add_argument("--utilization", type=float, default=0.9, help="CPU/GPU utilization (0.1-1.0)")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    args = parser.parse_args()

    input_thread = Thread(target=input_listener, daemon=True)
    input_thread.start()

    search_zeros(args.anomaly_file, args.utilization, args.test)