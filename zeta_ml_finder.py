# -*- coding: utf-8 -*-

import argparse
import numpy as np
import cupy as cp
import pandas as pd
import torch
import json
import time
import sqlite3
from datetime import datetime

def zeta_approx_gpu_batch(sigma_vals, t_vals, N=1000):
    """Batch computation of approximate zeta function using GPU."""
    # Convert inputs to CuPy arrays
    sigma_gpu = cp.array(sigma_vals, dtype=cp.float64)
    t_gpu = cp.array(t_vals, dtype=cp.float64)
    s_gpu = sigma_gpu + 1j * t_gpu  # Complex s = σ + it

    # Compute n^-s for n = 1 to N in parallel
    n = cp.arange(1, N + 1, dtype=cp.float64)
    n_s = n[:, cp.newaxis] ** (-s_gpu)  # Broadcasting: n^-s for all s
    zeta_vals = cp.sum(n_s, axis=0)     # Sum over n for each s

    # Return results to CPU as numpy array
    return cp.asnumpy(zeta_vals)

def zeta_scan_logarithmic(sigma_range, t_min, t_max, num_points=1000, batch_size=100):
    """Logarithmic sampling based on zero density."""
    sigma_min, sigma_max = sigma_range
    t_vals = np.logspace(np.log10(t_min), np.log10(t_max), num_points)
    sigma_vals = np.linspace(sigma_min, sigma_max, num_points)
    scan_results = []

    for i in range(0, len(t_vals), batch_size):
        t_batch = t_vals[i:i + batch_size]
        sigma_batch = np.random.uniform(sigma_min, sigma_max, len(t_batch))
        if len(t_batch) == 0:
            break
        zeta_vals = zeta_approx_gpu_batch(sigma_batch, t_batch)
        z_abs = np.abs(zeta_vals)  # Use numpy abs directly on complex results
        scan_results.extend([[s, t, z] for s, t, z in zip(sigma_batch, t_batch, z_abs)])

    # Log searched region
    log_searched_region(sigma_min, sigma_max, t_min, t_max)
    return np.array(scan_results)

def log_searched_region(sigma_min, sigma_max, t_min, t_max):
    """Log searched region to database."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect("searched_regions.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO searched_regions (sigma_start, sigma_end, t_start, t_end, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (sigma_min, sigma_max, t_min, t_max, timestamp))
    conn.commit()
    conn.close()

def find_anomalies(scan_data, threshold=1e-5, test_mode=False):
    """Detect anomalies with a stricter threshold."""
    anomalies = scan_data[scan_data[:, 2] < threshold]
    print(f"Total scanned points: {len(scan_data)}")
    print(f"Total anomalies detected: {len(anomalies)}")

    if test_mode:
        fake_anomaly = [0.55, 3.1e12, 1e-6]
        anomalies = np.vstack([anomalies, fake_anomaly]) if anomalies.size else np.array([fake_anomaly])
        print("Test mode: Injected fake anomaly at σ=0.55, t=3.1e12")

    if len(anomalies) > 0:
        df = pd.DataFrame(anomalies, columns=["sigma", "t", "zeta_abs"])
        df.to_csv("anomalies.csv", index=False)
        for anomaly in anomalies:
            log_anomaly(anomaly[0], anomaly[1], anomaly[2])
        print("\a")
    return anomalies

def log_anomaly(sigma, t, zeta_abs):
    """Log detected anomalies."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "sigma": float(sigma),
        "t": float(t),
        "zeta_abs": float(zeta_abs),
        "status": "anomaly_detected"
    }
    with open("anomalies_detected.log", "a") as log_file:
        log_file.write(json.dumps(log_entry) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zeta Anomaly Detection with GPU")
    parser.add_argument("--sigma_min", type=float, default=0.51, help="Minimum sigma")
    parser.add_argument("--sigma_max", type=float, default=0.99, help="Maximum sigma")
    parser.add_argument("--t_min", type=float, default=3.0001753329e12, help="Minimum t")
    parser.add_argument("--t_max", type=float, default=1e15, help="Maximum t")
    parser.add_argument("--num_points", type=int, default=1000, help="Number of points")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    args = parser.parse_args()

    print("Scanning Zeta function space...")
    scan_data = zeta_scan_logarithmic((args.sigma_min, args.sigma_max), args.t_min, args.t_max, args.num_points)

    print("Detecting anomalies...")
    anomalies = find_anomalies(scan_data, test_mode=args.test)

    print(f"{len(anomalies)} anomalies detected!")