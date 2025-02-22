# -*- coding: utf-8 -*-
import os
import subprocess
import sqlite3
import sys
import torch

# List of required dependencies
DEPENDENCIES = [
    "numpy",
    "mpmath",
    "matplotlib",
    "torch",
    "scikit-learn",
    "psutil",
    "pandas",
    "cupy",
    "rich",
    "pynvml",
    "keyboard",
    "pygetwindow"
]

def check_python():
    """Ensure Python version is at least 3.8."""
    print("Checking Python version...")
    if sys.version_info < (3, 8):
        print("[ERROR] Python 3.8 or higher is required.")
        sys.exit(1)
    print(f"[OK] Python {sys.version.split()[0]} detected. Version OK.")

def check_dependencies():
    """Check and install missing dependencies."""
    print("Checking dependencies...")
    missing = []
    for package in DEPENDENCIES:
        try:
            if package == "scikit-learn":
                __import__("sklearn")  # Special case: import as 'sklearn'
            elif package == "pynvml":
                __import__("pynvml")  # Already correct
            elif package == "pygetwindow":
                __import__("pygetwindow")  # Already correct
            else:
                __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"[WARNING] Installing missing dependencies: {', '.join(missing)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        except subprocess.CalledProcessError:
            print(f"[ERROR] Failed to install {', '.join(missing)}.")
            sys.exit(1)
    else:
        print("[OK] All dependencies are installed.")

    print("\n[INFO] Installed Package Versions:")
    for package in DEPENDENCIES:
        try:
            if package == "scikit-learn":
                module = __import__("sklearn")
            else:
                module = __import__(package)
            version = getattr(module, "__version__", "Unknown Version")
            print(f"   - {package}: {version}")
        except ImportError:
            print(f"   - [ERROR] {package} is not installed properly!")

def check_cuda():
    """Check CUDA availability."""
    print("\nChecking CUDA availability...")
    if torch.cuda.is_available():
        print(f"[OK] CUDA is available. Detected {torch.cuda.device_count()} GPU(s).")
    else:
        print("[WARNING] CUDA not available. GPU acceleration will not be used.")

def init_database():
    """Initialize the local database for searched regions and verified zeros."""
    print("\nInitializing local database...")
    db_path = "searched_regions.db"
    if not os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE searched_regions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sigma_start REAL,
                sigma_end REAL,
                t_start REAL,
                t_end REAL,
                timestamp TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE verified_zeros (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sigma REAL,
                t REAL,
                zeta TEXT,
                timestamp TEXT
            )
        """)
        conn.commit()
        conn.close()
        print("[OK] Database initialized.")
    else:
        print("[OK] Database already exists.")

def run_test():
    """Run a short test scan to verify setup."""
    print("\nRunning test search...")
    test_cmd = [sys.executable, "zeta_ml_finder.py", "--sigma_min", "0.51", "--sigma_max", "0.52",
                "--t_min", "100", "--t_max", "200", "--num_points", "100", "--test"]
    try:
        subprocess.run(test_cmd, check=True)
        print("[OK] Test completed successfully. Check anomalies_detected.log and anomalies.csv.")
    except subprocess.CalledProcessError:
        print("[ERROR] Test run failed.")
        sys.exit(1)

def main():
    """Main setup routine."""
    print("\n[START] Starting setup...")
    check_python()
    check_dependencies()
    check_cuda()
    init_database()
    run_test()
    print("\n[OK] Setup complete! You can now run the program.")

if __name__ == "__main__":
    main()