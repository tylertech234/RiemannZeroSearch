# -*- coding: utf-8 -*-

import os
import time
import subprocess
import psutil
import pynvml
import keyboard
import pygetwindow as gw
from rich.console import Console, Group
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.live import Live

# Initialize Rich console
console = Console()

# Global stats and variables
start_time = time.time()
total_anomalies = 0
total_counterexamples = 0
paused = False
num_points = 10000
t_min = 3.0001753329e12
t_max = 1e15
point_increment = 1000
N_approximation = 1000
should_exit = False
last_scan_result = "Last Scan: N/A"

# Initialize NVIDIA GPU management (if available)
try:
    pynvml.nvmlInit()
    gpu_available = True
    gpu_count = pynvml.nvmlDeviceGetCount()
except pynvml.NVMLError:
    gpu_available = False
    gpu_count = 0

def run_command(cmd):
    """Run a shell command and return its output."""
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = ""
    while process.poll() is None:
        if not paused:
            line = process.stdout.readline()
            if line:
                output += line
        time.sleep(0.1)
    output += process.stdout.read()
    err = process.stderr.read()
    return output, err

def update_stats(finder_output, cruncher_output):
    """Update global stats from script outputs."""
    global total_anomalies, total_counterexamples, last_scan_result
    points = 0
    anomalies = 0
    for line in finder_output.splitlines():
        if "Total scanned points:" in line:
            points = int(line.split(":")[1].strip())
        if "Total anomalies detected:" in line:
            anomalies = int(line.split(":")[1].strip())
            total_anomalies += anomalies
    if cruncher_output and "Confirmed counterexample found!" in cruncher_output:
        total_counterexamples += 1
    last_scan_result = f"Last Scan: {points} points, {anomalies} anomalies detected"

def get_runtime():
    """Calculate runtime in hours, minutes, seconds."""
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    return hours, minutes, seconds

def get_system_usage():
    """Get CPU and GPU usage."""
    cpu_usage = psutil.cpu_percent(interval=0.1)
    gpu_usage = []
    if gpu_available:
        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_usage.append(f"GPU {i}: {util.gpu}%")
    return cpu_usage, " | ".join(gpu_usage) if gpu_usage else "No GPU detected"

def is_terminal_focused():
    """Check if the terminal window is focused."""
    try:
        active_window = gw.getActiveWindow()
        return "cmd" in active_window.title.lower() or "powershell" in active_window.title.lower() or "terminal" in active_window.title.lower()
    except Exception:
        return True

def create_display(progress):
    """Create the live display with stats and progress."""
    hours, minutes, seconds = get_runtime()
    cpu_usage, gpu_usage = get_system_usage()
    
    stats_table = Table(show_header=True, header_style="bold", width=console.width)
    stats_table.add_column("Runtime", style="cyan")
    stats_table.add_column("Anomalies", style="magenta")
    stats_table.add_column("Counterexamples", style="green")
    stats_table.add_row(f"{hours}h {minutes}m {seconds}s", str(total_anomalies), str(total_counterexamples))
    
    usage_table = Table(show_header=True, header_style="bold", width=console.width)
    usage_table.add_column("CPU Usage", style="yellow")
    usage_table.add_column("GPU Usage", style="yellow")
    usage_table.add_row(f"{cpu_usage}%", gpu_usage)
    
    status_text = f"{last_scan_result}"
    shortcuts = "Shortcuts: [F1] +Points | [F2] -Points | [F3] Reset Range | [F4] Pause/Resume | [F5] Toggle Increment | [F6] +Precision | [Esc] Exit"
    return Group(stats_table, usage_table, progress, status_text, shortcuts)

def on_esc_pressed(event):
    """Handle Esc key press to exit."""
    global should_exit
    if is_terminal_focused():
        should_exit = True
        console.print("Exiting Riemann Zero Search...", style="bold red")

def main():
    global start_time, paused, num_points, t_min, t_max, point_increment, N_approximation, should_exit

    # Register Esc key handler
    keyboard.on_press_key("esc", on_esc_pressed)

    progress = Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        console=console
    )
    
    with Live(create_display(progress), console=console, refresh_per_second=4) as live:
        while not should_exit:
            if is_terminal_focused():
                if keyboard.is_pressed("f1"):
                    num_points += point_increment
                    console.print(f"Points increased to {num_points}", style="blue")
                    time.sleep(0.2)
                if keyboard.is_pressed("f2") and num_points > point_increment:
                    num_points -= point_increment
                    console.print(f"Points decreased to {num_points}", style="blue")
                    time.sleep(0.2)
                if keyboard.is_pressed("f3"):
                    t_min, t_max = 3.0001753329e12, 1e15
                    console.print(f"Range reset to t_min={t_min}, t_max={t_max}", style="blue")
                    time.sleep(0.2)
                if keyboard.is_pressed("f4"):
                    paused = not paused
                    console.print(f"{'Paused' if paused else 'Resumed'}", style="bold yellow")
                    time.sleep(0.2)
                if keyboard.is_pressed("f5"):
                    point_increment = 1000 if point_increment == 100 else 100
                    console.print(f"Point increment set to {point_increment}", style="blue")
                    time.sleep(0.2)
                if keyboard.is_pressed("f6"):
                    N_approximation += 1000
                    console.print(f"Zeta approximation N increased to {N_approximation}", style="blue")
                    time.sleep(0.2)

            if not paused:
                if os.path.exists("anomalies.csv"):
                    os.remove("anomalies.csv")

                finder_task = progress.add_task("[cyan]Scanning Zeta function space...", total=100, visible=True)
                finder_cmd = f"python zeta_ml_finder.py --t_min {t_min} --t_max {t_max} --num_points {num_points}"
                finder_out, finder_err = run_command(finder_cmd)
                progress.update(finder_task, completed=100, visible=False)
                if finder_err:
                    console.print(f"[red]Finder error: {finder_err}")

                if os.path.exists("anomalies.csv"):
                    cruncher_task = progress.add_task("[yellow]Verifying anomalies...", total=100, visible=True)
                    cruncher_cmd = "python zeta_crunch.py --anomaly_file anomalies.csv"
                    cruncher_out, cruncher_err = run_command(cruncher_cmd)
                    progress.update(cruncher_task, completed=100, visible=False)
                    if cruncher_err:
                        console.print(f"[red]Cruncher error: {cruncher_err}")
                    update_stats(finder_out, cruncher_out)
                else:
                    update_stats(finder_out, "")

                live.update(create_display(progress))

            time.sleep(0.1)  # Minimal delay to prevent busy-waiting

if __name__ == "__main__":
    main()