#!/usr/bin/env python3
"""
GPU and Resource Monitoring for WhisperLiveKit

This script provides real-time monitoring of GPU and system resources 
while WhisperLiveKit is processing multiple users.

Usage:
    python monitor_gpu.py --interval 1 --log-file gpu_stats.log --port 8000

Requirements:
    pip install psutil gputil matplotlib pandas requests
"""

import argparse
import time
import datetime
import json
import os
import sys
import threading
import queue
import requests
import subprocess
import psutil
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import logging
from typing import Dict, List, Optional, Tuple
import signal
import atexit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("gpu_monitor.log")
    ]
)
logger = logging.getLogger("GPU-Monitor")

# Global variables
monitoring_data = []
exit_event = threading.Event()
stats_queue = queue.Queue()
fig = None
ax1 = None
ax2 = None
ax3 = None
line1 = None
line2 = None
line3 = None
bar = None
plot_initialized = False

class GPUStats:
    """Class to collect GPU statistics using nvidia-smi"""
    
    @staticmethod
    def get_gpu_stats() -> Dict:
        """Get current GPU statistics"""
        try:
            # Run nvidia-smi to get GPU stats
            result = subprocess.run(
                [
                    "nvidia-smi", 
                    "--query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                check=True
            )
            
            gpus = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                    
                values = [val.strip() for val in line.split(",")]
                if len(values) >= 7:
                    gpu = {
                        "index": int(values[0]),
                        "name": values[1],
                        "temperature": float(values[2]),
                        "gpu_utilization": float(values[3]),
                        "memory_utilization": float(values[4]),
                        "memory_total": float(values[5]),
                        "memory_used": float(values[6]),
                        "memory_free": float(values[5]) - float(values[6])
                    }
                    gpus.append(gpu)
            
            return {
                "timestamp": time.time(),
                "datetime": datetime.datetime.now().isoformat(),
                "gpus": gpus,
                "gpu_count": len(gpus)
            }
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running nvidia-smi: {e}")
            return {
                "timestamp": time.time(),
                "datetime": datetime.datetime.now().isoformat(),
                "gpus": [],
                "gpu_count": 0,
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"Error getting GPU stats: {e}")
            return {
                "timestamp": time.time(),
                "datetime": datetime.datetime.now().isoformat(),
                "gpus": [],
                "gpu_count": 0,
                "error": str(e)
            }

class SystemStats:
    """Class to collect system statistics using psutil"""
    
    @staticmethod
    def get_system_stats() -> Dict:
        """Get current system resource statistics"""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            return {
                "timestamp": time.time(),
                "cpu_percent": cpu_percent,
                "memory_total": memory.total / (1024 ** 3),  # GB
                "memory_used": memory.used / (1024 ** 3),    # GB
                "memory_free": memory.available / (1024 ** 3),  # GB
                "memory_percent": memory.percent
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {
                "timestamp": time.time(),
                "cpu_percent": 0,
                "memory_total": 0,
                "memory_used": 0,
                "memory_free": 0,
                "memory_percent": 0,
                "error": str(e)
            }

class WhisperLiveKitMonitor:
    """Monitor WhisperLiveKit server status and active connections"""
    
    def __init__(self, server_url: str):
        """Initialize with server URL"""
        self.server_url = server_url
        if not self.server_url.startswith(('http://', 'https://')):
            self.server_url = f"http://{self.server_url}"
        
    def get_server_status(self) -> Dict:
        """Get WhisperLiveKit server status"""
        try:
            # Get server status from the /status endpoint
            response = requests.get(f"{self.server_url}/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return {
                    "timestamp": time.time(),
                    "active_connections": data.get("active_connections", 0),
                    "server_config": data.get("server_config", {}),
                    "version": data.get("version", "unknown"),
                    "status": "running"
                }
            else:
                return {
                    "timestamp": time.time(),
                    "active_connections": 0,
                    "status": "error",
                    "error": f"Status code {response.status_code}"
                }
        except requests.RequestException as e:
            logger.warning(f"Error connecting to WhisperLiveKit server: {e}")
            return {
                "timestamp": time.time(),
                "active_connections": 0,
                "status": "error",
                "error": str(e)
            }

def monitor_resources(interval: float, server_url: str, log_file: Optional[str] = None):
    """
    Main monitoring function that collects GPU, system and server stats
    at regular intervals and saves them to a file
    """
    logger.info(f"Starting resource monitoring with {interval}s interval")
    
    gpu_monitor = GPUStats()
    system_monitor = SystemStats()
    server_monitor = WhisperLiveKitMonitor(server_url)
    
    start_time = time.time()
    
    try:
        while not exit_event.is_set():
            # Collect stats
            timestamp = time.time()
            gpu_stats = gpu_monitor.get_gpu_stats()
            system_stats = system_monitor.get_system_stats()
            server_stats = server_monitor.get_server_status()
            
            # Combine stats
            stats = {
                "timestamp": timestamp,
                "elapsed_seconds": timestamp - start_time,
                "datetime": datetime.datetime.now().isoformat(),
                "gpu": gpu_stats,
                "system": system_stats,
                "server": server_stats
            }
            
            # Queue for plotting
            stats_queue.put(stats)
            
            # Log stats to console
            active_users = server_stats.get("active_connections", 0)
            gpu_info = ""
            if gpu_stats["gpu_count"] > 0:
                gpu = gpu_stats["gpus"][0]  # First GPU
                gpu_info = f"GPU: {gpu['gpu_utilization']:.1f}% | Memory: {gpu['memory_used']:.1f}/{gpu['memory_total']:.1f} GB ({(100 * gpu['memory_used'] / gpu['memory_total']):.1f}%)"
            
            logger.info(
                f"Users: {active_users} | {gpu_info} | "
                f"CPU: {system_stats['cpu_percent']:.1f}% | RAM: {system_stats['memory_used']:.1f}/{system_stats['memory_total']:.1f} GB"
            )
            
            # Save to global list
            monitoring_data.append(stats)
            
            # Save to log file if specified
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(json.dumps(stats) + "\n")
            
            # Sleep until next interval
            time.sleep(interval)
    
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Error during monitoring: {e}")
    finally:
        logger.info(f"Monitored for {time.time() - start_time:.1f} seconds")
        exit_event.set()

def save_results(output_dir: str):
    """Save monitoring results to files"""
    if not monitoring_data:
        logger.warning("No monitoring data to save")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw data as JSON
    with open(os.path.join(output_dir, 'gpu_monitoring_data.json'), 'w') as f:
        json.dump(monitoring_data, f, indent=2)
    
    # Create DataFrame for analysis
    rows = []
    for stats in monitoring_data:
        row = {
            "timestamp": stats["timestamp"],
            "elapsed_seconds": stats["elapsed_seconds"],
            "active_users": stats["server"].get("active_connections", 0),
            "cpu_percent": stats["system"]["cpu_percent"],
            "ram_used_gb": stats["system"]["memory_used"],
            "ram_total_gb": stats["system"]["memory_total"],
            "ram_percent": stats["system"]["memory_percent"],
        }
        
        # Add GPU stats if available
        if stats["gpu"]["gpu_count"] > 0:
            for idx, gpu in enumerate(stats["gpu"]["gpus"]):
                prefix = f"gpu{idx}_"
                row.update({
                    f"{prefix}utilization": gpu["gpu_utilization"],
                    f"{prefix}memory_used_gb": gpu["memory_used"],
                    f"{prefix}memory_total_gb": gpu["memory_total"],
                    f"{prefix}memory_percent": 100 * gpu["memory_used"] / gpu["memory_total"],
                    f"{prefix}temperature": gpu["temperature"]
                })
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Save CSV
    df.to_csv(os.path.join(output_dir, 'gpu_monitoring_data.csv'), index=False)
    
    # Generate plots
    try:
        # GPU Utilization vs Users
        plt.figure(figsize=(12, 6))
        plt.plot(df["elapsed_seconds"], df["gpu0_utilization"], 'b-', label="GPU Utilization %")
        plt.plot(df["elapsed_seconds"], df["active_users"] * 10, 'r-', label="Active Users (x10)")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Percentage / Count")
        plt.title("GPU Utilization vs Active Users")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'gpu_vs_users.png'))
        
        # GPU Memory Usage
        plt.figure(figsize=(12, 6))
        plt.plot(df["elapsed_seconds"], df["gpu0_memory_used_gb"], 'g-', label="GPU Memory Used (GB)")
        plt.fill_between(df["elapsed_seconds"], df["gpu0_memory_used_gb"], alpha=0.3)
        plt.axhline(y=df["gpu0_memory_total_gb"].iloc[0], color='r', linestyle='--', label="Total GPU Memory")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Memory (GB)")
        plt.title("GPU Memory Usage Over Time")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'gpu_memory_usage.png'))
        
        # Memory usage per user
        if df["active_users"].max() > 0:
            user_counts = sorted(df["active_users"].unique())
            memory_by_users = {}
            
            for user_count in user_counts:
                user_df = df[df["active_users"] == user_count]
                if not user_df.empty:
                    memory_by_users[user_count] = user_df["gpu0_memory_used_gb"].mean()
            
            if memory_by_users:
                plt.figure(figsize=(10, 6))
                users = list(memory_by_users.keys())
                memory_values = list(memory_by_users.values())
                plt.bar(users, memory_values, color='orange')
                plt.xlabel("Number of Active Users")
                plt.ylabel("Average GPU Memory Usage (GB)")
                plt.title("GPU Memory Usage by User Count")
                plt.xticks(users)
                plt.grid(True, axis='y')
                plt.savefig(os.path.join(output_dir, 'memory_by_user_count.png'))
        
        logger.info(f"Plots saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error generating plots: {e}")

def update_plot(frame):
    """Update function for matplotlib animation"""
    global plot_initialized, line1, line2, line3, bar, ax1, ax2, ax3, fig
    
    # Get latest data from queue
    data_points = []
    try:
        while not stats_queue.empty():
            data_points.append(stats_queue.get_nowait())
    except queue.Empty:
        pass
    
    # Initialize plot elements if not already done
    if not plot_initialized:
        # Initial empty data
        line1, = ax1.plot([], [], 'b-', label="GPU Utilization %")
        line2, = ax2.plot([], [], 'g-', label="GPU Memory (GB)")
        line3, = ax3.plot([], [], 'r-', label="Users")
        
        # Set legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax3.legend(loc='lower right')
        
        plot_initialized = True
    
    # If no new data, just return existing artists
    if not data_points:
        return line1, line2, line3
    
    # Extract data
    timestamps = []
    gpu_utils = []
    gpu_mems = []
    user_counts = []
    
    for point in data_points:
        timestamps.append(point["elapsed_seconds"])
        
        # GPU utilization
        if point["gpu"]["gpu_count"] > 0:
            gpu = point["gpu"]["gpus"][0]  # First GPU
            gpu_utils.append(gpu["gpu_utilization"])
            gpu_mems.append(gpu["memory_used"])
        else:
            gpu_utils.append(0)
            gpu_mems.append(0)
        
        # User count
        user_counts.append(point["server"].get("active_connections", 0))
    
    # Get existing data
    old_x = line1.get_xdata()
    old_y1 = line1.get_ydata()
    old_y2 = line2.get_ydata()
    old_y3 = line3.get_ydata()
    
    # Use numpy arrays for data
    import numpy as np
    # Combine with new data
    if len(old_x) > 0:  # Only append if there's existing data
        new_x = np.append(old_x, timestamps)
        new_y1 = np.append(old_y1, gpu_utils)
        new_y2 = np.append(old_y2, gpu_mems)
        new_y3 = np.append(old_y3, user_counts)
    else:
        new_x = np.array(timestamps)
        new_y1 = np.array(gpu_utils)
        new_y2 = np.array(gpu_mems)
        new_y3 = np.array(user_counts)
    
    # Update lines
    line1.set_data(new_x, new_y1)
    line2.set_data(new_x, new_y2)
    line3.set_data(new_x, new_y3)
    
    # Rescale axes
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    ax3.relim()
    ax3.autoscale_view()
    
    return line1, line2, line3

def start_live_plot():
    """Initialize and start a live plot of resource usage"""
    global fig, ax1, ax2, ax3
    
    # Create figure with multiple axes
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Configure axes
    ax1.set_ylabel("GPU Utilization %")
    ax1.set_ylim(0, 100)
    ax1.grid(True)
    
    ax2.set_ylabel("GPU Memory (GB)")
    ax2.grid(True)
    
    ax3.set_ylabel("Active Users")
    ax3.set_xlabel("Time (seconds)")
    ax3.grid(True)
    
    # Set title
    fig.suptitle("WhisperLiveKit Resource Monitoring", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Create animation
    ani = FuncAnimation(fig, update_plot, interval=1000, blit=True)
    plt.show()

def signal_handler(sig, frame):
    """Handle Ctrl+C to cleanly exit"""
    logger.info("Stopping monitoring...")
    exit_event.set()
    
def cleanup():
    """Clean up resources on exit"""
    if monitoring_data:
        save_results("monitoring_results")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="GPU and Resource Monitoring for WhisperLiveKit")
    parser.add_argument("--interval", type=float, default=1.0, help="Monitoring interval in seconds")
    parser.add_argument("--log-file", type=str, default=None, help="Log file to save raw monitoring data")
    parser.add_argument("--server", type=str, default="localhost:8000", help="WhisperLiveKit server address (host:port)")
    parser.add_argument("--no-plot", action="store_true", help="Disable live plotting")
    
    args = parser.parse_args()
    
    # Register signal handler and cleanup
    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(cleanup)
    
    # Start monitoring thread
    server_url = args.server
    monitor_thread = threading.Thread(
        target=monitor_resources,
        args=(args.interval, server_url, args.log_file),
        daemon=True
    )
    monitor_thread.start()
    
    # Start live plotting if enabled
    if not args.no_plot:
        start_live_plot()
    else:
        # Wait for exit event if not plotting
        try:
            while not exit_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
    
    # Wait for monitor thread to finish
    exit_event.set()
    monitor_thread.join(timeout=5)
    
    # Save results on exit
    save_results("monitoring_results")
    logger.info("Monitoring complete, results saved to 'monitoring_results' directory")

if __name__ == "__main__":
    main()