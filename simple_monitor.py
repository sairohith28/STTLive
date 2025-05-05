#!/usr/bin/env python3
"""
Simple GPU and Resource Monitoring for WhisperLiveKit

This script provides basic monitoring of GPU and system resources without requiring
a graphical display. Results are saved to CSV and text files for later analysis.

Usage:
    python simple_monitor.py --interval 1 --server localhost:8000 --duration 3600
"""

import argparse
import time
import datetime
import json
import os
import subprocess
import requests
import csv
import signal
import atexit
import logging
import sys
from typing import Dict, List, Optional

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
exit_requested = False

def get_gpu_stats() -> Dict:
    """Get current GPU statistics using nvidia-smi"""
    try:
        # Run nvidia-smi to get GPU stats
        result = subprocess.run(
            [
                "nvidia-smi", 
                "--query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free",
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
            if len(values) >= 8:
                gpu = {
                    "index": int(values[0]),
                    "name": values[1],
                    "temperature": float(values[2]),
                    "gpu_utilization": float(values[3]),
                    "memory_utilization": float(values[4]),
                    "memory_total": float(values[5]),
                    "memory_used": float(values[6]),
                    "memory_free": float(values[7])
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

def get_system_stats() -> Dict:
    """Get basic system statistics"""
    try:
        # Get memory usage using free command
        mem_info = subprocess.run(
            ["free", "-g"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the output
        mem_lines = mem_info.stdout.strip().split('\n')
        if len(mem_lines) >= 2:
            mem_values = mem_lines[1].split()
            if len(mem_values) >= 7:
                memory_total = float(mem_values[1])
                memory_used = float(mem_values[2])
                memory_free = float(mem_values[3])
                memory_percent = 100 * memory_used / memory_total if memory_total > 0 else 0
            else:
                memory_total = memory_used = memory_free = memory_percent = 0
        else:
            memory_total = memory_used = memory_free = memory_percent = 0
            
        # Get CPU usage
        cpu_info = subprocess.run(
            ["top", "-bn1"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse CPU usage
        cpu_percent = 0
        for line in cpu_info.stdout.strip().split('\n'):
            if line.startswith('%Cpu(s):'):
                parts = line.split(',')
                if len(parts) > 0:
                    try:
                        cpu_percent = float(parts[0].replace('%Cpu(s):', '').strip())
                    except ValueError:
                        cpu_percent = 0
                break
                
        return {
            "timestamp": time.time(),
            "cpu_percent": cpu_percent,
            "memory_total": memory_total,
            "memory_used": memory_used,
            "memory_free": memory_free,
            "memory_percent": memory_percent
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

def get_server_status(server_url: str) -> Dict:
    """Get WhisperLiveKit server status"""
    try:
        # Format server URL
        if not server_url.startswith(('http://', 'https://')):
            server_url = f"http://{server_url}"
            
        # Get server status from the /status endpoint
        response = requests.get(f"{server_url}/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            model_name = data.get("server_config", {}).get("model", "unknown")
            return {
                "timestamp": time.time(),
                "active_connections": data.get("active_connections", 0),
                "model": model_name,
                "version": data.get("version", "unknown"),
                "status": "running"
            }
        else:
            return {
                "timestamp": time.time(),
                "active_connections": 0,
                "model": "unknown",
                "status": "error",
                "error": f"Status code {response.status_code}"
            }
    except requests.RequestException as e:
        logger.warning(f"Error connecting to WhisperLiveKit server: {e}")
        return {
            "timestamp": time.time(),
            "active_connections": 0,
            "model": "unknown",
            "status": "error",
            "error": str(e)
        }

def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully exit"""
    global exit_requested
    logger.info("Stopping monitoring...")
    exit_requested = True

def save_monitoring_data(output_dir: str, data: List[Dict]):
    """Save monitoring data to CSV and JSON files"""
    if not data:
        logger.warning("No monitoring data to save")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save raw JSON data
    json_file = os.path.join(output_dir, f"gpu_monitor_{timestamp}.json")
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Create CSV file
    csv_file = os.path.join(output_dir, f"gpu_monitor_{timestamp}.csv")
    
    # Prepare CSV fields
    fieldnames = [
        "timestamp", "datetime", "active_users", 
        "gpu_utilization", "gpu_memory_used", "gpu_memory_total", "gpu_memory_percent",
        "cpu_percent", "memory_used_gb", "memory_total_gb", "memory_percent"
    ]
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for record in data:
            # Extract the data into a flattened structure
            row = {
                "timestamp": record.get("timestamp", 0),
                "datetime": record.get("datetime", ""),
                "active_users": record.get("server", {}).get("active_connections", 0),
                "cpu_percent": record.get("system", {}).get("cpu_percent", 0),
                "memory_used_gb": record.get("system", {}).get("memory_used", 0),
                "memory_total_gb": record.get("system", {}).get("memory_total", 0),
                "memory_percent": record.get("system", {}).get("memory_percent", 0),
            }
            
            # Add GPU data if available
            if record.get("gpu", {}).get("gpu_count", 0) > 0:
                gpu = record["gpu"]["gpus"][0]  # First GPU
                row.update({
                    "gpu_utilization": gpu.get("gpu_utilization", 0),
                    "gpu_memory_used": gpu.get("memory_used", 0),
                    "gpu_memory_total": gpu.get("memory_total", 0),
                    "gpu_memory_percent": 100 * gpu.get("memory_used", 0) / gpu.get("memory_total", 1)
                })
            else:
                row.update({
                    "gpu_utilization": 0,
                    "gpu_memory_used": 0,
                    "gpu_memory_total": 0,
                    "gpu_memory_percent": 0
                })
            
            writer.writerow(row)
    
    # Create a human-readable summary
    summary_file = os.path.join(output_dir, f"gpu_monitor_summary_{timestamp}.txt")
    
    with open(summary_file, 'w') as f:
        f.write("========= WhisperLiveKit GPU Monitoring Summary =========\n\n")
        f.write(f"Monitoring period: {data[0]['datetime']} to {data[-1]['datetime']}\n")
        f.write(f"Total data points: {len(data)}\n\n")
        
        # Calculate averages
        avg_gpu_util = sum(
            d["gpu"]["gpus"][0]["gpu_utilization"] 
            for d in data 
            if d["gpu"]["gpu_count"] > 0
        ) / len(data) if data else 0
        
        avg_gpu_mem = sum(
            d["gpu"]["gpus"][0]["memory_used"] 
            for d in data 
            if d["gpu"]["gpu_count"] > 0
        ) / len(data) if data else 0
        
        max_gpu_util = max(
            (d["gpu"]["gpus"][0]["gpu_utilization"] 
             for d in data 
             if d["gpu"]["gpu_count"] > 0),
            default=0
        )
        
        max_gpu_mem = max(
            (d["gpu"]["gpus"][0]["memory_used"] 
             for d in data 
             if d["gpu"]["gpu_count"] > 0),
            default=0
        )
        
        max_users = max(
            (d["server"]["active_connections"] for d in data),
            default=0
        )
        
        if data and data[0]["gpu"]["gpu_count"] > 0:
            total_gpu_mem = data[0]["gpu"]["gpus"][0]["memory_total"]
        else:
            total_gpu_mem = 0
        
        # Memory per user calculation
        user_memory_usage = {}
        for d in data:
            users = d["server"]["active_connections"]
            if users > 0 and d["gpu"]["gpu_count"] > 0:
                if users not in user_memory_usage:
                    user_memory_usage[users] = []
                user_memory_usage[users].append(d["gpu"]["gpus"][0]["memory_used"])
        
        # Calculate averages per user count
        avg_memory_per_users = {}
        for users, mem_values in user_memory_usage.items():
            avg_memory_per_users[users] = sum(mem_values) / len(mem_values)
        
        # Write statistics
        f.write(f"Peak active users: {max_users}\n")
        f.write(f"GPU Model: {data[0]['gpu']['gpus'][0]['name'] if data and data[0]['gpu']['gpu_count'] > 0 else 'Unknown'}\n")
        f.write(f"WhisperLiveKit Model: {data[0]['server']['model'] if data else 'Unknown'}\n\n")
        
        f.write("=== GPU Statistics ===\n")
        f.write(f"Average GPU Utilization: {avg_gpu_util:.1f}%\n")
        f.write(f"Peak GPU Utilization: {max_gpu_util:.1f}%\n")
        f.write(f"Average GPU Memory Usage: {avg_gpu_mem:.1f} GB\n")
        f.write(f"Peak GPU Memory Usage: {max_gpu_mem:.1f} GB\n")
        f.write(f"Total GPU Memory: {total_gpu_mem:.1f} GB\n\n")
        
        f.write("=== Memory Usage by User Count ===\n")
        for users in sorted(avg_memory_per_users.keys()):
            mem_usage = avg_memory_per_users[users]
            f.write(f"{users} users: {mem_usage:.1f} GB ({100 * mem_usage / total_gpu_mem:.1f}% of total)\n")
            
            # Calculate per-user memory
            if users > 0:
                per_user = mem_usage / users
                f.write(f"  → {per_user:.2f} GB per user\n")
                
                # Calculate additional memory for this user
                if users > 1 and (users - 1) in avg_memory_per_users:
                    prev_mem = avg_memory_per_users[users - 1]
                    additional = mem_usage - prev_mem
                    f.write(f"  → +{additional:.2f} GB for the {users}th user\n")
            
            f.write("\n")
        
        # Detect if memory usage increases linearly
        if len(avg_memory_per_users) >= 2:
            user_counts = sorted(avg_memory_per_users.keys())
            if len(user_counts) >= 2:
                # Calculate memory growth rate
                first_user = user_counts[0]
                last_user = user_counts[-1]
                
                if last_user > first_user:
                    growth_rate = (avg_memory_per_users[last_user] - avg_memory_per_users[first_user]) / (last_user - first_user)
                    f.write(f"Memory growth rate: {growth_rate:.2f} GB per additional user\n\n")
                    
                    # Estimate max users
                    if growth_rate > 0:
                        memory_headroom = total_gpu_mem - avg_memory_per_users[first_user]
                        estimated_max_users = first_user + int(memory_headroom / growth_rate)
                        f.write(f"Estimated maximum users (based on memory growth): ~{estimated_max_users}\n\n")
                
        # CPU usage
        avg_cpu = sum(d["system"]["cpu_percent"] for d in data) / len(data) if data else 0
        f.write(f"Average CPU Usage: {avg_cpu:.1f}%\n\n")
        
        f.write("========= End of Monitoring Summary =========\n")
    
    logger.info(f"Monitoring data saved to {output_dir}:")
    logger.info(f"  - Raw data: {json_file}")
    logger.info(f"  - CSV data: {csv_file}")
    logger.info(f"  - Summary: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Simple GPU Monitoring for WhisperLiveKit")
    parser.add_argument("--interval", type=float, default=1.0, help="Monitoring interval in seconds")
    parser.add_argument("--server", type=str, default="localhost:8000", help="WhisperLiveKit server URL")
    parser.add_argument("--duration", type=int, default=0, help="Duration to monitor in seconds (0 for indefinite)")
    parser.add_argument("--output", type=str, default="monitoring_results", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Register cleanup function
    def cleanup():
        if monitoring_data:
            save_monitoring_data(args.output, monitoring_data)
    
    atexit.register(cleanup)
    
    logger.info(f"Starting GPU monitoring (interval: {args.interval}s, server: {args.server})")
    logger.info("Press Ctrl+C to stop monitoring and save results")
    
    start_time = time.time()
    end_time = start_time + args.duration if args.duration > 0 else float('inf')
    
    try:
        while time.time() < end_time and not exit_requested:
            # Collect data
            timestamp = time.time()
            datetime_str = datetime.datetime.now().isoformat()
            
            gpu_stats = get_gpu_stats()
            system_stats = get_system_stats()
            server_stats = get_server_status(args.server)
            
            # Combine stats
            data_point = {
                "timestamp": timestamp,
                "datetime": datetime_str,
                "elapsed_seconds": timestamp - start_time,
                "gpu": gpu_stats,
                "system": system_stats,
                "server": server_stats
            }
            
            # Add to collection
            monitoring_data.append(data_point)
            
            # Print current stats
            active_users = server_stats.get("active_connections", 0)
            
            gpu_info = ""
            if gpu_stats["gpu_count"] > 0:
                gpu = gpu_stats["gpus"][0]  # First GPU
                gpu_util = gpu["gpu_utilization"]
                gpu_mem_used = gpu["memory_used"]
                gpu_mem_total = gpu["memory_total"]
                gpu_mem_percent = 100 * gpu_mem_used / gpu_mem_total if gpu_mem_total > 0 else 0
                
                gpu_info = f"GPU: {gpu_util:.1f}% | Memory: {gpu_mem_used:.1f}/{gpu_mem_total:.1f} GB ({gpu_mem_percent:.1f}%)"
            
            sys_info = f"CPU: {system_stats['cpu_percent']:.1f}% | RAM: {system_stats['memory_used']:.1f}/{system_stats['memory_total']:.1f} GB"
            
            logger.info(f"Users: {active_users} | {gpu_info} | {sys_info}")
            
            # Sleep until next interval
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Error during monitoring: {e}")
    finally:
        duration = time.time() - start_time
        logger.info(f"Monitored for {duration:.1f} seconds ({len(monitoring_data)} data points)")
        
        # Save results
        save_monitoring_data(args.output, monitoring_data)

if __name__ == "__main__":
    main()