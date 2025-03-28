#!/bin/bash
# Graceful shutdown script for Screen sessions and Google Cloud startup processes

# ----------------------------------------
# Part 1: Terminate Screen sessions
# ----------------------------------------
echo "Sending shutdown signals to KwaTtsProcess..."
screen -S KwaTtsProcess -X stuff $'\003'  # Send Ctrl+C equivalent
sleep 5  # Allow graceful shutdown

# Force kill remaining Screen sessions
screen -ls | awk '/KwaTtsProcess/ {print $1}' | while read -r session; do
    echo "Force terminating Screen session: $session"
    screen -XS "${session}" quit
done

# ----------------------------------------
# Part 2: Terminate Google Cloud Startup Script Processes
# ----------------------------------------
echo "Stopping Google Cloud startup script processes..."

# 1. Kill the metadata script runner (if running)
sudo systemctl stop google-metadata-scripts.service

# 2. Kill any remaining child processes (e.g., your Python bot)
pkill -f "python src/KwaTTS.py"

# 3. Optional: Verify no startup processes remain
echo "Remaining Google startup processes:"
ps aux | grep -E 'google_metadata_script_runner|KwaTTS.py'

echo "Shutdown complete"