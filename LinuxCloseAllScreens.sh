#!/bin/bash
# Graceful shutdown script for KwaTTs Discord Bot

# 1. Gracefully terminate the Screen session (Ctrl+C)
echo "Gracefully stopping KwaTtsProcess..."
screen -S KwaTtsProcess -X stuff $'\003'  # Send SIGINT (Ctrl+C)
sleep 5  # Wait for clean exit

# 2. Force-kill Screen sessions if still running
screen -ls | awk '/KwaTtsProcess/ {print $1}' | while read -r session; do
    echo "Force-killing session: $session"
    screen -XS "${session}" quit
done

# 3. Ensure Python process is dead (backup check)
pkill -f "python src/KwaTTS.py"

echo "Bot shutdown complete"