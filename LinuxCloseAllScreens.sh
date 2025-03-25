#!/bin/bash
# Graceful shutdown script

echo "Sending shutdown signals to KwaTtsProcess..."
screen -S KwaTtsProcess -X stuff $'\003'  # Send Ctrl+C equivalent
sleep 5  # Give it time to shutdown gracefully

# Force kill if still running after 5 seconds
screen -ls | awk '/KwaTtsProcess/ {print $1}' | while read -r session; do
    echo "Force terminating session: $session"
    screen -XS "${session}" quit
done

echo "Shutdown complete"