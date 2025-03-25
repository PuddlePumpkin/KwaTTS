#!/bin/bash
# Script to safely kill all detached screen sessions

set -euo pipefail  # Enable strict error handling

echo "Active screen sessions:"
screen -ls

read -p "Do you want to kill all DETACHED sessions? [y/N] " -n 1 -r
echo  # Move to new line

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Safely get session list and terminate
    screen -ls | awk '/Detached/ {print $1}' | while read -r session; do
        echo "Killing session: $session"
        screen -XS "${session}" quit
    done
    echo "All detached sessions terminated"
else
    echo "Aborted"
fi