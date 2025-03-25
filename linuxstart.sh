#!/bin/bash
screen -S KwaTtsProcess -d -m bash -c \
"cd ~/KwaTTS && \
source venv/bin/activate && \
sudo venv/bin/python src/KwaTTS.py"