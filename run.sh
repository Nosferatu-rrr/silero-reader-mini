#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
nohup python3 main.py > log.txt 2>&1 &
