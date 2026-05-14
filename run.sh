#!/bin/bash
mkdir -p outputs
echo "Step 1: Solving BRT..."
python solve_brt.py
echo "Step 2: Simulating..."
python simulate.py
echo "Step 3: Plotting..."
python plot.py
echo "Done! Check outputs/"