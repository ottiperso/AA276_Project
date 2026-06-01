#!/bin/bash
mkdir -p outputs/plots
mkdir -p outputs/data
echo "Step 1: Solving BRT..."
python solve_brt.py
echo "Step 2: Simulating..."
python simulate.py
echo "Step 3: Plotting..."
python plot.py
echo "Step 4: Thrust ratio sweep..."
python thrust_sweep.py
echo "Done! Check outputs/"