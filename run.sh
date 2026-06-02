#!/bin/bash
mkdir -p outputs/plots
mkdir -p outputs/data
echo "Step 1: Solving BRT..."
python solve_brt.py
echo "Step 2: Simulating..."
python simulate.py
echo "Step 3: Plotting..."
python plot.py

echo "Step 4: Thrust ratio sweep (one process per F_P to avoid OOM)..."
for fp in 2.0 2.2 2.4 2.6 2.8 3.0 3.5 4.0 5.0; do
    echo "  Solving F_P=$fp..."
    python thrust_sweep.py $fp
done
echo "  Plotting sweep results..."
python thrust_sweep.py plot

echo "Done! Check outputs/"