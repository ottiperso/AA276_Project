#!/bin/bash
mkdir -p outputs/plots
mkdir -p outputs/data
mkdir -p outputs/data/sweep

echo "Step 1: Solving BRT..."
python solve_brt.py

echo "Step 2: Simulating..."
python simulate.py

echo "Step 3: Plotting..."
python plot.py

echo "Step 3b: Convergence plot..."
python plot_convergence.py

echo "Step 3c: BRT volume over time..."
python plot_brt_volume_over_time.py

echo "Step 4: BRT Validation..."
python validate_brt.py --n_samples 2000 --t_horizon 15.0 --dt 0.01
 
echo "Step 5: Thrust ratio sweep (one process per F_P to avoid OOM)..."
rm -f outputs/data/sweep/result_FP*.npy
for fp in 2.0 2.2 2.4 2.6 2.8 3.0 3.5 4.0 5.0; do
    echo "  Solving F_P=$fp..."
    python thrust_sweep.py $fp
done
python thrust_sweep.py plot

echo "Done! Check outputs/"