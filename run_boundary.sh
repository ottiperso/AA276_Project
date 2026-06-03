#!/bin/bash
mkdir -p outputs/plots/animations

echo "Step 1: Simulate boundary ICs..."
python boundary_sim.py

echo "Step 2: Animate boundary ICs..."
for ic in boundary_pz_pos boundary_pz_neg boundary_dvz_pos boundary_dvz_neg boundary_diag; do
    echo "  Animating $ic..."
    python animate_brt.py --ic $ic --skip 3 --fps 30
done

echo "Done! Check outputs/plots/animations/"