import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

diffs = np.load('outputs/data/convergence.npy')
times = np.load('outputs/data/times.npy')

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(np.abs(times[1:]), diffs, 'b-o', markersize=4, linewidth=1.5)
ax.set_xlabel('Time horizon $|t|$ (s)', fontsize=13)
ax.set_ylabel('Max value change (log scale)', fontsize=13)
ax.set_title('BRT Convergence: Max $|V_{t+1} - V_t|$ per timestep', fontsize=13)
ax.grid(True, which='both')
fig.tight_layout()
fig.savefig('outputs/plots/convergence.png')
print('Saved outputs/plots/convergence.png')