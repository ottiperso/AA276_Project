import numpy as np
import hj_reachability as hj
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm

values_all = np.load('outputs/data/values.npy')
times      = np.load('outputs/data/times.npy')

GRID_RESOLUTION = (15, 15, 15, 15, 15, 15)
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
    hj.sets.Box(
        np.array([-8., -8., -8., -8., -8., -8.]),
        np.array([ 8.,  8.,  8.,  8.,  8.,  8.])
    ),
    GRID_RESOLUTION
)

sample_min    = np.array([-8., -8., -8., -8., -8., -8.])
sample_max    = np.array([ 8.,  8.,  8.,  8.,  8.,  8.])
domain_volume = np.prod(sample_max - sample_min)
num_samples   = int(1e5)
batch_size    = int(1e3)
num_batches   = int(num_samples / batch_size)

# subsample timesteps so this doesn't take forever
step = max(1, len(times) // 20)  # ~20 points
t_indices = list(range(0, len(times), step))
if t_indices[-1] != len(times) - 1:
    t_indices.append(len(times) - 1)

volumes = []
for idx in tqdm(t_indices, desc='Computing BRT volumes'):
    interp = RegularGridInterpolator(
        ([np.array(v) for v in grid.coordinate_vectors]),
        np.array(values_all[idx]),
        bounds_error=False, fill_value=None
    )
    num_inside = 0
    for _ in range(num_batches):
        samples = np.random.uniform(low=sample_min, high=sample_max,
                                    size=(batch_size, 6))
        num_inside += np.sum(interp(samples) < 0)
    volumes.append(num_inside * domain_volume / num_samples)

t_plot = [times[i] for i in t_indices]

np.save('outputs/data/brt_volume_over_time.npy',
        np.array(list(zip(t_plot, volumes))))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(np.abs(t_plot), volumes, 'b-o', markersize=5, linewidth=2)
ax.set_xlabel('Time horizon $|t|$ (s)', fontsize=13)
ax.set_ylabel('BRT Volume (m$^6$)', fontsize=13)
ax.set_title('BRT Volume vs Time Horizon', fontsize=13)
ax.set_ylim(0, None)
ax.grid(True)
fig.tight_layout()
fig.savefig('outputs/plots/brt_volume_over_time.png')
print('Saved outputs/plots/brt_volume_over_time.png')