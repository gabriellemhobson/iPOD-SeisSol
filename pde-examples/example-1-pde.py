import numpy as np
import pickle as pkl
import pandas as pd

summary = pd.read_csv('/Users/ghobson/Documents/Research/iPOD-SeisSol/pde-examples/data/Advection_1D_k1.0_step_summary.csv')
print(summary.shape)
file_names = summary.values[:,0]
time_vals = summary.values[:,1]

snapshots = list()
for fname in file_names:
    print('fname',fname)
    snap = np.fromfile(fname, dtype = np.float64) # correct dtype??
    #df = pd.DataFrame(data)
    snapshots.append(snap)

# print(snapshots)

controls = dict()
controls['time'] = list()
for t in time_vals:
    controls['time'].append(t)
print(controls)