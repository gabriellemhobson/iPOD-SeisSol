import numpy as np
import pickle as pkl
import pandas as pd

import pod as podtools

summary = pd.read_csv('/Users/ghobson/Documents/Research/iPOD-SeisSol/pde-examples/data/Advection_1D_k1.0_step_summary.csv')
print(summary.shape)
file_names = summary.values[:,0]
time_vals = summary.values[:,1]

snapshots = list()
for fname in file_names:
    snap = np.fromfile(fname, dtype = np.float64) # correct dtype?? Getting different results based on what I choose
    snapshots.append(snap)

print(snapshots)

controls = dict()
controls['time'] = list()
for t in time_vals:
    controls['time'].append(t)
print(controls)

# Build the POD reduced order model
pod = podtools.PODMultivariate(remove_mean=False)
pod.database_append(controls, snapshots) # see pod_base.py
pod.setup_basis() # this is giving a runtime warning
pod.setup_interpolant(rbf_type='polyh', bounds_auto=True)

print('Singular values:\n', pod.singular_values)
e = pod.get_ric()
print('Relative Information Content (RIC):\n', e)

# LOOCV measures
measure = podtools.rbf_loocv(pod, norm_type="linf")
measure = np.absolute(measure)

ordering = np.argsort(measure)
print('m[smallest][Linf] =',('%1.4e' % measure[ordering[0]]))
print('m[largest ][Linf] =',('%1.4e' % measure[ordering[-1]]))
print('measure:\n', measure)
print('snapshot index min/max:', ordering[0], ordering[-1])

measure = podtools.rbf_loocv(pod, norm_type="rms")
measure = np.absolute(measure)

ordering = np.argsort(measure)
print('m[smallest][rms] =',('%1.4e' % measure[ordering[0]]))
print('m[largest ][rms] =',('%1.4e' % measure[ordering[-1]]))
print('measure:\n', measure)
print('snapshot index min/max:', ordering[0], ordering[-1])