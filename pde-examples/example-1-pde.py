import numpy as np
import pickle as pkl
import pandas as pd
import h5py
import matplotlib.pyplot as plt

import pod as podtools

def try_1():
    #summary = pd.read_csv('/Users/ghobson/Documents/Research/iPOD-SeisSol/pde-examples/data/Advection_1D_k1.0_step_summary.csv')
    summary = pd.read_csv('/Users/ghobson/Documents/Research/iPOD-SeisSol/pde-examples/data/Advection_1D_k1.0_step_pickles.csv')
    print(summary.shape)
    file_names = summary.values[:,0]
    time_vals = summary.values[:,1]
    '''
    snapshots = list()
    for fname in file_names:
        snap = np.fromfile(fname, dtype = np.float64) # correct dtype?? Getting different results based on what I choose
        snapshots.append(snap)
    '''

    snapshots = list()
    for fname in file_names:
        file = open(fname, "rb")
        snap = pkl.load(file)
        file.close()
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

def try_2():
    hf = h5py.File('/Users/ghobson/Documents/Research/iPOD-SeisSol/pde-examples/testfile.hdf5', 'r')
    snapshots = list()
    for snapname in hf.keys():
        snap = hf[(snapname)]
        snap = np.array(snap)
        snapshots.append(snap)
    #print(snapshots)

    controls = dict()
    controls['time'] = list()
    for t in hf.keys():
        t = int(t)
        controls['time'].append(t)
    print(controls)

    # Build the POD reduced order model
    pod = podtools.PODMultivariate(remove_mean=False)
    pod.database_append(controls, snapshots) # this can be called multiple times, but controls must always be the same
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

    # Evaluate the POD model at an arbitrary instant in time
    x0 = pod.evaluate([2100.0])
    #print(x0)

    plt.cla()
    plt.clf()
    plt.close()
    #plt.figure(figsize = (10,10))
    plt.plot(np.linspace(0,1,len(x0)),x0)
    plt.show()

    # Push POD data into a new H5 file
    h5f = h5py.File("pod_eval.h5", "w")
    grp = h5f.create_group("mesh0/")
    #dset = h5f.create_dataset("mydataset", (100,), dtype='i')
    dset = grp.create_dataset('pod', data=x0)
    h5f.close()


if __name__ == '__main__':
    # try_1()
    try_2()