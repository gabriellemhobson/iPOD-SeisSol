import numpy as np
import pickle as pkl
import pandas as pd
import h5py
import matplotlib.pyplot as plt

import pod as podtools

# Please see main_func(): the other functions are old and will soon be removed.

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

def main_func():
    import sys
    sys.path.append("/Users/ghobson/Documents/Research/pde_solver/togit/pde_solver/")
    import pde_1D_advection as forward_model
    import pdebase as pdebase

    # run the forward model, this will solve the forward model and save every 200 timesteps using pdebase
    #forward_model.calling_advection_1d()

    # having some issues contacting pde.load_solution()
    pde = pdebase.PDEBase('Advection_1D', ['k'], [1.0])
    #pde = pdebase.PDEBaseLoadCheckpointFile('Advection_1D_k1.0_step20000.pkl') # Load a checkpoint from a file - avoids having to remember what parameter values were used.
    print(pde)
    
    # now to load some of the solutions we saved, every 1000 timesteps
    timesteps = np.linspace(0,20000,21,dtype=int)
    snapshots = list()
    for k in timesteps:
        snap = pde.load_solution(n=k)
        snap = np.array(snap)
        snapshots.append(snap)
    # print(snapshots)

    controls = dict()
    controls['time'] = list()
    for t in timesteps:
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
    
    # Evaluate the error between the POD model and the forward model at every 200 timesteps

    def eval_error(diff,norm_type):
        if norm_type == "l1":
            err = np.linalg.norm(diff,ord=1)
        elif norm_type == "l2":
            err = np.linalg.norm(diff)
        elif norm_type == "linf":
            err = np.max(np.absolute(diff))
        elif norm_type == "rms":
            err = np.linalg.norm(diff) / np.sqrt(float(len(diff)))
        else: 
            print('Issue in eval_error()')
        return err

    dense_timesteps = np.linspace(0,20000,101,dtype=int)
    err_l1 = np.zeros((len(dense_timesteps)))
    err_l2 = np.zeros((len(dense_timesteps)))
    err_linf = np.zeros((len(dense_timesteps)))
    err_rms = np.zeros((len(dense_timesteps)))
    m = 0
    for k in dense_timesteps:
        x0 = pod.evaluate([k])
        forward_sol = np.array(pde.load_solution(n=k))
        diff = forward_sol - x0
        print('max(diff)',max(diff))
        err_l1[m] = eval_error(diff,norm_type="l1")
        err_l2[m] = eval_error(diff,norm_type="l2")
        err_linf[m] = eval_error(diff,norm_type="linf")
        err_rms[m] = eval_error(diff,norm_type="rms")
        m += 1

    # compute all the LOOCV errors
    # seems like l1 isn't integrated yet
    # measure_l1 = np.absolute(podtools.rbf_loocv(pod, norm_type="l1"))
    measure_l2 = np.absolute(podtools.rbf_loocv(pod, norm_type="l2"))
    measure_linf = np.absolute(podtools.rbf_loocv(pod, norm_type="linf"))
    measure_rms = np.absolute(podtools.rbf_loocv(pod, norm_type="rms"))

    # colors for plotting
    color_1 = (213/255,29/255,38/255)
    color_2 = (251/255,173/255,104/255)
    color_3 = (49/255,124/255,180/255)
    color_4 = (94/255,63/255,151/255)
    color_5 = (17/255,139/255,59/255)
    color_6 = (165/255,97/255,36/255)

    '''
    # linear axes plot
    plt.figure()
    plt.plot(dense_timesteps,np.max(measure_l2)*np.ones((len(dense_timesteps))),linestyle='dashed',c=color_1)
    plt.plot(dense_timesteps,np.max(measure_linf)*np.ones((len(dense_timesteps))),linestyle='dashed',c=color_2)
    plt.plot(dense_timesteps,np.max(measure_rms)*np.ones((len(dense_timesteps))),linestyle='dashed',c=color_3)
    plt.plot(dense_timesteps,err_l1,'.',c=color_4)
    plt.plot(dense_timesteps,err_l2,'.',c=color_1)
    plt.plot(dense_timesteps,err_linf,'.',c=color_2)
    plt.plot(dense_timesteps,err_rms,'.',c=color_3)
    plt.xlabel('timesteps')
    plt.ylabel('Error')
    plt.legend(('LOO l2','LOO linf','LOO rms','l1','l2','linf','rms'))
    plt.savefig('error_comparison_with_llocv.png',dpi=400)
    '''
    # semilogy plot
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(dense_timesteps,np.max(measure_l2)*np.ones((len(dense_timesteps))),linestyle='dashed',c=color_1)
    ax.plot(dense_timesteps,np.max(measure_linf)*np.ones((len(dense_timesteps))),linestyle='dashed',c=color_2)
    ax.plot(dense_timesteps,np.max(measure_rms)*np.ones((len(dense_timesteps))),linestyle='dashed',c=color_3)
    ax.plot(dense_timesteps,err_l1,'.',c=color_4)
    ax.plot(dense_timesteps,err_l2,'.',c=color_1)
    ax.plot(dense_timesteps,err_linf,'.',c=color_2)
    ax.plot(dense_timesteps,err_rms,'.',c=color_3)
    ax.set_xlabel('timesteps')
    ax.set_ylabel('Log(Error)')
    ax.set_yscale('log')
    ax.legend(('sup LOO l2','sup LOO linf','sup LOO rms','l1','l2','linf','rms'))
    fig.savefig('error_comparison_with_llocv.png',dpi=400)
    plt.show()

    # semilogy plot with measures
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(timesteps,measure_l2,linestyle='dashed',c=color_1)
    ax.plot(timesteps,measure_linf,linestyle='dashed',c=color_2)
    ax.plot(timesteps,measure_rms,linestyle='dashed',c=color_3)
    ax.plot(dense_timesteps,err_l1,'.',c=color_4)
    ax.plot(dense_timesteps,err_l2,'.',c=color_1)
    ax.plot(dense_timesteps,err_linf,'.',c=color_2)
    ax.plot(dense_timesteps,err_rms,'.',c=color_3)
    ax.set_xlabel('timesteps')
    ax.set_ylabel('Log(Error)')
    ax.set_yscale('log')
    ax.legend(('LOO l2','LOO linf','LOO rms','l1','l2','linf','rms'))
    fig.savefig('error_comparison_with_measures.png',dpi=400)
    plt.show()

    # semilogy plot with restricted ylim
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(timesteps,measure_l2,marker='*',c=color_1)
    ax.plot(timesteps,measure_linf,marker='*',c=color_2)
    ax.plot(timesteps,measure_rms,marker='*',c=color_3)
    ax.plot(dense_timesteps,err_l1,'.',c=color_4)
    ax.plot(dense_timesteps,err_l2,'.',c=color_1)
    ax.plot(dense_timesteps,err_linf,'.',c=color_2)
    ax.plot(dense_timesteps,err_rms,'.',c=color_3)
    ax.set_xlabel('timesteps')
    ax.set_ylabel('Log(Error)')
    ax.set_yscale('log')
    ax.set_ylim(1e-5,1e1)
    ax.legend(('LOO l2','LOO linf','LOO rms','l1','l2','linf','rms'))
    fig.savefig('error_comparison_with_measures_ylimsm.png',dpi=400)
    plt.show()

if __name__ == '__main__':
    # try_1()
    # try_2()
    main_func()
