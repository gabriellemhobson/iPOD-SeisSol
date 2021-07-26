import numpy as np
import pickle as pkl
import pandas as pd
import h5py
import matplotlib.pyplot as plt

import pod as podtools

def main_func():
    import sys
    sys.path.append("/Users/ghobson/Documents/Research/pde_solver/togit/pde_solver/")
    import pde_1D_advection as forward_model
    import pdebase as pdebase

    # run the forward model, this will solve the forward model and save every 200 timesteps using pdebase
    forward_model.calling_advection_1d(checkpoint_frequency_in=20)
    
    # pde = pdebase.PDEBase('Advection_1D', ['k'], [1.0])
    pde = pdebase.PDEBaseLoadCheckpointFile('Advection_1D_k1.0_step20000.pkl') # Load a checkpoint from a file - avoids having to remember what parameter values were used.
    print(pde.checkpoints)

    # get time from pde.checkpoints to use in controls
    timesteps = []
    times = []
    for tuple in pde.checkpoints:
        timesteps.append(tuple[0])
        times.append(tuple[1])
    
    # now to load some of the solutions we saved, every 1000 timesteps
    timesteps_selected = []
    times_selected = []
    for k in range(len(timesteps)):
        if timesteps[k] % 1000 == 0:
            timesteps_selected.append(timesteps[k])
            times_selected.append(times[k])

    snapshots = list()
    for k in timesteps_selected:
        snap = pde.load_solution(n=k)
        snap = np.array(snap)
        snapshots.append(snap)
    print('len(snapshots)',len(snapshots))

    controls = dict()
    controls['time'] = list()
    for t in times_selected:
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

    err_l1 = np.zeros((len(times)))
    err_l2 = np.zeros((len(times)))
    err_linf = np.zeros((len(times)))
    err_rms = np.zeros((len(times)))
    m = 0
    for k in range((len(timesteps))):
        x0 = pod.evaluate([times[k]])
        forward_sol = np.array(pde.load_solution(n=timesteps[k]))
        diff = forward_sol - x0
        err_l1[m] = eval_error(diff,norm_type="l1")
        err_l2[m] = eval_error(diff,norm_type="l2")
        err_linf[m] = eval_error(diff,norm_type="linf")
        err_rms[m] = eval_error(diff,norm_type="rms")
        m += 1

    # compute all the LOOCV errors
    # seems like l1 isn't integrated yet
    # measure_l1 = np.absolute(podtools.rbf_loocv(pod, norm_type="l1"))
    measure_l1 = np.absolute(podtools.rbf_loocv(pod, norm_type="l1"))
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

    # semilogy plot
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(times,np.max(measure_l1)*np.ones((len(times))),linestyle='dashed',c=color_4)
    ax.plot(times,np.max(measure_l2)*np.ones((len(times))),linestyle='dashed',c=color_1)
    ax.plot(times,np.max(measure_linf)*np.ones((len(times))),linestyle='dashed',c=color_2)
    ax.plot(times,np.max(measure_rms)*np.ones((len(times))),linestyle='dashed',c=color_3)
    ax.plot(times,err_l1,'.',c=color_4)
    ax.plot(times,err_l2,'.',c=color_1)
    ax.plot(times,err_linf,'.',c=color_2)
    ax.plot(times,err_rms,'.',c=color_3)
    ax.set_xlabel('t')
    ax.set_ylabel('Log(Error)')
    ax.set_yscale('log')
    ax.legend(('sup LOO l1','sup LOO l2','sup LOO linf','sup LOO rms','l1','l2','linf','rms'))
    fig.savefig('error_comparison_with_llocv.png',dpi=400)
    plt.show()

    # semilogy plot with measures
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(times_selected,measure_l1,linestyle='dashed',c=color_4)
    ax.plot(times_selected,measure_l2,linestyle='dashed',c=color_1)
    ax.plot(times_selected,measure_linf,linestyle='dashed',c=color_2)
    ax.plot(times_selected,measure_rms,linestyle='dashed',c=color_3)
    ax.plot(times,err_l1,'.',c=color_4)
    ax.plot(times,err_l2,'.',c=color_1)
    ax.plot(times,err_linf,'.',c=color_2)
    ax.plot(times,err_rms,'.',c=color_3)
    ax.set_xlabel('t')
    ax.set_ylabel('Log(Error)')
    ax.set_yscale('log')
    ax.legend(('LOO l1','LOO l2','LOO linf','LOO rms','l1','l2','linf','rms'))
    fig.savefig('error_comparison_with_measures.png',dpi=400)
    plt.show()

    # semilogy plot with restricted ylim
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(times_selected,measure_l1,marker='*',c=color_4)
    ax.plot(times_selected,measure_l2,marker='*',c=color_1)
    ax.plot(times_selected,measure_linf,marker='*',c=color_2)
    ax.plot(times_selected,measure_rms,marker='*',c=color_3)
    ax.plot(times,err_l1,'.',c=color_4)
    ax.plot(times,err_l2,'.',c=color_1)
    ax.plot(times,err_linf,'.',c=color_2)
    ax.plot(times,err_rms,'.',c=color_3)
    ax.set_xlabel('t')
    ax.set_ylabel('Log(Error)')
    ax.set_yscale('log')
    ax.set_ylim(1e-5,1e1)
    ax.legend(('LOO l1','LOO l2','LOO linf','LOO rms','l1','l2','linf','rms'))
    fig.savefig('error_comparison_with_measures_ylimsm.png',dpi=400)
    plt.show()

if __name__ == '__main__':
    # try_1()
    # try_2()
    main_func()
