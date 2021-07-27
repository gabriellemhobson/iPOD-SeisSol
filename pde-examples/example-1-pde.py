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

def adaptive():
    import sys
    sys.path.append("/Users/ghobson/Documents/Research/pde_solver/togit/pde_solver/")
    import pde_1D_advection as forward_model
    import pdebase as pdebase

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

    # run the forward model, this will solve the forward model and save every 200 timesteps using pdebase
    checkpoint_freq = 10
    forward_model.calling_advection_1d(checkpoint_frequency_in=checkpoint_freq)
    
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

    # Evaluate the error between the POD model and the forward model at every 200 timesteps
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
    measure_l1 = np.absolute(podtools.rbf_loocv(pod, norm_type="l1"))
    measure_l2 = np.absolute(podtools.rbf_loocv(pod, norm_type="l2"))
    measure_linf = np.absolute(podtools.rbf_loocv(pod, norm_type="linf"))
    measure_rms = np.absolute(podtools.rbf_loocv(pod, norm_type="rms"))

    # set a tolerance
    tol = 1e-1
    # find the max
    max_err = np.max(measure_l2)
    print('max_err initially',max_err)
    max_iters = 1e2
    for n in range(int(max_iters)):
        if max_err > tol:
            # locate the largest measure_l2 error
            loc = np.argmax(measure_l2)
            print('loc',loc)
            time_loc = times[loc]
            print('time_loc',time_loc)
            # add more of the checkpointed snaps close by to the basis
            # find the closest ones and append
            # for now, ignoring the order with which they are added to the basis
            if loc-1 > 0:
                if (timesteps[loc-1] not in timesteps_selected) and (times[loc-1] not in times_selected):
                    timesteps_selected.append(timesteps[loc+1])
                    times_selected.append(times[loc+1])
                    print('Checkpoint to the right of max err appended')

                    snap = pde.load_solution(n=timesteps[loc-1])
                    snap = np.array(snap)
                    snapshots.append(snap)
                    print('len(snapshots)',len(snapshots))

                    controls['time'].append(times[loc-1])
                    print(controls)
                else:
                    print('repeat')

                
            if loc+1 < len(timesteps):
                if (timesteps[loc+1] not in timesteps_selected) and (times[loc+1] not in times_selected):
                    timesteps_selected.append(timesteps[loc+1])
                    times_selected.append(times[loc+1])
                    print('Checkpoint to the right of max err appended')
                    
                    snap = pde.load_solution(n=timesteps[loc+1])
                    snap = np.array(snap)
                    snapshots.append(snap)
                    print('len(snapshots)',len(snapshots))

                    controls['time'].append(times[loc+1])
                    print(controls)
                else:
                    print('repeat')

            # Build the POD reduced order model
            del pod
            pod = podtools.PODMultivariate(remove_mean=False)
            pod.database_append(controls, snapshots) # this can be called multiple times, but controls must always be the same
            pod.setup_basis() 
            pod.setup_interpolant(rbf_type='polyh', bounds_auto=True)

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
            measure_l1 = np.absolute(podtools.rbf_loocv(pod, norm_type="l1"))
            measure_l2 = np.absolute(podtools.rbf_loocv(pod, norm_type="l2"))
            measure_linf = np.absolute(podtools.rbf_loocv(pod, norm_type="linf"))
            measure_rms = np.absolute(podtools.rbf_loocv(pod, norm_type="rms"))

            # recompute error and max
            max_err = np.max(measure_l2)
            print('max_err',max_err)
        elif max_err <= tol:
            print('The max error is below set tolerance')
            break 
        # rinse and repeat until all LOOCV errors are below tolerance


def excluding_ic():
    import sys
    sys.path.append("/Users/ghobson/Documents/Research/pde_solver/togit/pde_solver/")
    import pde_1D_advection as forward_model
    import pdebase as pdebase

    # run the forward model, this will solve the forward model and save every 200 timesteps using pdebase
    checkpoint_freq = 200
    #forward_model.calling_advection_1d(checkpoint_frequency_in=checkpoint_freq)
    
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

    # removing the initial condition, starting at N
    N = 4000
    start = int(N/checkpoint_freq)
    timesteps_new = timesteps[start:]
    times_new = times[start:]
    
    selected_freq = 1000
    start_selected = int(N/selected_freq)
    timesteps_selected_new = timesteps_selected[start_selected:]
    times_selected_new = times_selected[start_selected:]

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

    snapshots_new = list()
    for k in timesteps_selected_new:
        snap = pde.load_solution(n=k)
        snap = np.array(snap)
        snapshots_new.append(snap)
    print('len(snapshots_new)',len(snapshots_new))

    controls_new = dict()
    controls_new['time'] = list()
    for t in times_selected_new:
        controls_new['time'].append(t)
    print('controls_new',controls_new)

    # Build the POD reduced order model
    pod = podtools.PODMultivariate(remove_mean=False)
    pod.database_append(controls, snapshots) # this can be called multiple times, but controls must always be the same
    pod.setup_basis() # this is giving a runtime warning
    pod.setup_interpolant(rbf_type='polyh', bounds_auto=True)

    pod_new = podtools.PODMultivariate(remove_mean=False)
    pod_new.database_append(controls_new, snapshots_new) # this can be called multiple times, but controls must always be the same
    pod_new.setup_basis() # this is giving a runtime warning
    pod_new.setup_interpolant(rbf_type='polyh', bounds_auto=True)

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
    measure_l1 = np.absolute(podtools.rbf_loocv(pod, norm_type="l1"))
    measure_l2 = np.absolute(podtools.rbf_loocv(pod, norm_type="l2"))
    measure_linf = np.absolute(podtools.rbf_loocv(pod, norm_type="linf"))
    measure_rms = np.absolute(podtools.rbf_loocv(pod, norm_type="rms"))

    # now for the errors excluding the initial conditions
    err_l1_new = np.zeros((len(times_new)))
    err_l2_new = np.zeros((len(times_new)))
    err_linf_new = np.zeros((len(times_new)))
    err_rms_new = np.zeros((len(times_new)))
    m = 0
    for k in range((len(timesteps_new))):
        x0_new = pod_new.evaluate([times_new[k]])
        forward_sol_new = np.array(pde.load_solution(n=timesteps_new[k]))
        diff_new = forward_sol_new - x0_new
        err_l1_new[m] = eval_error(diff_new,norm_type="l1")
        err_l2_new[m] = eval_error(diff_new,norm_type="l2")
        err_linf_new[m] = eval_error(diff_new,norm_type="linf")
        err_rms_new[m] = eval_error(diff_new,norm_type="rms")
        m += 1

    # compute all the LOOCV errors
    measure_l1_new = np.absolute(podtools.rbf_loocv(pod_new, norm_type="l1"))
    measure_l2_new = np.absolute(podtools.rbf_loocv(pod_new, norm_type="l2"))
    measure_linf_new = np.absolute(podtools.rbf_loocv(pod_new, norm_type="linf"))
    measure_rms_new = np.absolute(podtools.rbf_loocv(pod_new, norm_type="rms"))


    # colors for plotting
    color_1 = (213/255,29/255,38/255)
    color_2 = (251/255,173/255,104/255)
    color_3 = (49/255,124/255,180/255)
    color_4 = (94/255,63/255,151/255)
    color_5 = (17/255,139/255,59/255)
    color_6 = (165/255,97/255,36/255)

    # semilogy plot with restricted ylim
    fig, (ax0,ax1,ax2,ax3) = plt.subplots(4,1)
    ax0.plot(times_selected,measure_l1,marker='*',c=color_4)
    ax0.plot(times_selected_new,measure_l1_new,marker='^',c='k')
    ax0.plot(times,err_l1,'.',c=color_4)
    ax0.plot(times_new,err_l1_new,'.',c='k')
    ax0.set_xlabel('t')
    ax0.set_ylabel('Log(Error)')
    ax0.set_yscale('log')
    ax0.set_ylim(1e-5,5e1)
    ax0.legend(('LOO l1, including ic','LOO l1, excluding ic','l1, including ic','l1, excluding ic'))

    ax1.plot(times_selected,measure_l2,marker='*',c=color_1)
    ax1.plot(times_selected_new,measure_l2_new,marker='^',c='k')
    ax1.plot(times,err_l2,'.',c=color_1)
    ax1.plot(times_new,err_l2_new,'.',c='k')
    ax1.set_xlabel('t')
    ax1.set_ylabel('Log(Error)')
    ax1.set_yscale('log')
    ax1.set_ylim(1e-5,5e1)
    ax1.legend(('LOO l2, including ic','LOO l2, excluding ic','l2, including ic','l2, excluding ic'))

    ax2.plot(times_selected,measure_linf,marker='*',c=color_2)
    ax2.plot(times_selected_new,measure_linf_new,marker='^',c='k')
    ax2.plot(times,err_linf,'.',c=color_2)
    ax2.plot(times_new,err_linf_new,'.',c='k')
    ax2.set_xlabel('t')
    ax2.set_ylabel('Log(Error)')
    ax2.set_yscale('log')
    ax2.set_ylim(1e-5,5e1)
    ax2.legend(('LOO linf, including ic','LOO linf, excluding ic','linf, including ic','linf, excluding ic'))

    ax3.plot(times_selected,measure_rms,marker='*',c=color_3)
    ax3.plot(times_selected,measure_rms,marker='^',c='k')
    ax3.plot(times,err_rms,'.',c=color_3)
    ax3.plot(times_new,err_rms_new,'.',c='k')
    ax3.set_xlabel('t')
    ax3.set_ylabel('Log(Error)')
    ax3.set_yscale('log')
    ax3.set_ylim(1e-5,5e1)
    ax3.legend(('LOO rms, including ic','LOO rms, excluding ic','rms, including ic','rms, excluding ic'))

    #fig.savefig('errors_excluding_ic_1000.png',dpi=400)
    plt.show()

if __name__ == '__main__':
    # main_func()
    # adaptive()
    excluding_ic()
