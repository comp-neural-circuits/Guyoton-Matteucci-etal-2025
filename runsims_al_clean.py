'''
code for running a series of simulation with fixed modality order (V to S)
2025 m. getz
'''

import sys
import glob
import os
import numpy as np
import simnet_al_clean as snal
import params_clean as pm


def run_single(ntrials, switch, save=True, run2=True, **kwargs):

    pm.ntrials = ntrials

    pm.extstim = 'V-const'
    pm.wmax = 1
    pm.sigw = 0.1 #width of initial Wz distribution

    #Keep original mapping on first trial
    pm.switch = False

    print('running '+ str(ntrials) + ' trials')

    # run first trial
    if kwargs:
        re_t0, ri_t0, xe_t0, Wlearn0, Wdict0, perf0, ttl0 = snal.sim(True, **kwargs)
    else:
        re_t0, ri_t0, xe_t0, Wlearn0, Wdict0, perf0, ttl0 = snal.sim(True)

    # run second trial and switch modality
    if run2:
        Wee = Wdict0['Wee']
        Wei = Wdict0['Wei']
        Wie = Wdict0['Wie']
        Wii = Wdict0['Wii']
        Wz1 = Wlearn0[:3*2*pm.dim,-1].reshape(-1,1)
        Wz2 = Wlearn0[3*2*pm.dim:,-1].reshape(-1,1)

        pm.extstim = 'S-const'
        pm.switch = switch
        Wdict = {'Wee':Wee, 'Wei':Wei, 'Wie':Wie, 'Wii':Wii, 'Wz1':Wz1, 'Wz2':Wz2}

        if kwargs:
            Wdict.update(kwargs)

        re_t1, ri_t1, xe_t1, Wlearn1, Wdict1, perf1, ttl1 = snal.sim(True, **Wdict)

    if run2 and save:
        np.savez(name, Wlearn0=Wlearn0, xe_t0=xe_t0, perf0=perf0[0], resp0=perf0[1], stim0=perf0[2], Wlearn1=Wlearn1, xe_t1=xe_t1, perf1=perf1[0], resp1=perf1[1], stim1=perf1[2])

    elif not run2 and save:
        np.savez(name, Wlearn0=Wlearn0, xe_t0=xe_t0, perf0=perf0[0], resp0=perf0[1], stim0=perf0[2], ttl0=ttl0)

    if run2 and  not save:
        return Wlearn0, xe_t0, perf0, ttl0, Wlearn1, xe_t1, perf1, ttl1
    elif not run2 and not save:
        return Wlearn0, xe_t0, perf0, ttl0    
    else:
        return

def sim_loader(trial, dirname, fname, switch, **kwargs):
    '''
    handles saving and loading of simulations
    only saves full weight traces for first (assuming 0-based indexing) trial
    '''
    ntrials = 100 #140 used in paper

    Wlearn0, xe_t0, perf0, ttl0, Wlearn1, xe_t1, perf1, ttl1 = run_single(ntrials, switch=switch, save=False, **kwargs)

    if trial == 0: #save the complete output weight trace
        np.savez(dirname+fname, Wlearn0=Wlearn0, xe_t0=xe_t0, perf0=perf0[0], resp0=perf0[1], stim0=perf0[2], Wlearn1=Wlearn1, xe_t1=xe_t1, perf1=perf1[0], resp1=perf1[1], stim1=perf1[2], ttl0=ttl0, ttl1=ttl1)

    else: #save only beginning and end output weights
        np.savez(dirname+fname, W0=Wlearn0[:,0], W01=Wlearn0[:,-1], xe_t0=xe_t0, perf0=perf0[0], resp0=perf0[1], stim0=perf0[2], W10=Wlearn1[:,0], W11=Wlearn1[:,-1], xe_t1=xe_t1, perf1=perf1[0], resp1=perf1[1], stim1=perf1[2], ttl0=ttl0, ttl1=ttl1)

    return

def run_trials_par(trial):
    ''' 
    uses sim_loader to prepare trial, then makes multiple calls to run_single()
    save all data in separate files

    trial: input from system, meta-trial number (see end of file)
    runs one of each trial type to recereate all presented time-to-[re]learn distributions
    '''

    path = os.getcwd()

    dirname = '/al_model/'
    afb = {'sfba':0.04} #changes AL feedback; default is 0.04

    kwargs = afb
    print('check directory exists: '+path+dirname)

    fname = 'full_cong_testtrials_sigd'
    name = fname+str(pm.sigd).split('.')[1]+'_gam'+str(pm.gam1).split('.')[0]+'_th'+str(pm.th0)+'_trial'+str(trial)
    print('running :'+name)
    switch = False #whether to switch the contingency together with the modality
    sim_loader(trial, path+dirname, name, switch, **kwargs)

    fname = 'full_incong_testtrials_sigd'
    name = fname+str(pm.sigd).split('.')[1]+'_gam'+str(pm.gam1).split('.')[0]+'_th'+str(pm.th0)+'_trial'+str(trial)
    print('running :'+name)
    switch = True
    sim_loader(trial, path+dirname, name, switch, **kwargs)

    fname = 'rl-lesion_cong_testtrials_sigd'
    name = fname+str(pm.sigd).split('.')[1]+'_gam'+str(pm.gam1).split('.')[0]+'_th'+str(pm.th0)+'_trial'+str(trial)
    print('running :'+name)
    switch = False
    kwargs.update({'kill_rl':True})
    sim_loader(trial, path+dirname, name, switch, **kwargs)

    fname = 'rl-lesion_incong_testtrials_sigd'
    name = fname+str(pm.sigd).split('.')[1]+'_gam'+str(pm.gam1).split('.')[0]+'_th'+str(pm.th0)+'_trial'+str(trial)
    print('running :'+name)
    switch = True
    kwargs.update({'kill_rl':True})
    sim_loader(trial, path+dirname, name, switch, **kwargs)

    fname = 'al-lesion_cong_testtrials_sigd'
    name = fname+str(pm.sigd).split('.')[1]+'_gam'+str(pm.gam1).split('.')[0]+'_th'+str(pm.th0)+'_trial'+str(trial)
    print('running :'+name)
    switch = False
    kwargs.update({'kill_al':True})
    sim_loader(trial, path+dirname, name, switch, **kwargs)

    fname = 'al-lesion_incong_testtrials_sigd'
    name = fname+str(pm.sigd).split('.')[1]+'_gam'+str(pm.gam1).split('.')[0]+'_th'+str(pm.th0)+'_trial'+str(trial)
    print('running :'+name)
    switch = True
    kwargs.update({'kill_al':True})
    sim_loader(trial, path+dirname, name, switch, **kwargs)
    return

if __name__=='__main__':
    # for executing multiple parallel calls
    #  we used a bash script which passed a number to this script call
    v = sys.argv
    t = int(v[1])
    run_trials_par(t)
