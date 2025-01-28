'''
Asymmetric model with regions V1/S1/RL only
2025 m.getz
'''
import numpy as np
import params_clean as pm
import methods_clean as me

def gen_weights(dim, p):
    '''
    dim: half the full size of each region; assuming two tuned pops
     assume same along both dims
    '''
    Wrr = me.sub_rand_weights(dim, dim, p)
    Wrv = me.sub_rand_weights(dim, dim, p)
    Wrs = me.sub_rand_weights(dim, dim, p)
    Wvr = me.sub_rand_weights(dim, dim, p)
    Wvv = me.sub_rand_weights(dim, dim, p)
    Wvs = np.matrix(np.zeros((2*dim, 2*dim)))
    Wsr = me.sub_rand_weights(dim, dim, p)
    Wsv = np.matrix(np.zeros((2*dim, 2*dim)))
    Wss = me.sub_rand_weights(dim, dim, p)

    Wrx = np.hstack((np.hstack((Wrr, Wrv)), Wrs))
    Wvx = np.hstack((np.hstack((Wvr, Wvv)), Wvs))
    Wsx = np.hstack((np.hstack((Wsr, Wsv)), Wss))

    return np.vstack((np.vstack((Wrx, Wvx)), Wsx))

def f(x):
    ''' saturating transfer function 
        note: f(0) > 0 so baseline drive is 'built in'
    '''
    rho = 15
    alpha = 30
    c = 20
    return alpha*np.clip(np.divide(1, 1 + np.exp(-(x - c)/rho)) - np.divide(1, 1 + np.exp(c/rho)), 0, None)

def sim(learn=True, store_t=True, **kwargs):
    '''
    learn (bool) determines whether weights are updated

    optional kwargs which overwrite computed values:
        Blocks of weight matrices Wab (recurrent or output): Wee, Wei, Wie, Wii, Wzx
        inputs s1, s2

    returns: re_t: e rate over time; array
        ttl: trials to learn; int
    '''

    dt = pm.dt #msec

    #SPECIFY WEIGHT STRUCTURE
    #3 regions: RL, V1, S1, interconnected - assume I tracks E
    # Wee: Wrr, Wrv, Wrs; Wvr, Wvv, Wvs; Wsr, Wsv, Wss

    dim = pm.dim
    dim2 = 2*pm.dim
    Ne = 3*2*pm.dim
    Ni = 3*2*pm.dim
    N = Ne + Ni
    wmax = pm.wmax
    Wzmn = pm.Jz

    loadW = False

    if not loadW:
        # function places segregated assemblies along diagonal of submatrices 
        #  random connection otherwise
        Wee = pm.Jee*gen_weights(pm.dim, pm.pee)
        Wie = pm.Jie*gen_weights(pm.dim, pm.pie)
        Wei = pm.Jei*gen_weights(pm.dim, pm.pei)
        Wii = pm.Jii*gen_weights(pm.dim, pm.pii)

        ## REMOVE I PROJECTION NEURONS FROM V/S TO R
        Wei[:2*dim, 2*dim:] = 0
        Wei[2*dim:, :2*dim] = 0
        Wii[:2*dim, 2*dim:] = 0
        Wii[2*dim:, :2*dim] = 0

        ## SCALE DOWN FEEDBACK CONNECTIONS
        sfb0 = 0.4
        Wee[2*dim:2*2*dim, :2*dim] = sfb0*Wee[2*dim:2*2*dim, :2*dim] #r to v feedback
        Wie[2*dim:2*2*dim, :2*dim] = sfb0*Wie[2*dim:2*2*dim, :2*dim] #r to v fiedback
        Wee[2*2*dim:3*2*dim, :2*dim] = sfb0*Wee[2*2*dim:3*2*dim, :2*dim] #r to s feedback
        Wie[2*2*dim:3*2*dim, :2*dim] = sfb0*Wie[2*2*dim:3*2*dim, :2*dim] #r to s feedback
        
        ## SCALE UP FEEDFORWARD CONNECTIONS
        sff0 = 3.2
        Wee[:2*dim, 2*dim:] = sff0*Wee[:2*dim, 2*dim:]
        Wie[:2*dim, 2*dim:] = sff0*Wie[:2*dim, 2*dim:]

        ## SELECTIVELY ADJUST FF AND FB PROJECTIONS - V
        if 'sffv' in kwargs:
            sff = kwargs['sffv']
            Wee[:2*dim, 2*dim:4*dim] = (sff/sff0)*Wee[:2*dim, 2*dim:4*dim] #scale v to r ff connections
            Wie[:2*dim, 2*dim:4*dim] = (sff/sff0)*Wie[:2*dim, 2*dim:4*dim] #scale v to r ff connections

        if 'sfbv' in kwargs:
            sfb = kwargs['sfbv']
            Wee[2*dim:4*dim, :2*dim] = (sfb/sfb0)*Wee[2*dim:4*dim, :2*dim] #scale r to v fb connections

        if 'sffs' in kwargs:
            sff = kwargs['sffs']
            Wee[:2*dim, 4*dim:] = (sff/sff0)*Wee[:2*dim, 4*dim:] #scale s to r ff connections

        if 'sfbs' in kwargs:
            sfb = kwargs['sfbs']
            Wee[4*dim:, :2*dim] = (sfb/sfb0)*Wee[4*dim:, :2*dim] #scale r to s fb connections
            Wie[4*dim:, :2*dim] = (sfb/sfb0)*Wie[4*dim:, :2*dim]

        ## READOUT FROM ALL AREAS
        Wz1 = pm.sigw*np.random.randn(3*2*dim,1) + Wzmn
        Wz2 = pm.sigw*np.random.randn(3*2*dim,1) + Wzmn
        Wz1 = np.clip(Wz1, 0, wmax)
        Wz2 = np.clip(Wz2, 0, wmax)

    else:
        wfile = np.load('weights.npz')
        Wie = np.matrix(wfile['Wie'])
        Wee = np.matrix(wfile['Wee'])
        Wei = np.matrix(wfile['Wei'])
        Wii = np.matrix(wfile['Wii'])

        Wz1 = np.matrix(wfile['Wz1'])
        Wz2 = np.matrix(wfile['Wz2'])

    if 'Wei' in kwargs:
        Wei = kwargs['Wei']
    if 'Wie' in kwargs:
        Wie = kwargs['Wie']
    if 'Wee' in kwargs:
        Wee = kwargs['Wee']
    if 'Wii' in kwargs:
        Wii = kwargs['Wii']
    if 'Wz1' in kwargs:
        Wz1 = kwargs['Wz1']
    if 'Wz2' in kwargs:
        Wz2 = kwargs['Wz2']

    Wz11avg = Wzmn
    Wz12avg = Wzmn
    Wz13avg = Wzmn
    Wz21avg = Wzmn
    Wz22avg = Wzmn
    Wz23avg = Wzmn

    Wdict = {'Wee':Wee, 'Wei':Wei, 'Wie':Wie, 'Wii':Wii, 'Wz1':Wz1, 'Wz2':Wz2}

    taun = pm.taun

    ## WEIGHT UPDATE PARAMS
    gam1 = pm.gam1 #learning rate

    ## OUTPUT WEIGHT NORMALIZATION
    # arbitrarily normalize to initial condition
    W0z1 = np.sum(Wz1)
    W0z2 = np.sum(Wz2)

    #initial state; could use theoretical est. to avoid transient
    xe = np.zeros((Ne, 1))
    xi = np.zeros((Ni, 1))
    re = np.zeros((Ne, 1))
    ri = np.zeros((Ni, 1))

    # constant baseline input
    be = pm.b
    bi = pm.bi

    burn = pm.burn #Integration before stim presentations. units: steps
    ntrials = pm.ntrials
    t_l = 400 #Trial duration; stim and inter-stim interval. units: msec
    s_l = 200 #stim duration. units: msec
    r_l = 80 #reward/feedback duration. units: msec

    # Time-course of stimulation
    #  as binary series
    # Train with random output amplification - i.e. behavioral guessing

    # Type 1: (V)is/(S)omatosens only - alternate upper/lower randomly
    t_ldt = int(t_l/dt)
    s_ldt = int(s_l/dt)
    r_ldt = int(r_l/dt)

    T = ntrials*t_l

    # PRESENT STIMULI RANDOMLY
    n_stim = ntrials
    stim_idx1 = (np.random.rand(n_stim,1) < 0.5).astype('int')
    stim_idx2 = np.abs(stim_idx1 - 1)

    if 'stim_idx1' in kwargs:
        stim_idx1 = kwargs['stim_idx1']
    if 'stim_idx2' in kwargs:
        stim_idx2 = kwargs['stim_idx2']

    print('stim order: ',stim_idx1.T+(2*stim_idx2.T))

    # present stim and then reward afterwards
    s1 = (np.mod(np.arange(0,int(T/dt)), t_ldt) > t_ldt-s_ldt).astype('int').reshape(-1, t_ldt)
    s2 = (np.mod(np.arange(0,int(T/dt)), t_ldt) > t_ldt-s_ldt).astype('int').reshape(-1, t_ldt)
    s1 = np.multiply(s1, stim_idx1).reshape(-1,1)
    s2 = np.multiply(s2, stim_idx2).reshape(-1,1)

    ds = np.diff(s1, 1, 0, prepend=0) + np.diff(s2, 1, 0, prepend=0) #records stim onset and offset

    # IF STIM RECEIVED AS INPUT, UPDATE SEQUENCES:
    if 's1' in kwargs:
        s1 = kwargs['s1']
    if 's2' in kwargs:
        s2 = kwargs['s2']

    # align reward timing to stim (with delay s_l - r_l)
    r1 = (np.mod(np.arange(0,int(T/dt)), t_ldt) > t_ldt-r_ldt).astype('int')

    # set integration time for output variable in between stim onset and reward signal
    # Type 2: (S)om only - alternate upper/lower randomly
    # Type 3: (R)L only - alternate upper/lower randomly
    s1 =  np.vstack((np.zeros((burn, 1)), s1))
    s2 =  np.vstack((np.zeros((burn, 1)), s2))
    ds =  np.vstack((np.zeros((burn, 1)), ds))
    r1 =  np.vstack((np.zeros((burn, 1)), r1.reshape(-1,1)))

    zi_t = 0 #toggle to 1 to start integration, to 0 to stop it

    ## Set variables locally
    c_ne = pm.c_ne
    c_ni = pm.c_ni
    Tlim = burn + np.round(T/dt).astype(int)

    #time-course data
    re_t = np.zeros((Ne, np.round(Tlim/10).astype(int)))
    ri_t = np.zeros((Ni, np.round(Tlim/10).astype(int)))
    xe_t = np.zeros((Ne, np.round(Tlim/10).astype(int)))
    noise_t = np.zeros((Ne, np.round(Tlim/10).astype(int)))
    noisi_t = np.zeros((Ni, np.round(Tlim/10).astype(int)))

    if store_t:
        Wlearn = np.zeros((2*3*dim2, np.round(Tlim/10).astype(int))) #stack Wz1, Wz2
    else:
        Wlearn = np.zeros((2*3*dim2,)) #stack Wz1, Wz2

    if pm.extstim == 'V-const':
        # vis stimuli
        Iexte1 = pm.c1e*np.multiply(np.arange(0,Ne)>=dim2, np.arange(0,Ne)<dim2+dim).astype('int').reshape(-1,1)
        Iexte2 = pm.c1e*np.multiply(np.arange(0,Ne)>=dim2+dim, np.arange(0,Ne)<dim2+dim2).astype('int').reshape(-1,1)

    elif pm.extstim == 'S-const':
        # tactile stimuli
        Iexte1 = pm.c1e*np.multiply(np.arange(0,Ne)>=2*dim2, np.arange(0,Ne)<2*dim2+dim).astype('int').reshape(-1,1)
        Iexte2 = pm.c1e*np.multiply(np.arange(0,Ne)>=2*dim2+dim, np.arange(0,Ne)<3*dim2).astype('int').reshape(-1,1)

    if dim == 1:
        n_l = np.random.randn(dim, Tlim) #insure T/dt is not mis-rounded
        n_le = np.matrix(np.random.randn(1, Tlim))
        n_li = np.matrix(np.random.randn(1, Tlim))
    else:
        n_le = np.matrix(np.ones((dim, 1)))*np.matrix(np.random.randn(1, Tlim))
        n_li = np.matrix(np.ones((dim, 1)))*np.matrix(np.random.randn(1, Tlim))

    ##OUTPUT DRIVE
    z1 = 0.5
    z2 = 0.5
    z = np.array((z1, z2))
    alpha = 1/Ne

    switch = pm.switch
    reward_vec = np.zeros((3, Tlim))
    perf = [] #record if trial was successful or not based on integrated z output
    resp = [] #record of responses. 1: lick; 2: no lick

    if 'kill_rl' in kwargs:
        kill_rl = kwargs['kill_rl']
    else:
        kill_rl = False

    if kill_rl:
        Wz1[:2*pm.dim] = 0
        Wz2[:2*pm.dim] = 0
        Wee[:, :2*pm.dim] = 0
        Wee[:2*pm.dim, :] = 0
        Wie[:, :2*pm.dim] = 0
        Wie[:2*pm.dim, :] = 0
        alpha = 1/(2*pm.dim) #rescale relative to number of units read out

    ctz = 0 #integer count
    t_ct = 0 #trial counter

    z1i = 0
    z2i = 0
    c_z = pm.sigd #note: z~O(1) => z*dt~O(10^-2)
    th0 = pm.th0

    z1c = 0.25

    ttl = -1 # time to learn
    learned = False
    ncorrect = 20 #num correct in a row to consider task learned
    pcorrect = 0 #P(correct) as a moving window of len ncorrect

    # WHETHER TO STOP SIM EARLY AFTER LEARNING
    if 'stop' in kwargs:
        stop = kwargs['stop']
    else:
        stop = False

    R = -1 #for testing
    rmax = -1
    stim = -1

    while ctz < Tlim:
        #FIX THIS
        n_e = np.random.randn(Ne, 1)
        n_i = np.random.randn(Ni, 1)
        n_ce = n_le[:, ctz]
        n_ci = n_li[:, ctz]

        xe += (-xe*dt + np.sqrt(2*taun*dt)*n_e)/taun
        xi += (-xi*dt + np.sqrt(2*taun*dt)*n_i)/taun

        # UNCORRELATED ADDITIVE INPUT NOISE
        noise = np.matrix(xe)
        noisi = np.matrix(xi)

        #b IS WEAK BASELINE FOR SPONTANEOUS ACTIVITY
        Ie = be + s1[ctz]*Iexte1 + s2[ctz]*Iexte2 + Wee*re - Wei*ri + c_ne*noise
        Ii = bi + s1[ctz]*Iexte1 + s2[ctz]*Iexte2 - Wii*ri + Wie*re + c_ni*noisi

        re_ss = f(Ie)
        ri_ss = f(Ii)

        re += (dt*(-re + re_ss))/pm.taue
        ri += (dt*(-ri + ri_ss))/pm.taui

        ##linear i/o
        z1 = me.thresh(alpha*np.dot(Wz1.T, re))
        z2 = me.thresh(alpha*np.dot(Wz2.T, re))

        if not s1[ctz] and not s2[ctz]:
            #set z1c to approximate baseline z
            # snapshot of baseline value
            z1b = z1
            z2b = z2

        if ds[ctz] > 0:
            # stimulus onset
            zi_t = 1
        elif ds[ctz] < 0:
            # stimulus offset; stop any z integration
            zi_t = 0

        # if within integration time for output response
        if ctz > burn-1 and zi_t == 1:
            z1i += z1*dt + c_z*np.random.randn()
            z2i += z2*dt + c_z*np.random.randn()

            if z1i >= th0 or z2i >= th0:
                zi_t = 0
                rmax = np.argmax([z1i, z2i])
                stim = np.argmax([s1[ctz], s2[ctz]])

        ## REWARD PARADIGM
        if ctz > burn-1 and (ctz - (burn-1)) % t_ldt == 0:
            #reset output integration variables
            rmax = np.argmax([z1i, z2i])
            stim = np.argmax([s1[ctz], s2[ctz]])

            if z1i == z2i:
                print(0)
                perf.append(0)
                resp.append(0)

            elif z1i < th0 and z2i < th0:
                if switch and stim == 1:
                    print(-1)
                    perf.append(-1)
                    resp.append(0) #treat the same as not licking i.e. zi=2

                elif switch and stim == 0:
                    print(1)
                    perf.append(1)
                    resp.append(0) #treat the same as not licking i.e. zi=2

                if not switch and stim == 0:
                    print(-1)
                    perf.append(-1)
                    resp.append(0) #treat the same as not licking i.e. zi=2

                elif not switch and stim == 1:
                    print(1)
                    perf.append(1)
                    resp.append(0) #treat the same as not licking i.e. zi=2

            elif not switch:
                if np.equal(rmax, stim): #compare max response with stim 
                    print(1)
                    perf.append(1)
                    resp.append(rmax+1)

                elif not np.equal(rmax, stim):
                    print(-1)
                    perf.append(-1)
                    resp.append(rmax+1)

            elif switch:
                if np.equal(rmax, stim): #compare max response with stim 
                    print(-1)
                    perf.append(-1)
                    resp.append(rmax+1)

                elif not np.equal(rmax, stim):
                    print(1)
                    perf.append(1)
                    resp.append(rmax+1)

            z1i = 0
            z2i = 0

        #CHECK ONGOING PERFORMANCE
        if len(perf) >= ncorrect and learned == False:
            pcorrect = np.sum(np.where(np.array(perf)[-ncorrect:]==1, 1, 0))/ncorrect

            if pcorrect >= 0.75:
                # require pcorrect >= VALUE to stop sim
                # RETURN LEARNING TIME AS FIRST INSTANCE OF BOTH CORRECT
                ttl = len(perf) - ncorrect
                learned = True

                jj = int(ctz/10)
                if stop:
                    print('learned - stopping')
                    re_t = re_t[:,:jj]
                    ri_t = ri_t[:,:jj]
                    xe_t = xe_t[:,:jj]
                    Wlearn = Wlearn[:,:jj]
                    perf_out = (perf, resp, stim_idx1.T+(2*stim_idx2.T))
                    return re_t, ri_t, xe_t, Wlearn, Wdict, perf_out, ttl

        # ADJUST OUTPUT UNITS WITH REWARD SIGNAL
        if learn and (ctz > burn-1) and zi_t==0:
            z1c = z1 - z1b
            z2c = z2 - z2b

            if not switch and rmax != -1:
                if np.equal(rmax, stim): #compare max response with stim 
                    z1 += z1c*s1[ctz]
                    z2 += z2c*s2[ctz]

                elif not np.equal(rmax, stim):
                    # additive model of reward signal
                    z1 -= z1c*s2[ctz]
                    z2 -= z2c*s1[ctz]

            elif switch and rmax != -1:
                if np.equal(rmax, stim): #compare max response with stim 
                    z1 -= z1c*s1[ctz]
                    z2 -= z2c*s2[ctz]

                elif not np.equal(rmax, stim):
                    # additive model of reward signal
                    z1 += z1c*s2[ctz]
                    z2 += z2c*s1[ctz]

            z1 = me.thresh(z1)
            z2 = me.thresh(z2)

            Wz1 += gam1*z1*re
            Wz2 += gam1*z2*re

        elif learn:
            Wz1 += gam1*z1*re
            Wz2 += gam1*z2*re

        Wz1 = np.clip(Wz1, 0, wmax)
        Wz2 = np.clip(Wz2, 0, wmax)

        ## RENORMALIZE Wz's
        Wz1[2*pm.dim:4*pm.dim] = (Wz12avg/np.mean(Wz1[2*pm.dim:4*pm.dim]))*Wz1[2*pm.dim:4*pm.dim]
        Wz1[4*pm.dim:] = (Wz13avg/np.mean(Wz1[4*pm.dim:]))*Wz1[4*pm.dim:]
        Wz2[2*pm.dim:4*pm.dim] = (Wz22avg/np.mean(Wz2[2*pm.dim:4*pm.dim]))*Wz2[2*pm.dim:4*pm.dim]
        Wz2[4*pm.dim:] = (Wz23avg/np.mean(Wz2[4*pm.dim:]))*Wz2[4*pm.dim:]

        if kill_rl:
            Wz1[:2*pm.dim] = 0
            Wz2[:2*pm.dim] = 0
        else:
            Wz1[:2*pm.dim] = (Wz11avg/np.mean(Wz1[:2*pm.dim]))*Wz1[:2*pm.dim]
            Wz2[:2*pm.dim] = (Wz21avg/np.mean(Wz2[:2*pm.dim]))*Wz2[:2*pm.dim]

        if ctz%10 == 0 and store_t:
            re_t[:, int(ctz/10):int((ctz+10)/10)] = re
            ri_t[:, int(ctz/10):int((ctz+10)/10)] = ri

            xe_t[:2, int(ctz/10):int((ctz+10)/10)] = np.array((z1, z2)).reshape(-1,1)
            xe_t[2, int(ctz/10):int((ctz+10)/10)] = rmax
            xe_t[3:5, int(ctz/10):int((ctz+10)/10)] = np.array((s1[ctz], s2[ctz])).reshape(-1,1)
            xe_t[5:7, int(ctz/10):int((ctz+10)/10)] = np.array((z1i, z2i)).reshape(-1,1)
            xe_t[7:9, int(ctz/10):int((ctz+10)/10)] = np.array((z1, z2)).reshape(-1,1)
            xe_t[9, int(ctz/10):int((ctz+10)/10)] = zi_t

            Wlearn[:, int(ctz/10):int((ctz+10)/10)] = np.vstack((Wz1, Wz2))

        ctz += 1

    print('performance: ', perf)
    perf_out = (perf, resp, stim_idx1.T+(2*stim_idx2.T))
    return re_t, ri_t, xe_t, Wlearn, Wdict, perf_out, ttl


if __name__=='__main__':
    pm.ntrials = 30
    Ne = 3*2*pm.dim
    Ni = 3*2*pm.dim
    N = Ne + Ni
    pm.extstim = 'V-const'
    pm.switch = False

    re_t, ri_t, xe_t, Wlearn, Wdict, perf_out, ttl = sim()
    np.savez('testsim.npz', re=re_t, ri=ri_t, Wlearn=Wlearn, Wdict=Wdict, perf=perf)
