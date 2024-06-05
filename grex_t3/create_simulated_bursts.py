# Simple script to simulate FRBs with a given DM and fluence

import numpy as np
import simulate_frb 

def create_simulated_frbs(dm, fluence=40., 
                          width_sec=12e-6):

    dt = 8.192e-6

    ntime, nfreq = 4*16384, 2048

    data, params = simulate_frb.gen_simulated_frb(NFREQ=nfreq,
        NTIME=ntime,
        sim=True,
        fluence=fluence, # This controls the pulse SNR
        spec_ind=0.0,
        width=width_sec,
        dm=dm, # dispersion measure
        background_noise=np.zeros([nfreq, ntime]),
        delta_t=dt,
        plot_burst=False,
        freq=(1530.0, 1280.0),
        FREQ_REF=1530.0,
        scintillate=False,
        scat_tau_ref=0.0,
        disp_ind=2.0,
        conv_dmsmear=False,
    )

    fdir = '/home/user/grex/pipeline/fake/'
    fnout = fdir + 'simulated_frb_dm{:.2f}_fluence{:.2f}.dat'.format(dm, fluence)

    with open(fnout, 'wb') as f:
        byte_data = data.astype(np.int8)
        f.write(np.ravel(byte_data,'F').tobytes())

if __name__=='__main__':
    dm_list = np.linspace(15., 200, 5)
    
    for dm in dm_list:
        create_simulated_frbs(dm) 
        print("Created DM={}".format(dm))
