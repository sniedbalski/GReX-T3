# Simple script to simulate FRBs with a given DM and fluence
import argparse
import numpy as np
import simulate_frb 

def create_simulated_frbs(dm, fluence=40., 
                          width_sec=12e-6):

    dt = 8.192e-6

    ntime, nfreq = 4*16384, 2048
    freq_hi, freq_low = 1530., 1280.
    
    disp_delay = 4148 * dm * (freq_hi**-2 - freq_low**-2)
    ntime = int(4 * 16384)#max(16384, int(abs(1.5 * disp_delay / dt)))
    print("Assuming %d samples" % ntime)
    
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
        freq=(freq_hi, freq_low),
        FREQ_REF=freq_hi,
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simulate FRBs with given DM and fluence")
    parser.add_argument('--dm_start', type=float, default=25, help='Starting DM value')
    parser.add_argument('--dm_end', type=float, default=1000, help='Ending DM value')
    parser.add_argument('--nfrb', type=int, default=10, help='Number of DM steps')
    parser.add_argument('--fluence', type=float, default=40.0, help='Fluence of the FRB')
    parser.add_argument('--width_sec', type=float, default=64e-6, help='Width of the FRB pulse in seconds')

    args = parser.parse_args()

    dm_list = np.linspace(args.dm_start, args.dm_end, args.nfrb)

    for dm in dm_list:
        create_simulated_frbs(dm, fluence=args.fluence, width_sec=args.width_sec)
        print("Created DM={}".format(dm))
