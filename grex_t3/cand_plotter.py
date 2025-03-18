### Plotting T3 candidates 05/08/24 updated
### usage: poetry run python cand_plotter.py <full .json name>


import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt
import xarray as xr
import json
import os
import logging
from your.formats.filwriter import make_sigproc_object
import clean_rfi
from grex_t3 import database

T3_path = os.getenv("POETRY_PROJECT_DIR")
sys.path.append(T3_path)
import candproc_tools as ct
import analysis_tools as at

dir_mon  = "/hdd/data/voltages/"
dir_plot = "/hdd/data/candidates/T3/candplots/"
dir_fil  = "/hdd/data/candidates/T3/candfils/"
cluster_output = "/hdd/data/candidates/T2/cluster_output.csv"
logfile = '/home/cugrex/grex/t3/services/T3_plotter.log'
logging.basicConfig(filename=logfile,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logging.info('Starting cand_plotter.py!')

### To run cand_plotter.py independently, uncomment this line.
# js = str(sys.argv[1]) # full .json filename

def get_cand(JSON):
    """
    Reads the input json file
    Returns a table containing ('mjds' 'snr' 'ibox' 'dm' 'ibeam' 'cntb' 'cntc' 'specnum', 'isinjection')
    """
    try:
        f = open('/hdd/data/candidates/T2/'+JSON)
        data = json.load(f)
        tab = pd.json_normalize(data[JSON.split(".")[0]], meta=['id'])
        f.close()
        return tab
        
    except Exception as e:
        print(f"Error loading JSON file: {e}") # log error message
        logging.error("Error getting candidate json file: %s", str(e))
        return None  
    

def gen_cand(fn_vol, fn_tempfil, fn_filout, JSON, v=False): # tab - json file
    """
    ----------
    Inputs:
    fn_vol = input voltage filename
    fn_tempfil = temporary filterbank filename, will be removed after generating a plot
    fn_filout = output .fil filename
    JSON = candidate .json filename (e.g. 240321aazm.json)
    v = True: log verbose information
    ----------
    Returns:
    cand = dedispersed, downsampled Candidate object usinng the YOUR package
    tab = .json table
    """

    tab = get_cand(JSON)
    # t0 = tab["mjds"].values[0] # ToA of the candidate
    ds_stokesi = 1
    dt = 8.192e-6 * ds_stokesi # s 
    Dt = 4.15 * tab["dm"].values[0] * (1/1.28**2 - 1/1.53**2) / 1e3 # delay in seconds given DM
    
    (stokesi, T0, dur) = ct.read_voltage_data(fn_vol, 
                                              timedownsample=None, 
                                              freqdownsample=None, 
                                              verbose=True, 
                                              nbit='float32')
    print('Done reading .nc and calc stokes I')

    # choose the center of the 
    mm = (stokesi.shape[0]//2) # assume the pulse is centered
    window_width = int(Dt*2.5 / dt)
    if window_width > stokesi.shape[0]:
        window_width = stokesi.shape[0]
    if v==True:
        logging.info(f"index mm = {mm}, stokesi shape = {stokesi.shape}.")
        print(f"index mm = {mm}, stokesi shape = {stokesi.shape}.")

    # dispersed candidate window in xarray dataarray format.
    cand_disp = stokesi[(mm-window_width//2):(mm+window_width//2), :].copy()
    # RFI clean
    clean_rfi.clean_block(cand_disp.values, 5, 3)
    # replace NaNs with the mean value, otherwise the Your library won't calculate dmtime for us (all NaNs in there)
    cand_disp.values = cand_disp.fillna(np.nanmean(cand_disp))

    # write the dispersed pulse to a temporary .fil file
    # do not save intermediate file 
    ct.write_sigproc(fn_tempfil, cand_disp, t_start=T0+(mm-window_width//2)*dt/86400) 

    if v==True:
        logging.info(f"Done writing to a temporary .fil file.")
        print(f"Done writing to a temporary .fil file.")

    # Using Liam's candproc_tools.py to read the temporary .fil, dedisperse, and calculate DMtime
    cand = ct.read_proc_fil(fnfil=fn_tempfil, 
                            dm=tab["dm"].values[0], 
                            tcand=2.0, 
                            width=1, 
                            device=0, 
                            tstart=0,
                            tstop=(cand_disp.time.max()-cand_disp.time.min()).values*86400, 
                            zero_topbottom=False,
                            ndm=32, 
                            dmtime_transform=True)
    if v==True:
        logging.info("Done reading the temporary .fil and dedispersing the candidate.")
    print('Done dedispersing')

    # generate a smaller window containing the dedispersed pulse and save to .fil
    # (before downsampling)
    window_time = 256 * tab['ibox'].values[0] # would like to keep 256 samples after downsampling
    # find the index of the pulse 
    mm = int(cand.data.shape[0] / 2) # assume the pulse is centered
    if window_time > cand.data.shape[0]:
        window_time = cand.data.shape[0]
    if v==True:
        logging.info(f"Writing out to .fil file, cand index mm = {mm}, cand.data shape = {cand.data.shape}.")
        print(f"Writing out to .fil file, cand index mm = {mm}, cand.data shape = {cand.data.shape}.")
    # select a smaller time window for the dedispersed pulse
    data_freqtime = cand.dedispersed[mm-window_time//2:mm+window_time//2, :] 

    print("Testing the .h5 writer")
    cand.dedispersed = data_freqtime
    cand.dmt = cand.dmt[:, mm-window_time//2:mm+window_time//2]
    cand.save_h5(dir_fil, JSON.split('.')[0] + '.h5')

    # write to .fil
    nchans = cand.nchans
    foff = cand.foff
    fch1 = cand.fch1
    tsamp = cand.tsamp 
    sigproc_object = make_sigproc_object(
                                    rawdatafile=fn_filout,
                                    source_name="bar",
                                    nchans=nchans,
                                    foff=foff,  # MHz
                                    fch1=fch1,  # MHz
                                    tsamp=tsamp,  # seconds
                                    tstart=cand.tstart+(mm-window_width//2)*dt/86400,  
                                    src_raj=112233.44,  # HHMMSS.SS
                                    src_dej=112233.44,  # DDMMSS.SS
                                    machine_id=0,
                                    nbeams=1,
                                    ibeam=0,
                                    nbits=32,
                                    nifs=1,
                                    barycentric=0,
                                    pulsarcentric=0,
                                    telescope_id=6,
                                    data_type=0,
                                    az_start=-1,
                                    za_start=-1,)
    
    sigproc_object.write_header(fn_filout)
    sigproc_object.append_spectra(data_freqtime, fn_filout)
    logging.info(f"Done saving the dedispersed pulse into filterbank file {fn_filout}.")

    # downsampling 
    # frequency ds by 16
    cand.decimate(key = 'ft', 
                  decimate_factor = 16, 
                  axis = 1) 
    # time ds by ibox
    cand.decimate(key = 'ft', 
                  decimate_factor = tab['ibox'].values[0],
                  axis = 0,
                  pad = True) 
    # time ds in DMtime domain
    cand.decimate(key='dmt',
                  decimate_factor = tab['ibox'].values[0],
                  axis = 1,
                  pad = True) 
    # update downsampled time resolution in cand.
    cand.tsamp = cand.tsamp * tab['ibox'].values[0]
    if v==True:
        logging.info(f"Done downsampling: cand.dedispersed.shape = {cand.dedispersed.shape}; cand.dmt.shape = {cand.dmt.shape}.")
        print(f"Done downsampling: cand.dedispersed.shape = {cand.dedispersed.shape}; cand.dmt.shape = {cand.dmt.shape}.")

    return(cand, tab)



def plot_grex(cand, tab, JSON, v=False, classify_ml=False): 

    """
    Plots:
        Downsampled, dedispersed pulse, 
        DM vs time, 
        Nearby candidates from cluster_output.csv within a larger time window.
    ----------
    Inputs:
    cand = downsampled, dedispersed candidate object, the first output from gen_cand() function
    tab = candidate .json table from gen_cand()
    JSON = .json filename
    classify_ml = Boolean determines whether or not data are classified by ML model
    ----------
    Returns None
    """
    
    isinjection = database.is_injection(tab["dm"].values[0], database.connect("/hdd/data/candidates.db"))

    # number of samples in the downsampled window
    window_time = 1024
    if window_time > (cand.data.shape[0]/tab['ibox'].values[0]):
        window_time = int(cand.data.shape[0]/tab['ibox'].values[0])
    ntime, nchans = cand.dedispersed.shape[0], cand.dedispersed.shape[1]
    # roughly from 1300MHz to 1500MHz, removing junks near the two edges. (in downsampled space)
    f_low = int(278/16) 
    f_high = nchans - int(164/16) 
    
    cluster = pd.read_csv(cluster_output)
    # snr,if,specnum,mjds,ibox,idm,dm,ibeam,cl,cntc,cntb,trigger
    # candidates nearby within 60s
    cluster_time = 60./86400 # candidates within nearby 60 s
    this_cand = np.where(np.abs(cluster['mjds']-tab["mjds"].values[0])<cluster_time)[0] 
    
    # find the index of the start of the pulse <- center of cand.data
    mm = int(ntime / 2)
    # ToA of cand in the time window in seconds
    t_in_nc = mm * cand.tsamp 

    # Dedispersed pulse, remove channel mean
    data_freqtime = cand.dedispersed[mm-window_time//2:mm+window_time//2, f_low:f_high].copy() # roughly from 1300MHz to 1500MHz
    data_freqtime = (data_freqtime - 
                     np.mean(data_freqtime, axis=0, keepdims=True))
    data_freqtime = data_freqtime.T
    if v==True:
        logging.info(f"In plot_grex, index mm = {mm}, data shape = {data_freqtime.shape}.")

    max_pulse = np.argmax(data_freqtime.mean(0))
    data_freqtime = np.roll(data_freqtime, int(len(data_freqtime[0])//2-max_pulse), axis=1)
    data_timestream = data_freqtime.mean(0)
        
    # DM time
    data_dmt = cand.dmt[:, mm-window_time//2:mm+window_time//2].copy()
    data_dmt = (data_dmt - 
                np.mean(data_dmt, axis=1, keepdims=True))

    data_dmt = np.roll(data_dmt, int(len(data_dmt[0])//2-max_pulse), axis=1)
    
    # Construct time array for the window
    times = np.linspace(0,cand.tsamp*ntime,ntime) * 1e3 # Convert to milliseconds
    times = times[mm-window_time//2:mm+window_time//2]
    tmin, tmax = times[0]-t_in_nc*1000, times[-1]-t_in_nc*1000
    # Construct the downsampled frequency array, after truncating highest and lowest edges
    freqs = np.linspace(cand.fch1+(nchans-f_low)*cand.foff*16, cand.fch1+cand.foff*16*(nchans-f_high), f_high-f_low)
    freqmin, freqmax = freqs[0], freqs[-1]

    # Calculate std of the given window.
    snr_tools = at.SNR_Tools()
    snr_t3, stds = snr_tools.calc_snr_presto(data_timestream, verbose=True)

    if classify_ml:
        probability_real = model(data_freqtime)
    
    # Plot
    logging.info("Starting to plot!")
    fig = plt.figure(figsize=(10,15))
    grid = plt.GridSpec(9, 6)

    # Background color is lightcoral if it's an injection
    if isinjection:
        fig.patch.set_facecolor('lightcoral')  
        for ax in fig.get_axes():
            ax.set_facecolor('lightcoral') 
    else:
        fig.patch.set_facecolor('white')  
        for ax in fig.get_axes():
            ax.set_facecolor('white')

    # first row, collapse frequency -> time stream
    plt.subplot(grid[0, :6])
    plt.plot(times, (data_timestream-np.mean(data_timestream))/stds, lw=1., color='black')
    plt.ylabel('SNR')
    plt.xlim(times.min(), times.max())
    plt.xticks([], [])

    # the dedispersed pulse
    plt.subplot(grid[1:3, :6])
    vmax = np.mean(data_freqtime) + 2*np.std(data_freqtime)
    plt.imshow(data_freqtime, 
               aspect='auto', 
               vmax=vmax,
               extent=(times.min()-t_in_nc*1000, times.max()-t_in_nc*1000, freqs.min(), freqs.max()),
               interpolation='nearest')
    DM0_delays = tmin + cand.dm * 4.15E6 * (freqmin**-2 - freqs**-2) # zero DM sweep
    plt.plot(DM0_delays, freqs, c='r', lw='2', alpha=0.35)
    plt.xlabel('Time (ms)+ MJD {}'.format(cand.tstart+(mm*cand.tsamp)/86400), fontsize=12)
    plt.ylabel('Frequency (MHz)', fontsize=12)
    plt.xlim(tmin, tmax)

    # DM vs. time
    plt.subplot(grid[4:6, 0:6])
    plt.imshow(data_dmt, 
               aspect='auto', 
               interpolation='nearest',
               extent=(times.min()-t_in_nc*1000, times.max()-t_in_nc*1000, 0, 2*cand.dm))
    plt.xlabel('Time (ms)+ MJD {}'.format(cand.tstart+(mm*cand.tsamp)/86400), fontsize=12)
    plt.ylabel(r'DM ($pc\cdot cm^{-3}$)', fontsize=12)
    plt.ylim(cand.dm-50, cand.dm + 50)
    plt.xlim(tmin, tmax)    
    
    # DM vs. MJD in cluster_output.csv
    plt.subplot(grid[7:9, 0:6])
    plt.scatter((cluster["mjds"][this_cand].values - tab['mjds'].values[0])*86400, 
                cluster["dm"][this_cand].values, 
                s=cluster['snr'][this_cand].values / cluster['snr'][this_cand].values.max() * 100, # "normalize" marker size to 100
                c=cluster['snr'][this_cand].values)
    plt.xlim(-cluster_time*86400/2, cluster_time*86400/2)
    plt.xlabel('Time (s) + MJD {}'.format(tab['mjds'].values[0]), fontsize=12)
    plt.ylabel(r'DM ($pc\cdot cm^{-3}$)', fontsize=12)


    # doesn't seem to work?
    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.8, 
                        top=0.8, 
                        wspace=0.05, 
                        hspace=0.01)

    # Add some candidate description as the tile
    fig.text(0.1, 0.875, 'DM = {} pc/cm^3'.format(tab["dm"].values[0]),
             fontsize = 12,fontweight='semibold')
    fig.text(0.1, 0.86, 'Arriving time = MJD {}'.format(tab['mjds'].values[0]),
             fontsize = 12,fontweight='semibold')
    fig.text(0.1, 0.845, 'SNR Heimdall = {} SNR T3 = {}'.format(tab['snr'].values[0], snr_t3),
             fontsize = 12,fontweight='semibold')
    fig.text(0.1,0.83, 'Filename:'+JSON,
             fontsize = 12,fontweight='semibold')
    fig.text(0.1, 0.815, 'Ibox width = {}'.format(tab['ibox'].values[0]),
             fontsize = 12, fontweight='semibold')
    
    plt.savefig(dir_plot + 'grex_cand{}.png'.format(JSON.split('.')[0]), bbox_inches='tight')

    logging.info("Done saving the plot.")
    
    plt.show()

    return()

### To run cand_plotter.py independently, uncomment this.
# if __name__ == '__main__':
#     candname = js.split('.')[0] # candidate name 
#     vol_fn = dir_mon + "grex_dump-"+candname+".nc" # corresponding voltage netcdf file
#     fn_tempfil = dir_plot + "intermediate.fil" # output temporary filterbank file, removed afterwards
#     fn_outfil = dir_fil + f"cand{candname}.fil" # output dedispersed candidate filterbank file 

#     (cand, tab) = gen_cand(vol_fn, fn_tempfil, fn_outfil, candname+'.json')
#     plot_grex(cand, tab, candname+".json") 

#     cmd = "rm {}".format(fn_tempfil)
#     print(cmd)
#     os.system(cmd)

