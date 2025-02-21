import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import json

from fdmt import transform

class Read_Files:
    def __init__(self,json_file,voltage_file,logger):
        self.logger = logger
        self.logger.info('Reading and extracting Stokes I from voltage file.')

        self.json_file = json_file
        self.voltage_file = voltage_file
        
        self.get_cand()
        self.read_voltage_data()
        self.downsample()
        self.flatten()

        #self.candidate_data = None #get_cand    #NEEDED  #snr, dm, mjds
        #self.dynamic_spectrum = None #read_voltage_data   #NEEDED
        #self.start_mjd = None #read_voltage_data   #?
        #self.dt = None

    def get_cand(self): #Extract .json params
        """
        Reads the input json file
        Returns a table containing ('mjds' 'snr' 'ibox' 'dm' 'ibeam' 'cntb' 'cntc' 'specnum')
        """
        try:
            with open(self.json_file) as f:
                data = json.load(f)
                self.candidate_data = pd.json_normalize(data[self.json_file.split("/")[-1].split(".")[0]], meta=['id'])
            #return tab
        
        except Exception as e:
            self.logger.error("Error getting candidate json file: %s", str(e))
            #return None
    
    def read_voltage_data(self, nbit='uint32'):
        """Read the voltage data and return as a flattened dynamic spectrum."""

        try:
            ds = xr.open_dataset(self.voltage_file, chunks={"time": 2048})
        except Exception as e:
            self.logger.error("Error opening voltage file: %s", str(e))

        try:
            # Create complex numbers from Re/Im
            voltages = ds["voltages"].sel(reim="real") + ds["voltages"].sel(reim="imaginary") * 1j
            # Make Stokes I by converting to XX/YY and then creating XX**2 + YY**2
            stokesi = np.square(np.abs(voltages)).astype(nbit)
            stokesi = stokesi.sum(dim='pol').astype(nbit).compute()

            stokesi = stokesi.compute()

            self.dt = 8.192e-6
            SI = xr.DataArray(data=stokesi.data, dims=['time', 'freq'])
            self.start_mjd = stokesi.time.values.min()
            self.stokesI = SI.assign_coords(time= xr.DataArray((stokesi.time.values - self.start_mjd)*86400,dims='time'), freq= stokesi.freq.values)

            self.logger.info("Stokes I data generated")

        except Exception as e:
            self.logger.error("Error computing Stokes I values: %s", str(e))

    def downsample(self, timedownsample=1, freqdownsample=1):
        if self.stokesI is None:
            raise ValueError("Stokes I data is not loaded. Run read_voltage_data first.")

        #stokesi = self.stokesi
        if timedownsample != 1:
            self.stokesI = self.stokesI.coarsen(time=timedownsample, boundary='trim').mean()
        if freqdownsample != 1:
            self.stokesI = self.stokesI.coarsen(freq=freqdownsample, boundary='trim').mean()
        
        #self.stokesi = stokesi
        self.dt = self.dt * timedownsample
        self.logger.info(f"Data downsampled with time={timedownsample}, freq={freqdownsample}")

    def flatten(self, q=False):
        try:
            """Flatten the downsampled Stokes I spectrum."""
            if self.stokesI is None:
                raise ValueError("Stokes I data is not loaded. Run read_voltage_data first.")
            
            if q:
                self.bandpass = self.stokesI.rolling(time=int(0.01/self.dt), min_periods=1, center=True).mean()
                self.stokesI = self.stokesI - self.bandpass
            elif q==False:
                self.bandpass = xr.zeros_like(self.stokesI.freq)
            '''self.dynamic_spectrum = xr.DataArray(
                data=SI_flat.data,
                dims=['time', 'freq'],
                coords={
                    'time': (SI_flat.time.values - SI_flat.time.values.min()) * 86400,
                    'freq': SI_flat.freq.values
                }
            )'''
            self.logger.info("Data flattened and stored as dynamic spectrum.")
        except Exception as e:
            self.logger.error("Error while flattening dynamic spectrum: %s", str(e))

    def output(self):
        return self.stokesI, self.start_mjd, self.candidate_data, self.dt
    
class Process:
    def __init__(self, **kwargs):
        self.logger = kwargs['logger']
        self.logger.info('Processing candidate data.')
        
        old_dynamic_spectrum, start_mjd, candidate_data, dt = Read_Files(**kwargs).output()

        try:
            ind_maxsnr = candidate_data["snr"].argmax()
            dm_0 = candidate_data['dm'][ind_maxsnr]
            t_0 = candidate_data['mjds'][ind_maxsnr]
            delta_t = 4.15e-3 * dm_0 * (1/(old_dynamic_spectrum.freq.min()/1000)**2 - 1/(old_dynamic_spectrum.freq.max()/1000)**2)
            t_rel = (t_0 - start_mjd) * 86400

            sliced_dynamic_spectrum = old_dynamic_spectrum.sel(time=slice(t_rel-delta_t, t_rel+delta_t))
            self.dynamic_spectrum = sliced_dynamic_spectrum
            self.dt = dt
            self.dm_0 = dm_0

        except Exception as e:
            self.logger.error("Error in calculating dispersive time delay: %s", str(e))
        
        self.dm_t = None
        self.dm_opt = None
        self.t_opt = None

        self.dedispersed_spectrum = None

        self.dedisperse()
        
    def find_dm(self):
        """Find optimal DM and time for the candidate data."""
        try:
            kwargs = {
                'fch1': self.dynamic_spectrum.freq.max(), 
                'fchn': self.dynamic_spectrum.freq.min(),
                'tsamp': self.dt,
                'dm_min': self.dm_0 - 5,
                'dm_max': self.dm_0 + 5
            }
            dm_t_data = transform(self.dynamic_spectrum, **kwargs)
            dm_t = xr.DataArray(dm_t_data.data, dims=['dm', 'time'])

            t_opt = dm_t.where(dm_t == dm_t.max(), drop=True).time.values[0]
            dm_opt = dm_t.where(dm_t == dm_t.max(), drop=True).dm.values[0]
            self.dm_t,self.dm_opt,self.t_opt,self.t_rel = dm_t,dm_opt,t_opt
            self.logger.info("Optimal DM computed")

        except Exception as e:
            self.logger.error("Unable to compute optimal DM: %s", str(e))


    def dedisperse(self):
        """Dedisperse the dynamic spectrum for a given DM."""
        K = 4148
        if self.dm_opt != None:
            DM = self.dm_opt
        else:
            self.logger.info(f"Optimal DM not available. Using Heimdall DM instead.")
            DM = self.dm_0

        try:
            dt_vec = xr.DataArray(
                data=DM*K*((self.dynamic_spectrum.freq)**-2 - (self.dynamic_spectrum.freq.max())**-2),
                coords={'freq': self.dynamic_spectrum.freq},
                dims='freq'
            )
            tran_vec = xr.DataArray(
                data=np.fft.fftfreq(self.dynamic_spectrum.shape[0], self.dt),
                dims='tran'
            )
            spectrum = xr.DataArray(
                data=np.fft.fft(self.dynamic_spectrum.transpose('freq', 'time').values),
                dims=['freq', 'tran'],
                coords={'freq': self.dynamic_spectrum.freq, 'tran': tran_vec}
            )
            phasor = np.exp(-2 * 1j * np.pi * (dt_vec * tran_vec))
            self.dedispersed_spectrum = self.dynamic_spectrum.copy(data=np.fft.ifft((spectrum * phasor)).real.T)
            self.logger.info("Dynamc Spectrum dedispersed succesfully.")

        except Exception as e:
            self.logger.error("Error in performing dedispersion: %s", str(e))