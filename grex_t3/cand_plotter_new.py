import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray
import json

from fdmt import *

class Plotting:
    def __init__(self, path, json_file, voltage_file):
        self.path = path
        self.json_file = json_file
        self.voltage_file = voltage_file
        self.candidate_data = None 
        self.dynamic_spectrum = None 
        self.start_mjd = None 
        self.dt = None
        self.dm_t = None
        self.dm_opt = None
        self.t_opt = None
        self.t_rel = None
        self.dedispersed_spectrum = None
    
    def get_cand(self):
        """
        Reads the input json file
        Returns a table containing ('mjds' 'snr' 'ibox' 'dm' 'ibeam' 'cntb' 'cntc' 'specnum')
        """
        try:
            f = open(self.path+self.json_file)
            data = json.load(f)
            tab = pd.json_normalize(data[self.json_file.split(".")[0]], meta=['id'])
            f.close()
            self.candidate_data = tab
            return tab
        
        except Exception as e:
            print(f"Error loading JSON file: {e}") # log error message
            logging.error("Error getting candidate json file: %s", str(e))
            return None

    def read_voltage_data(self, timedownsample=1, freqdownsample=1, nbit='uint32'):
        """Read the voltage data and return as a flattened dynamic spectrum."""

        ds = xarray.open_dataset(self.path + self.voltage_file, chunks={"time": 2048})

        # Create complex numbers from Re/Im
        voltages = ds["voltages"].sel(reim="real") + ds["voltages"].sel(reim="imaginary") * 1j
        # Make Stokes I by converting to XX/YY and then creating XX**2 + YY**2
        stokesi = np.square(np.abs(voltages)).astype(nbit)
        stokesi = stokesi.sum(dim='pol').astype(nbit).compute()

        stokesi = stokesi.compute()

        if timedownsample != 1:
            stokesi = stokesi.coarsen(time=timedownsample, boundary='trim').mean()
        if freqdownsample != 1:
            stokesi = stokesi.coarsen(freq=freqdownsample, boundary='trim').mean()

        self.dt = 8.192e-6 * timedownsample
        SI = xarray.DataArray(data=stokesi.data, dims=['time', 'freq'])
        SI = SI.assign_coords(time= xarray.DataArray((stokesi.time.values - stokesi.time.values.min())*86400,dims='time'), freq= stokesi.freq.values)
        SI_flat = SI - SI.rolling(time=int(0.01/self.dt),min_periods=1,center=True).mean()
        self.dynamic_spectrum = SI_flat
        self.start_mjd = stokesi.time.min().values
        return SI_flat, stokesi.time.min().values * timedownsample

    def dedisperse(self,DM):
        """Dedisperse the dynamic spectrum for a given DM."""
        K = 4148
        dt_vec = xarray.DataArray(
            data=DM*K*((self.dynamic_spectrum.freq)**-2 - (self.dynamic_spectrum.freq.max())**-2),
            coords={'freq': self.dynamic_spectrum.freq},
            dims='freq'
        )
        tran_vec = xarray.DataArray(
            data=np.fft.fftfreq(self.dynamic_spectrum.shape[0], 8.192e-6),
            dims='tran'
        )
        spectrum = xarray.DataArray(
            data=np.fft.fft(self.dynamic_spectrum.transpose('freq', 'time').values),
            dims=['freq', 'tran'],
            coords={'freq': self.dynamic_spectrum.freq, 'tran': tran_vec}
        )
        phasor = np.exp(-2 * 1j * np.pi * (dt_vec * tran_vec))
        self.dedispersed_spectrum = self.dynamic_spectrum.copy(data=np.fft.ifft((spectrum * phasor)).real)
        return self.dynamic_spectrum.copy(data=np.fft.ifft((spectrum * phasor)).real)

    def process_candidate(self):
        """Find optimal DM and time for the candidate data."""
        ind_maxsnr = self.candidate_data["snr"].argmax()
        dm_0 = self.candidate_data['dm'][ind_maxsnr]
        t_0 = self.candidate_data['mjds'][ind_maxsnr]

        delta_t = 4.15e-3 * dm_0 * (1/(self.dynamic_spectrum.freq.min()/1000)**2 - 1/(self.dynamic_spectrum.freq.max()/1000)**2)
        t_rel = (t_0 - self.start_mjd) * 86400

        kwargs = {
            'fch1': self.dynamic_spectrum.freq.max(),
            'fchn': self.dynamic_spectrum.freq.min(),
            'tsamp': self.dt,
            'dm_min': dm_0 - 5,
            'dm_max': dm_0 + 5
        }
        dm_t_data = transform(self.dynamic_spectrum.sel(time=slice(t_rel-delta_t, t_rel+delta_t)), **kwargs)
        dm_t = xarray.DataArray(dm_t_data.data, dims=['dm', 'time'])

        t_opt = dm_t.where(dm_t == dm_t.max(), drop=True).time.values[0]
        dm_opt = dm_t.where(dm_t == dm_t.max(), drop=True).dm.values[0]
        self.dm_t,self.dm_opt,self.t_opt,self.t_rel = dm_t,dm_opt,t_opt,t_rel
        return dm_t, t_opt, dm_opt, t_rel

    def plot_results(self):
        """Plot the DM-Time and dedispersed spectrum."""
        plt.figure(figsize=(8, 5))
        self.dm_t.plot(vmin=0)
        plt.title("DM-Time Spectrum")

        plt.figure(figsize=(8, 5))
        #change 0.001 to variable
        self.dedispersed_spectrum.sel(time=slice(self.t_opt-0.001, self.t_opt+0.001)).plot(x='time', vmin=0)
        plt.title("Dedispersed Spectrum")

'''path="C:/Users/Alyss/Homework/Research/GREXNew/"
json_name="241023aarc.json"
volt_name="grex_dump-241023aarc.nc"
Test = Plotting(path,json_name,volt_name)
Test.get_cand()
Test.read_voltage_data()
Test.process_candidate()
Test.dedisperse(Test.dm_t)
Test.plot_results()'''