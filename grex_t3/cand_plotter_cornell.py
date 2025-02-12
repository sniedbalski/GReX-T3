import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray
import json

class Read_Files:
    def __init__(self,path,json_file,voltage_file):
        self.path = path
        self.json_file = json_file
        self.voltage_file = voltage_file

        self.candidate_data = None #get_cand    #NEEDED  #snr, dm, mjds
        self.dynamic_spectrum = None #read_voltage_data   #NEEDED
        self.start_mjd = None #read_voltage_data   #?
        self.dt = None

        self.get_cand()
        self.read_voltage_data()

    def get_cand(self): #Extract .json params
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
            #return tab
        
        except Exception as e:
            print(f"Error loading JSON file: {e}") # log error message
            logging.error("Error getting candidate json file: %s", str(e))
            #return None
    
    def read_voltage_data(self, timedownsample=1, freqdownsample=1, nbit='uint32'):
        """Read the voltage data and return as a flattened dynamic spectrum."""

        ds = xarray.open_dataset(self.path + self.voltage_file, chunks={"time": 2048})

        # Create complex numbers from Re/Im
        voltages = ds["voltages"].sel(reim="real") + ds["voltages"].sel(reim="imaginary") * 1j
        # Make Stokes I by converting to XX/YY and then creating XX**2 + YY**2
        stokesi = np.square(np.abs(voltages)).astype(nbit)
        stokesi = stokesi.sum(dim='pol').astype(nbit).compute()

        stokesi = stokesi.compute()

        self.downsample(self,timedownsample,freqdownsample)

        self.dt = 8.192e-6 * timedownsample
        SI = xarray.DataArray(data=stokesi.data, dims=['time', 'freq'])
        SI = SI.assign_coords(time= xarray.DataArray((stokesi.time.values - stokesi.time.values.min())*86400,dims='time'), freq= stokesi.freq.values)
        
        self.flatten(self, SI)
        
        #return SI_flat, stokesi.time.min().values * timedownsample

    def downsample(self, timedownsample=1, freqdownsample=1):
        if self.stokesi is None:
            raise ValueError("Stokes I data is not loaded. Run read_voltage_data first.")

        stokesi = self.stokesi
        if timedownsample != 1:
            stokesi = stokesi.coarsen(time=timedownsample, boundary='trim').mean()
        if freqdownsample != 1:
            stokesi = stokesi.coarsen(freq=freqdownsample, boundary='trim').mean()
        
        self.stokesi = stokesi
        self.dt = 8.192e-6 * timedownsample
        print(f"Data downsampled with time={timedownsample}, freq={freqdownsample}")

    def flatten(self):
        """Flatten the downsampled Stokes I spectrum."""
        if self.stokesi is None:
            raise ValueError("Stokes I data is not loaded. Run read_voltage_data first.")
        
        SI_flat = self.stokesi - self.stokesi.rolling(time=int(0.01/self.dt), min_periods=1, center=True).mean()
        self.dynamic_spectrum = xarray.DataArray(
            data=SI_flat.data,
            dims=['time', 'freq'],
            coords={
                'time': (SI_flat.time.values - SI_flat.time.values.min()) * 86400,
                'freq': SI_flat.freq.values
            }
        )
        print("Data flattened and stored as dynamic spectrum.")

    def output(self):
        return self.dynamic_spectrum, self.start_mjd, self.candidate_data, self.dt
    


class Process:
    def __init__(self, dynamic_spectrum, start_mjd, candidate_data, dt):
        
        old_dynamic_spectrum = dynamic_spectrum
        start_mjd = start_mjd

        ind_maxsnr = self.candidate_data["snr"].argmax()
        dm_0 = candidate_data['dm'][ind_maxsnr]
        t_0 = candidate_data['mjds'][ind_maxsnr]
        delta_t = 4.15e-3 * dm_0 * (1/(old_dynamic_spectrum.freq.min()/1000)**2 - 1/(old_dynamic_spectrum.freq.max()/1000)**2)
        t_rel = (t_0 - start_mjd) * 86400

        sliced_dynamic_spectrum = old_dynamic_spectrum.sel(time=slice(t_rel-delta_t, t_rel+delta_t))
        self.sliced_dynamic_spectrum = sliced_dynamic_spectrum
        self.dt = dt
        self.dm_0 = dm_0
        
        self.dm_t = None
        self.dm_opt = None
        self.t_opt = None

        self.dedispersed_spectrum = None
        

    def find_dm(self):
        """Find optimal DM and time for the candidate data."""

        kwargs = {
            'fch1': self.sliced_dynamic_spectrum.freq.max(), 
            'fchn': self.sliced_dynamic_spectrum.freq.min(),
            'tsamp': self.dt,
            'dm_min': self.dm_0 - 5,
            'dm_max': self.dm_0 + 5
        }
        dm_t_data = transform(self.dynamic_spectrum, **kwargs)
        dm_t = xarray.DataArray(dm_t_data.data, dims=['dm', 'time'])

        t_opt = dm_t.where(dm_t == dm_t.max(), drop=True).time.values[0]
        dm_opt = dm_t.where(dm_t == dm_t.max(), drop=True).dm.values[0]
        self.dm_t,self.dm_opt,self.t_opt,self.t_rel = dm_t,dm_opt,t_opt


    def dedisperse(self):
        """Dedisperse the dynamic spectrum for a given DM."""
        K = 4148
        DM = self.dm_opt
        dt_vec = xarray.DataArray(
            data=DM*K*((self.sliced_dynamic_spectrum.freq)**-2 - (self.sliced_dynamic_spectrum.freq.max())**-2),
            coords={'freq': self.sliced_dynamic_spectrum.freq},
            dims='freq'
        )
        tran_vec = xarray.DataArray(
            data=np.fft.fftfreq(self.sliced_dynamic_spectrum.shape[0], 8.192e-6),
            dims='tran'
        )
        spectrum = xarray.DataArray(
            data=np.fft.fft(self.dynamic_spectrum.transpose('freq', 'time').values),
            dims=['freq', 'tran'],
            coords={'freq': self.dynamic_spectrum.freq, 'tran': tran_vec}
        )
        phasor = np.exp(-2 * 1j * np.pi * (dt_vec * tran_vec))
        self.dedispersed_spectrum = self.sliced_dynamic_spectrum.copy(data=np.fft.ifft((spectrum * phasor)).real)
