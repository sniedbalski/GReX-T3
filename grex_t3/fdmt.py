import numpy as np

KDM = 1.0 / 2.41e-4
class InputBlock(object):
    """ Wraps a frequency-time 2D array """
    def __init__(self, data, fch1, fchn, tsamp):
        try:
            self._data = np.asarray(data, dtype=np.float32)
        except:
            raise ValueError("Cannot convert data to float32 numpy array")

        if not data.ndim == 2:
            raise ValueError("data must be two-dimensional")
        if not fch1 >= fchn:
            raise ValueError("The first channel must have the highest frequency")
        
        self._fch1 = float(fch1)
        self._fchn = float(fchn)
        self._tsamp = float(tsamp)

    @property
    def data(self):
        return self._data

    @property
    def tsamp(self):
        return self._tsamp

    @property
    def nchans(self):
        return self.data.shape[0]

    @property
    def nsamp(self):
        return self.data.shape[1]

    @property
    def fch1(self):
        return self._fch1

    @property
    def fchn(self):
        return self._fchn

    @property
    def foff(self):
        if self.nchans > 1:
            return (self.fchn - self.fch1) / (self.nchans - 1.0)
        else:
            return 0.0

    @property
    def freqs(self):
        return np.linspace(self.fch1, self.fchn, self.nchans)

    @property
    def delta_kdisp(self):
        """ Dispersion delay (seconds) across the band per unit of DM (pc cm^{-3}) """
        return KDM * (self.fchn**-2 - self.fch1**-2)

    @property
    def dm_step(self):
        """ Natural DM step of the fast DM transform for this data block """
        return self.tsamp / self.delta_kdisp

    @property
    def isplit(self):
        """ Line index where the tail part starts """
        if not self.nchans > 1:
            raise ValueError("Cannot split block with 1 line or less")

        # The frequency f at which the split should occur is such that:
        # f**-2 - fch1**-2 = 1/2 * (fchn**-2 - fch1**-2)
        # ie. half the total dispersion delay is accounted for between
        # fch1 and f
        f = (0.5 * self.fchn**-2 + 0.5 * self.fch1**-2) ** -0.5
        i = int((f - self.fch1) / self.foff + 0.5)
        return i

    def split(self):
        if not self.nchans > 1:
            raise ValueError("Cannot split block with 1 line or less")
        h = self.isplit
        head = InputBlock(self.data[:h], self.fch1, self.fch1 + (h-1) * self.foff, self.tsamp)
        tail = InputBlock(self.data[h:], self.fch1 + h * self.foff, self.fchn, self.tsamp)
        return head, tail
    
    def __str__(self):
        return "InputBlock(nsamp={s.nsamp}, nchans={s.nchans}, fch1={s.fch1:.3f}, fchn={s.fchn:.3f})".format(s=self)

    def __repr__(self):
        return str(self)


class OutputBlock(object):
    """ Wraps a DM-time 2D array """
    def __init__(self, input_block, ymin, ymax):
        ib = input_block
        ntrials = ymax - ymin + 1
        self._input_block = ib
        self._data = np.zeros(shape=(ntrials, ib.nsamp), dtype=np.float32)
        self._ymin = ymin
        self._ymax = ymax

    @property
    def data(self):
        """ Dedispersed data """
        return self._data

    @property
    def input_block(self):
        """ Input data block """
        return self._input_block

    @property
    def ntrials(self):
        """ Number of DM trials """
        return self.data.shape[0]

    @property
    def nsamp(self):
        """ Number of samples """
        return self.data.shape[1]

    @property
    def ymin(self):
        """ Dispersion delay (in samples) across the band for the first DM trial """
        return self._ymin

    @property
    def ymax(self):
        """ Dispersion delay (in samples) across the band for the last DM trial """
        return self._ymax

    @property
    def dm_step(self):
        """ DM step between consecutive trials """
        return self.input_block.tsamp / self.input_block.delta_kdisp

    @property
    def dm_min(self):
        """ First DM trial"""
        return self.ymin * self.dm_step
    @property
    def dm_max(self):
        """ Last DM trial """
        return self.ymax * self.dm_step

    @property
    def dms(self):
        """ List of DM trials, in the same order as they appear in the data """
        return np.linspace(self.dm_min, self.dm_max, self.ntrials)

    def __str__(self):
        return "OutputBlock(nsamp={s.nsamp}, ntrials={s.ntrials}, dm_min={s.dm_min:.3f}, dm_max={s.dm_max:.3f})".format(s=self)

    def __repr__(self):
        return str(self)
    
def transform(data, fch1, fchn, tsamp, dm_min, dm_max):
    """
    Compute the fast DM transform of a data block for the specified DM range

    Parameters
    ----------
    data: array-like
        Two dimensional input data, in frequency-time order 
        (i.e. lines are frequency channels)
    fch1: float
        Centre frequency of first channel in data
    fchn: float
        Centre frequency of last channel in data
    tsamp: float
        Sampling time in seconds
    dm_min: float
        Minimum trial DM
    dm_max: float
        Maximum trial DM

    Returns
    -------
    out: OutputBlock
        Object that wraps a 2D array with the dedispersed data.
    """
    if not dm_min >= 0:
        raise ValueError("dm_min must be >= 0")
    if not dm_max >= dm_min:
        raise ValueError("dm_max must be >= dm_min")

    block = InputBlock(data, fch1, fchn, tsamp)

    # Convert DMs to delays in samples
    ymin = int(dm_min / block.dm_step)
    ymax = int(np.ceil(dm_max / block.dm_step))
    return _transform_recursive(block, ymin, ymax)



def _transform_recursive(block, ymin, ymax):
    """
    Transform block with a range of dispersion shifts from ymin to ymax INclusive
    """
    out = OutputBlock(block, ymin, ymax)

    if block.nchans == 1:
        out.data[0] = block.data[0]
        return out

    ### Split
    head, tail = block.split()

    ### Transform
    ymin_head = int(ymin * head.delta_kdisp / block.delta_kdisp + 0.5)
    ymax_head = int(ymax * head.delta_kdisp / block.delta_kdisp + 0.5)
    thead = _transform_recursive(head, ymin_head, ymax_head)

    ymin_tail = int(ymin * tail.delta_kdisp / block.delta_kdisp + 0.5)
    ymax_tail = int(ymax * tail.delta_kdisp / block.delta_kdisp + 0.5)
    ttail = _transform_recursive(tail, ymin_tail, ymax_tail)

    ### Merge
    # y = delay in samples across the whole band
    for y in range(ymin, ymax+1):
        # yh = delay across head band
        # yt = delay across tail band
        # yb = delay at interface between head and tail
        yh = int(y * head.delta_kdisp / block.delta_kdisp + 0.5)
        yt = int(y * tail.delta_kdisp / block.delta_kdisp + 0.5)
        yb = y - yh - yt

        ih = yh - thead.ymin
        it = yt - ttail.ymin
        i = y - out.ymin
        out.data[i] = thead.data[ih] + np.roll(ttail.data[it], -(yh+yb))
    
    return out