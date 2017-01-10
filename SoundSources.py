import numpy as numpy
import matplotlib.pyplot as plt
import wave
import os



def read_wavfile(filename, **kwargs):
    """
    Returns list of sampled_waveforms, one per channel.
    Audio samples are in range -1.0 to +1.0 if gain=None is used
    Modified from: http://web.mit.edu/6.02/www/f2010/handouts/labs/lab7/waveforms.py
    
    Arguments: 
    gain: None for "Autogain", which scales peak value to 100% full-scale. 
        Use any number to apply a fixed gain. 
    """
    gain = kwargs.get('gain', 1.0)
    
    assert os.path.exists(filename),"file %s doesn't exist" % filename
    wav = wave.open(filename,'rb')
    nframes = wav.getnframes()
    assert nframes > 0,"%s doesn't have any audio data!" % filename
    nchan = wav.getnchannels()
    sample_rate = wav.getframerate()
    sample_width = wav.getsampwidth()

    # see http://ccrma.stanford.edu/courses/422/projects/WaveFormat/
    g = 1.0 if gain is None else gain
    if sample_width == 1:
        # data is unsigned bytes, 0 to 255
        dtype = numpy.uint8
        scale = g / 127.0
        offset = -1.0
    elif sample_width == 2:
        # data is signed 2's complement 16-bit samples (little-endian byte order)
        dtype = numpy.int16
        scale = g / 32767.0
        offset = 0.0
    elif sample_width == 4:
        # data is signed 2's complement 32-bit samples (little-endian byte order)
        dtype = numpy.int32
        scale = g / 2147483647.0
        offset = 0.0
    else:
        assert False,"unrecognized sample width %d" % sample_width

    outputs = [numpy.zeros(nframes, dtype=numpy.float64) for i in xrange(nchan)]

    count = 0
    while count < nframes:
        audio = numpy.frombuffer(wav.readframes(nframes-count), dtype=dtype)
        end = count + (len(audio) / nchan)
        for i in xrange(nchan):
            outputs[i][count:end] = audio[i::nchan]
        count = end
        
    # scale data appropriately
    for i in xrange(nchan):
        numpy.multiply(outputs[i], scale, outputs[i])
        if offset != 0: numpy.add(outputs[i],offset,outputs[i])

    # apply auto gain
    if gain is None:
        maxmag = max([max(numpy.absolute(outputs[i])) for i in xrange(nchan)])
        for i in xrange(nchan):
            numpy.multiply(outputs[i],1.0/maxmag,outputs[i])

    return [sampled_waveform(outputs[i],sample_rate=sample_rate) for i in xrange(nchan)]


def write_wavfile(*waveforms,**keywords):
    """
    Write a n-channel WAV file using samples from the n supplied waveforms
    Modified from: http://web.mit.edu/6.02/www/f2010/handouts/labs/lab7/waveforms.py
    """
    filename = keywords.get('filename',None)
    gain = keywords.get('gain',1.0)
    sample_width = keywords.get('sample_width',2)

    assert filename,"filename must be specified"
    nchan = len(waveforms)
    assert nchan > 0,"must supply at least one waveform"
    nsamples = waveforms[0].nsamples
    sample_rate = waveforms[0].sample_rate
    domain = waveforms[0].domain
    for i in xrange(1,nchan):
        assert waveforms[i].nsamples==nsamples,\
               "all waveforms must have the same number of samples"
        assert waveforms[i].sample_rate==sample_rate,\
               "all waveforms must have the same sample rate"
        assert waveforms[i].domain==domain,\
               "all waveforms must have the same domain"

    if gain is None:
        maxmag = max([max(numpy.absolute(waveforms[i].samples))
                      for i in xrange(nchan)])
        gain = 1.0/maxmag

    if sample_width == 1:
        dtype = numpy.uint8
        scale = 127.0 * gain
        offset = 127.0
    elif sample_width == 2:
        dtype = numpy.int16
        scale = 32767.0 * gain
        offset = 0
    elif sample_width == 4:
        dtype = numpy.int32
        scale = 2147483647.0 * gain
        offset = 0
    else:
        assert False,"sample_width must be 1, 2, or 4 bytes"

    # array to hold scaled data for 1 channel
    temp = numpy.empty(nsamples,dtype=numpy.float64)
    # array to hold frame data all channels
    data = numpy.empty(nchan*nsamples,dtype=dtype)

    # process the data
    for i in xrange(nchan):
        # apply appropriate scale and offset
        numpy.multiply(waveforms[i].samples,scale,temp)
        if offset != 0: numpy.add(temp,offset,temp)
        # interleave channel samples in the output array
        data[i::nchan] = temp[:]

    # send frames to wav file
    wav = wave.open(filename,'wb')
    wav.setnchannels(nchan)
    wav.setsampwidth(sample_width)
    wav.setframerate(sample_rate)
    wav.writeframes(data.tostring())
    wav.close()


def read_csvfile(csvFilesrc, **kwargs): 
    """
    Creates a multi-channel WAV file from each column in a CSV file.
    
    csvFilesrc: Source file. 1st row is header info.
    sampleRate: Sample rate of source and, ultimately, WAV file.
    """
    import csv
    
    sampleRate = kwargs.get('sampleRate', 8000)
    
    # Read CSV file by sniffing its dialect
    fileRows = []
    with open(csvFilesrc, 'rb') as csvFileHandle: 
        dialect = csv.Sniffer().sniff(csvFileHandle.read(1024))
        csvFileHandle.seek(0)
        csvReader = csv.DictReader(csvFileHandle, dialect=dialect)
        for row in csvReader: 
            fileRows.append(row)

    # Transpose the data: list of dicts to dict of lists
    dictOfLists = {}
    for rowDict in fileRows: 
        for k,v in rowDict.items():  
            try: 
                val = float(v)
            except ValueError as e: 
                print "Warning: Could not convert \"{0}\". Skipping".format(e.message)
                continue
            dictOfLists.setdefault(k, []).append(val)

    chans, audio = [], []
    for k,v in dictOfLists.iteritems(): 
        chans.append(k)
        audio.append(sampled_waveform(
            numpy.array(v),
            sample_rate = sampleRate, 
            domain = 'time', 
        ))

    return chans, audio
    
    
def sliceWav(wavFilename, timeSegments, channel=0, slicedFilenamePrefix='_slice'): 
    """
    Reads in wavFilename, chops it into segments defined by the timeSegment list, 
    and saves the files using slicedFilenamePrefix and an iterator.
    
    wavFilename: Source WAV file
    timeSegment: List of start and end time segments to slice of the format (start,end)
    channel: Channel to use in the source WAV file
    slicedFilenamePrefix: Prefix to use when naming the sliced files
    
    Examples: 
    sliceWav(wavFilename=filename, channel=0, timeSegments=[(0, 12.35), (12.35, 22.75), (22.75, 35), (35, 47)])
    """
    sampledWav = read_wavfile(wavFilename)[channel]
    wavFilenameNoExt = os.path.splitext(wavFilename)[0]
    for i,timeSegment in enumerate(timeSegments): 
        filename = '{0}{1}{2}.wav'.format(wavFilenameNoExt, slicedFilenamePrefix, i)
        
        start = 0 if timeSegment[0] == None else numpy.floor(timeSegment[0] * sampledWav.sample_rate)
        end = sampledWav.nsamples if timeSegment[1] == None else numpy.ceil(timeSegment[1] * sampledWav.sample_rate)
        
        if start < 0: 
            print "Warning: start timeSegment less than zero. Changing start sample to 0 for file \"{0}\".".format(filename)
            start = 0
        if end > sampledWav.nsamples: 
            print "Warning: end timeSegment greater than number of samples in WAV file. Changing the end point to the last sample for file \"{0}\".".format(filename)
            end = sampledWav.nsamples
        
        slice = sampled_waveform(sampledWav.samples[start:end], sample_rate=sampledWav.sample_rate)
        write_wavfile(slice, filename=filename, sample_width=2)
    return 

    
def hysteresis_threshold(x, high, low, state=False):
    """
    x: Input numpy array
    high: High threshold 
    low: Low threshold
    state: Initial state
    
    Returns: 
    y: Numpy array of boolean values showing the state of samples relative
    to the threshold limits
    """
    y = numpy.empty(len(x), dtype=numpy.bool)
    y_high = (x >= high)
    y_low = (x >= low)
    for i in xrange(len(x)):
        state = (state and y_low[i]) or y_high[i]
        y[i] = state
    return y

    
def cic(x, n):
    """
    x: Input numpy array
    n: Width of the comb filter
    
    Returns: 
    y: Numpy array of averaged values
    """
    y = numpy.zeros(x.shape, dtype=x.dtype)
    y[:n] = numpy.cumsum(x[:n])
    y[n:] = y[n-1] + numpy.cumsum(x[n:] - x[:-n])
    return y
    
    
def sliceWavBySilence(wavFilename, **kwargs): 
    """
    Reads in wavFilename, chops it into segments defined by durations of silence, 
    and saves the files using slicedFilenamePrefix and an iterator.
    
    wavFilename: Source WAV file
    timeSegment: List of start and end time segments to slice of the format (start,end)
    channel: Channel to use in the source WAV file
    slicedFilenamePrefix: Prefix to use when naming the sliced files
    
    Examples: 
    sliceWav(wavFilename=filename, channel=0, timeSegments=[(0, 12.35), (12.35, 22.75), (22.75, 35), (35, 47)])
    """
    audioOnVal = kwargs.get('audioOnVal', 0.031)
    audioOffVal = kwargs.get('audioOffVal', 0.008)
    channel = kwargs.get('channel', 0)
    slicedFilenamePrefix = kwargs.get('slicedFilenamePrefix', '_slice')
    maxSilenceDurationSec = kwargs.get('maxSilenceDurationSec', 0.1)
    slicePaddingSec = kwargs.get('slicePaddingSec', 0.0)
    
    sampledWav = read_wavfile(wavFilename)[channel]
    samples = sampledWav.samples

    # Calculate boolean hysteresis array
    audio_rms = numpy.sqrt(cic(samples**2, 32) / 32.)[16:]
    audio_state = hysteresis_threshold(
        x = audio_rms,
        high = audioOnVal,
        low = audioOffVal,
        state = False)
    
    # Calculate array of edge transitions (ignore first sample)
    edges = numpy.logical_xor(audio_state[1:], audio_state[:-1])
    edges = numpy.flatnonzero(edges)
    
    startingState = not audio_state[1:][edges[0]]
    # Starting state is true, so first edge is falling edge. 
    if startingState: 
        edges = numpy.insert(edges, 0, 0)

    # If there is an odd number of edges, last sample is falling edge
    if len(edges) % 2 != 0: 
        edges = numpy.append(edges, len(audio_state)-1)

    # Reshape array into (start,end) tuples when audio is present
    soundPeriods = numpy.reshape(edges, (-1, 2))
    
    # Ignore silent periods of short duration
    maxSilenceDurationSamp = maxSilenceDurationSec * sampledWav.sample_rate
    activePeriods = []
    activeStart,prevEnd = soundPeriods[0]
    for i,(start,end) in enumerate(soundPeriods): 
        duration = start - prevEnd
        if duration > maxSilenceDurationSamp or i == len(soundPeriods)-1:
            activePeriods.append([activeStart, prevEnd])
            activeStart = start
        prevEnd = end
    
    # Pad the audio file slices
    slicePaddingSamp = slicePaddingSec * sampledWav.sample_rate
    activePeriods = [[max(0, start - slicePaddingSamp), min(end + slicePaddingSamp, len(samples)-1)] for start,end in activePeriods]
    
    # Plot the audio file & the sliced regions
    # plt.plot(numpy.arange(len(samples)) / float(sampledWav.sample_rate), samples)
    # for start,end in activePeriods: 
        # plt.axvline(x=start / float(sampledWav.sample_rate), linewidth=2, color = 'k')
        # plt.axvline(x=end / float(sampledWav.sample_rate), linewidth=2, color = 'r')
    
    wavFilenameNoExt = os.path.splitext(wavFilename)[0]
    for i,(start,end) in enumerate(activePeriods): 
        slice = sampled_waveform(sampledWav.samples[start:end], sample_rate=sampledWav.sample_rate)
        filename = '{0}{1}{2}.wav'.format(wavFilenameNoExt, slicedFilenamePrefix, i)
        write_wavfile(slice, filename=filename, sample_width=2)
    
    return


# compute number of taps given sample_rate and transition_width.
# Stolen from the gnuradio firdes routines
def compute_ntaps(transition_width,sample_rate,window):
    delta_f = float(transition_width)/sample_rate
    width_factor = {
        'hamming': 3.3,
        'hann': 3.1,
        'blackman': 5.5,
        'rectangular': 2.0,
        }.get(window,None)
    assert width_factor,\
           "compute_ntaps: unrecognized window type %s" % window
    ntaps = int(width_factor/delta_f + 0.5)
    return (ntaps & ~0x1) + 1   # ensure it's odd

# compute specified window given number of taps
# formulae from Wikipedia
def compute_window(window,ntaps):
    order = float(ntaps - 1)
    if window == 'hamming':
        return [0.53836 - 0.46164*numpy.cos((2*numpy.pi*i)/order)
                for i in xrange(ntaps)]
    elif window == 'hann' or window == 'hanning':
        return [0.5 - 0.5*numpy.cos((2*numpy.pi*i)/order)
                for i in xrange(ntaps)]
    elif window == 'bartlett':
        return [1.0 - abs(2*i/order - 1)
                for i in xrange(ntaps)]
    elif window == 'blackman':
        alpha = .16
        return [(1-alpha)/2 - 0.50*numpy.cos((2*numpy.pi*i)/order)
                - (alpha/2)*numpy.cos((4*numpy.pi*i)/order)
                for i in xrange(ntaps)]
    elif window == 'nuttall':
        return [0.355768 - 0.487396*numpy.cos(2*numpy.pi*i/order)
                         + 0.144232*numpy.cos(4*numpy.pi*i/order)
                         - 0.012604*numpy.cos(6*numpy.py*i/order)
                for i in xrange(ntaps)]
    elif window == 'blackman-harris':
        return [0.35875 - 0.48829*numpy.cos(2*numpy.pi*i/order)
                        + 0.14128*numpy.cos(4*numpy.pi*i/order)
                        - 0.01168*numpy.cos(6*numpy.pi*i/order)
                for i in xrange(ntaps)]
    elif window == 'blackman-nuttall':
        return [0.3635819 - 0.4891775*numpy.cos(2*numpy.pi*i/order)
                          + 0.1365995*numpy.cos(4*numpy.pi*i/order)
                          - 0.0106411*numpy.cos(6*numpy.py*i/order)
                for i in xrange(ntaps)]
    elif window == 'flat top':
        return [1 - 1.93*numpy.cos(2*numpy.pi*i/order)
                  + 1.29*numpy.cos(4*numpy.pi*i/order)
                  - 0.388*numpy.cos(6*numpy.py*i/order)
                  + 0.032*numpy.cos(8*numpy.py*i/order)
                for i in xrange(ntaps)]
    elif window == 'rectangular' or window == 'dirichlet':
        return [1 for i in xrange(ntaps)]
    else:
        assert False,"compute_window: unrecognized window type %s" % window
    

# Stolen from the gnuradio firdes routines
def fir_taps(type,cutoff,sample_rate,
                 window='hamming',transition_width=None,ntaps=None,gain=1.0):
    if ntaps:
        ntaps = (ntaps & ~0x1) + 1   # make it odd
    else:
        assert transition_width,"compute_taps: one of ntaps and transition_width must be specified"
        ntaps = compute_ntaps(transition_width,sample_rate,window)

    window = compute_window(window,ntaps)

    middle = (ntaps - 1)/2
    taps = [0] * ntaps
    fmax = 0

    if isinstance(cutoff,tuple):
        fc = [float(cutoff[i])/sample_rate for i in (0,1)]
        wc = [2*numpy.pi*fc[i] for i in (0,1)]
    else:
        fc = float(cutoff)/sample_rate
        wc = 2*numpy.pi*fc

    if type == 'low-pass':
        # for low pass, gain @ DC = 1.0
        for i in xrange(ntaps):
            if i == middle:
                coeff = (wc/numpy.pi) * window[i]
                fmax += coeff
            else:
                n = i - middle
                coeff = (numpy.sin(n*wc)/(n*numpy.pi)) * window[i]
                fmax += coeff
            taps[i] = coeff
    elif type == 'high-pass':
        # for high pass gain @ nyquist freq = 1.0
        for i in xrange(ntaps):
            if i == middle:
                coeff = (1.0 - wc/numpy.pi) * window[i]
                fmax += coeff
            else:
                n = i - middle
                coeff = (-numpy.sin(n*wc)/(n*numpy.pi)) * window[i]
                fmax += coeff * numpy.cos(n*numpy.pi)
            taps[i] = coeff
    elif type == 'band-pass':
        # for band pass gain @ (fc_lo + fc_hi)/2 = 1.0
        # a band pass filter is simply the combination of
        #   a high-pass filter at fc_lo  in series with
        #   a low-pass filter at fc_hi
        # so convolve taps to get the effect of composition in series
        for i in xrange(ntaps):
            if i == middle:
                coeff = ((wc[1] - wc[0])/numpy.pi) * window[i]
                fmax += coeff
            else:
                n = i - middle
                coeff = ((numpy.sin(n*wc[1]) - numpy.sin(n*wc[0]))/(n*numpy.pi)) * window[i]
                fmax += coeff * numpy.cos(n*(wc[0] + wc[1])*0.5)
            taps[i] = coeff
    elif type == 'band-reject':
        # for band reject gain @ DC = 1.0
        # a band reject filter is simply the combination of
        #   a low-pass filter at fc_lo   in series with a
        #   a high-pass filter at fc_hi
        # so convolve taps to get the effect of composition in series
        for i in xrange(ntaps):
            if i == middle:
                coeff = (1.0 - ((wc[1] - wc[0])/numpy.pi)) * window[i]
                fmax += coeff
            else:
                n = i - middle
                coeff = ((numpy.sin(n*wc[0]) - numpy.sin(n*wc[1]))/(n*numpy.pi)) * window[i]
                fmax += coeff
            taps[i] = coeff
    else:
        assert False,"compute_taps: unrecognized filter type %s" % type

    gain = gain / fmax
    for i in xrange(ntaps): taps[i] *= gain
    return taps

    
class sampled_waveform:
    """
    Sampled_waveform base class
    Modified from: http://web.mit.edu/6.02/www/f2010/handouts/labs/lab7/waveforms.py
    """
    def __init__(self,samples,sample_rate=1e6,domain='time'):
        if not isinstance(samples,numpy.ndarray):
            samples = numpy.array(samples,dtype=numpy.float,copy=True)
        self.samples = numpy.array(samples, copy=True)   # a numpy array
        self.nsamples = len(samples)
        self.sample_rate = sample_rate
        self.domain = domain

    def _check(self,other):
        if isinstance(other,(int,float,numpy.ndarray)):
            return other
        elif isinstance(other,(tuple,list)):
            return numpy.array(other)
        elif isinstance(other,sampled_waveform):
            assert self.nsamples == other.nsamples,\
                   "both waveforms must have same number of samples"
            assert self.sample_rate == other.sample_rate,\
                   "both waveforms must have same sample rate"
            assert self.domain == other.domain,\
                   "both waveforms must be in same domain"
            return other.samples
        else:
            assert False,"unrecognized operand type"

    def real(self):
        return sampled_waveform(numpy.real(self.samples),
                                sample_rate=self.sample_rate,
                                domain=self.domain)

    def imag(self):
        return sampled_waveform(numpy.imag(self.samples),
                                sample_rate=self.sample_rate,
                                domain=self.domain)

    def magnitude(self):
        return sampled_waveform(numpy.absolute(self.samples),
                                sample_rate=self.sample_rate)

    def rms(self):
        return calcRms(self.samples)

    def crestFactor(self):
        return calcCrestFactor(self.samples)

    def angle(self):
        return sampled_waveform(numpy.angle(self.samples),
                                sample_rate=self.sample_rate)

    def __add__(self,other):
        ovalues = self._check(other)
        return sampled_waveform(self.samples + ovalues,
                                sample_rate=self.sample_rate,
                                domain=self.domain)

    def __radd__(self,other):
        ovalues = self._check(other)
        return sampled_waveform(self.samples + ovalues,
                                sample_rate=self.sample_rate,
                                domain=self.domain)

    def __sub__(self,other):
        ovalues = self._check(other)
        return sampled_waveform(self.samples - ovalues,
                                sample_rate=self.sample_rate,
                                domain=self.domain)

    def __rsub__(self,other):
        ovalues = self._check(other)
        return sampled_waveform(ovalues - self.samples,
                                sample_rate=self.sample_rate,
                                domain=self.domain)

    def __mul__(self,other):
        ovalues = self._check(other)
        return sampled_waveform(self.samples * ovalues,
                                sample_rate=self.sample_rate,
                                domain=self.domain)

    def __rmul__(self,other):
        ovalues = self._check(other)
        return sampled_waveform(self.samples * ovalues,
                                sample_rate=self.sample_rate,
                                domain=self.domain)

    def __div__(self,other):
        ovalues = self._check(other)
        return sampled_waveform(self.samples / ovalues,
                                sample_rate=self.sample_rate,
                                domain=self.domain)

    def __rdiv__(self,other):
        ovalues = self._check(other)
        return sampled_waveform(ovalues / self.samples,
                                sample_rate=self.sample_rate,
                                domain=self.domain)

    def __abs__(self):
        return sampled_waveform(numpy.absolute(self.samples),
                                sample_rate=self.sample_rate,
                                domain=self.domain)
        
    def __len__(self):
        return len(self.samples)

    def __mod__(self,other):
        ovalues = self._check(other)
        return sampled_waveform(numpy.fmod(self.samples,ovalues),
                                sample_rate=self.sample_rate,
                                domain=self.domain)


    def slice(self,start,stop,step=1):
        return sampled_waveform(self.samples[start:stop:step],
                                sample_rate=self.sample_rate,
                                domain=self.domain)

    def __getitem__(self,key):
        return self.samples.__getitem__(key)

    def __setitem__(self,key,value):
        if isinstance(value,sampled_waveform):
            value = value.samples
        self.samples.__setitem__(key,value)

    def __iter__(self):
        return self.samples.__iter__()

    def __str__(self):
        return str(self.samples)+('@%d samples/sec' % self.sample_rate)

    def resize(self,length):
        self.samples.resize(length)
        self.nsamples = length

    def convolve(self,taps):
        conv_res = numpy.convolve(self.samples,taps)
        offset = len(taps)/2
        return sampled_waveform(conv_res[offset:offset+self.nsamples],
                                sample_rate=self.sample_rate,
                                domain=self.domain)
                                
    def repeat(self, count):
        """
        Repeats (tiles) the existing waveform count times.
        """
        return sampled_waveform(
            numpy.tile(self.samples, count),
            sample_rate = self.sample_rate, 
            domain = self.domain)

    def appendSilence(self, numSilentSamps): 
        return sampled_waveform(
            numpy.append(self.samples, numpy.zeros(numSilentSamps)),
            sample_rate = self.sample_rate, 
            domain = self.domain)
    
    def modulate(self,hz,phase=0.0,gain=1.0):

        periods = float(self.nsamples*hz)/float(self.sample_rate)
        if abs(periods - round(periods)) > 1.0e-6:
            print "Warning: Non-integral number of modulation periods"
            print "nsamples=%d hz=%f sample_rate=%d periods=%f" % (self.nsamples, hz,self.sample_rate,periods)

        result = sinusoid(hz=hz,nsamples=self.nsamples,
                          sample_rate=self.sample_rate,phase=phase,
                          amplitude=gain)
        numpy.multiply(self.samples,result.samples,result.samples)
        return result

    def filter(self,type,cutoff,
               window='hamming',transition_width=None,ntaps=None,error=0.05,gain=1.0):
        if ntaps is None and transition_width is None:
            # ensure sufficient taps to represent a frequency resolution of error*cutoff
            ntaps = int(float(self.sample_rate)/(float(cutoff)*error))
            if ntaps & 1: ntaps += 1
        taps = fir_taps(type,cutoff,self.sample_rate,
                        window=window,transition_width=transition_width,
                        ntaps=ntaps,gain=gain)
        return self.convolve(taps)

    def quantize(self,thresholds):
        levels = [float(v) for v in thresholds]
        levels.sort()  # min to max
        nlevels = len(levels)
        output = numpy.zeros(self.nsamples,dtype=numpy.int)
        compare = numpy.empty(self.nsamples,dtype=numpy.bool)
        mask = numpy.zeros(self.nsamples,dtype=numpy.bool)
        mask[:] = True
        # work our way from min slicing level to max
        for index in xrange(nlevels):
            # find elements <= current slicing level
            numpy.less_equal(self.samples,levels[index],compare)
            # mask out those we've already categorized
            numpy.logical_and(mask,compare,compare)
            # set symbol value for outputs in this bucket
            output[compare] = index
            # now mark the newly bucketed values as processed
            mask[compare] = False
        # remaining elements are in last bucket
        output[mask] = nlevels
        return sampled_waveform(output,sample_rate=self.sample_rate)

    def play(self,gain=None):
        play(self.samples,self.sample_rate,gain=gain)

    def plot(self, xaxis=None, yaxis=None, title="", linetype="b", absplot=False, label=None):
        plt.figure()
        plt.title(title)
        if self.domain == 'time':
            x_range,x_prefix,x_scale = eng_notation((0,float(self.nsamples - 1)/self.sample_rate))
            if xaxis is None: xaxis = x_range
            else: xaxis = (xaxis[0]*x_scale,xaxis[1]*x_scale)
            x_step = (x_range[1] - x_range[0])/float(self.nsamples-1)
            plt.plot(numpy.arange(self.nsamples)*x_step + x_range[0],self.samples,linetype, label=label)
            if yaxis is None:
                yaxis = (min(self.samples),max(self.samples))
                dy = yaxis[1]-yaxis[0]
                yaxis = (yaxis[0] - .1*dy,yaxis[1] + .1*dy)
            plt.axis((xaxis[0],xaxis[1],yaxis[0],yaxis[1]))
            plt.xlabel(x_prefix+'s')
            plt.ylabel('V')
        elif self.domain == 'frequency':
            nyquist = self.sample_rate/2
            x_range,x_prefix,x_scale = eng_notation((-nyquist,nyquist))
            if xaxis is None:
                xaxis = x_range
            else:
                xaxis = (xaxis[0]*x_scale,xaxis[1]*x_scale)
            if (self.nsamples & 1) == 0:
                # even number of samples
                x_step = (x_range[1] - x_range[0])/float(self.nsamples)
            else:
                # odd number of samples
                x_step = (x_range[1] - x_range[0])/float(self.nsamples-1)

            dftscale = float(1.0)/float(self.nsamples)
            dft = dftscale * numpy.fft.fftshift(self.samples)
            if absplot:
                numpy.absolute(dft,dft)
                plt.plot(numpy.arange(self.nsamples)*x_step + x_range[0],dft,linetype, label=label)
                if yaxis is None:
                    yaxis = (min(dft),max(dft))
                plt.axis((xaxis[0],xaxis[1],yaxis[0],yaxis[1]))
                plt.xlabel(x_prefix+'Hz')
                plt.ylabel('Magnitude')
            else:
                plt.subplot(211)
                plt.title(title)
                K = (len(dft)-1)/2
                plt.plot(numpy.arange(self.nsamples)*x_step+x_range[0],dft.real,"b.", label=label)
                plt.plot(numpy.arange(self.nsamples)*x_step + x_range[0],dft.real, label=label)
                yaxis = (min(-0.1, min(dft.real)-0.05),max(0.1,max(dft.real)+0.05))
                plt.axis((xaxis[0],xaxis[1],yaxis[0],yaxis[1]))
                plt.xlabel(x_prefix+'Hz')
                #plt.xlabel('$\Omega$')
                plt.ylabel('Real')
                plt.subplot(212)
                plt.plot(numpy.arange(self.nsamples)*x_step + x_range[0],dft.imag,"b.", label=label)
                plt.plot(numpy.arange(self.nsamples)*x_step + x_range[0],dft.imag, label=label)
                yaxis = (min(-0.1, min(dft.imag)-0.05),max(0.1,max(dft.imag)+0.05))
                plt.axis((xaxis[0],xaxis[1],yaxis[0],yaxis[1]))
                plt.xlabel(x_prefix+'Hz')
                #plt.xlabel('$\Omega$')
                plt.ylabel('Imag')

    def spectrum(self,xaxis=None,yaxis=None,title="",npoints=None):
        if npoints is None: npoints = 256000
        npoints = min(self.nsamples,npoints)
        waveform = self.slice(0,npoints)
        if waveform.domain == 'frequency':
            waveform.plot(xaxis=xaxis,yaxis=yaxis,title=title)
        else:
            waveform.dft().plot(xaxis=xaxis,yaxis=yaxis,title=title)

    def eye_diagram(self,samples_per_symbol,title="Eye diagram"):
        assert self.domain == 'time',"eye_diagram: only valid for time domain waveforms"
        plt.figure()
        plt.title(title)
        nright = self.nsamples - self.nsamples % (2*samples_per_symbol)
        # reshape samples into ? rows by 2*samples_per_symbol columns
        psamples = numpy.reshape(self.samples[:nright],(-1,2*samples_per_symbol))
        # plot plots 2D arrays column-by-column
        plt.plot(psamples.T)

    def noise(self,distribution='normal',amplitude=1.0,loc=0.0,scale=1.0):
        return self + noise(self.nsamples,self.sample_rate,
                            distribution=distribution,
                            amplitude=amplitude,loc=loc,scale=scale)

    def dft(self):
        assert self.domain=='time',\
               'dft: can only apply to time domain waveforms'
        return sampled_waveform(numpy.fft.fft(self.samples),
                                sample_rate=self.sample_rate,
                                domain='frequency')

    def idft(self):
        assert self.domain=='frequency',\
           'idft: can only apply to frequency domain waveforms'
        return sampled_waveform(numpy.fft.ifft(self.samples),
                                sample_rate=self.sample_rate,
                                domain='time')

    def delay(self,nsamples=None,deltat=None):
        assert self.domain == 'time',\
               "delay: can only delay a time-domain waveform"
        assert (nsamples is not None or deltat is None) or \
               (nsamples is None or deltat is not None),\
               "delay: Exactly one of nsamples and delta must be specified"
        if nsamples is None:
            nsamples = int(float(self.sample_rate)/deltat)

        # Keep delayed result periodic
        result = numpy.copy(self.samples)
        if nsamples > 0:
            result[nsamples:] = self.samples[:-nsamples]
            result[0:nsamples] = self.samples[-nsamples:]
        return sampled_waveform(result,sample_rate=self.sample_rate)

    def resample(self, sample_rate, bw=None, ntaps=101, gain=1.0):
        nyquist = min(self.sample_rate,sample_rate)/2.0
        if bw is None: bw = nyquist
        else: bw = min(bw,nyquist)

        if self.sample_rate == sample_rate and bw == sample_rate/2.0:
            return self

        if self.sample_rate >= sample_rate:
            # decimation required
            fdecimation = float(self.sample_rate)/sample_rate
            decimation = int(fdecimation)
            assert sample_rate*decimation == self.sample_rate,\
                   "resample: required decimation (%g) must be an integer" % fdecimation
            # apply anti-aliasing filter
            samples = numpy.convolve(fir_taps('low-pass',bw,sample_rate,ntaps=ntaps,gain=gain),
                                     self.samples)
            if decimation != 1:
                samples = samples[::decimation]
        else:
            # interpolation required
            finterpolation = float(sample_rate)/self.sample_rate
            interpolation = int(finterpolation)
            assert self.sample_rate*interpolation == sample_rate,\
                   "resample: required interpolation (%g) must be an integer" % finterpolation
            samples = numpy.zeros(interpolation*self.nsamples,dtype=numpy.float64)
            samples[::interpolation] = self.samples
            # apply reconstruction filter
            samples = numpy.convolve(fir_taps('low-pass',bw,sample_rate,
                                              ntaps=ntaps,gain=gain*interpolation),
                                     samples)

        return sampled_waveform(samples,sample_rate=sample_rate,domain='time')

    def removeAudioBelowThreshold(self, threshold=0.02): 
        assert threshold >= 0.0
        assert threshold <= 1.0
        
        self.samples = [sample for sample in self.samples if abs(sample)>threshold]
        self.nsamples = len(self.samples)
        
        assert self.nsamples > 0
        return True 
        
        
        
# ##############################################################################

class sinusoid(sampled_waveform):
    """
    Modified from: http://web.mit.edu/6.02/www/f2010/handouts/labs/lab7/waveforms.py
    """
    def __init__(self, nsamples=256e3, hz=1000, sample_rate=256e3, amplitude=1.0, phase=0.0):
        assert hz <= sample_rate/2,"hz cannot exceed %gHz" % (sample_rate/2)
        phase_step = (2*numpy.pi*hz)/sample_rate
        temp = numpy.arange(nsamples,dtype=numpy.float64) * phase_step + phase
        numpy.cos(temp,temp)
        numpy.multiply(temp,amplitude,temp)
        sampled_waveform.__init__(self,temp,sample_rate=sample_rate)

def sin(nsamples=256e3, hz=1000, sample_rate=256e3, phase=0.0):
    return sinusoid(nsamples=nsamples,hz=hz,sample_rate=sample_rate,phase=-numpy.pi/2+phase)

def cos(nsamples=256e3,hz=1000,sample_rate=256e3,phase=0.0):
    return sinusoid(nsamples=nsamples,hz=hz,sample_rate=sample_rate,phase=phase)

        
        
class csinusoid(sampled_waveform):
    """
    Modified from: http://web.mit.edu/6.02/www/f2010/handouts/labs/lab7/waveforms.py
    """
    def __init__(self, nsamples=256e3, hz=1000, sample_rate=256e3, amplitude=1.0, phase=0.0):
        assert hz <= sample_rate/2,"hz cannot exceed %gHz" % (sample_rate/2)
        omega = (float(hz)/sample_rate) * 2.0 * numpy.pi
        temp = numpy.arange(0,nsamples) * omega + phase
        sampled_waveform.__init__(self,amplitude*numpy.exp(1j*temp),
                                  sample_rate=sample_rate)
                                  
# distribution 'normal', 'laplace', 'raleigh', 'uniform', 'triangular', 'impulse'
def noise(nsamples, sample_rate, distribution='normal', amplitude=1.0, loc=0.0, scale=1.0):
    if distribution in ('normal','gaussian'):
        noise = numpy.random.normal(loc,scale,size=nsamples)
    elif distribution == 'uniform':
        noise = numpy.random.uniform(loc-scale,loc+scale,size=nsamples)
    elif distribution == 'triangular':
        noise = numpy.random.triangular(loc-scale,loc,loc+scale,size=nsamples)
    elif distribution == 'impulse':
        # from gr_random.cc in the gnuradio code
        # scale: 5 => scratchy, 8 => geiger
        noise = numpy.random.uniform(size=nsamples)
        numpy.log(noise,noise)
        numpy.multiply(noise,-1.4142135623730951,noise)   # * -sqrt(2)
        noise[noise < scale] = 0.0
    elif distribution == 'laplace':
        noise = numpy.random.laplace(loc,scale,size=nsamples)
    elif distribution == 'raleigh':
        noise = numpy.random.raleigh(scale,size=nsamples)
    else:
        assert False,"unrecognized distribution %s" % distribution
    if amplitude != 1.0:
        numpy.multiply(noise,amplitude,noise)
    return sampled_waveform(noise,sample_rate=sample_rate,domain='time')

    
def wavfile(filename,gain=None,sample_rate=None,bw=None,nsamples=None):
    result = read_wavfile(filename,gain=gain)[0]
    if sample_rate:
        result = result.resample(sample_rate=sample_rate,bw=bw,ntaps=101)
    if nsamples:
        result.resize(nsamples)
    return result

    
def getNewmanPhase(*args, **kwargs):
    """ 
    Calculates Newman Phase
    Generates phase offset for N superimposed tones with a controlled crest factor
    Crest factor = ~4.6dB when N is a few hundred and slightly < ~4.6dB up to N squared
    
    Reference: 
    Boyd, S.; Multitone signals with low crest factor; IEEE Transactions on Circuits and Systems; 1986
    
    Arguments: 
    numTones: total number of tones
    toneIndex: index of tone (starting with 1)
    """
    numTones = kwargs.get('numTones', args[0] if len(args)>0 else 1)
    toneIndex = kwargs.get('toneIndex', args[1] if len(args)>1 else 1)
    assert numTones==int(numTones)
    assert toneIndex==int(toneIndex)
    assert toneIndex>0
    
    phase = (numpy.pi * numpy.power(toneIndex-1,2)) / float(numTones)
    phase = phase % (2*numpy.pi)
    return phase
    

def multitone(*args, **kwargs):
    """
    Generates a multitone audio source 
    """
    numSamples = kwargs.get('nsamples', 1024*5)
    sampleRate = kwargs.get('sample_rate', 16000)
    toneFreqs = kwargs.get('toneFrequencies', [800,1000,1200,1500])
    peakAmplitude = kwargs.get('peakAmplitude', 0.8)
    multitoneMethod = kwargs.get('multitoneMethod', getNewmanPhase)
    
    numTones = len(toneFreqs)
    silentAudio = numpy.zeros(numSamples, dtype=numpy.int)
    multitone = sampled_waveform(silentAudio, sample_rate=sampleRate)
    for toneIndex,toneFreq in enumerate(toneFreqs): 
        phase = multitoneMethod(numTones, toneIndex+1)
        tone = sin(nsamples=numSamples, hz=toneFreq, sample_rate=sampleRate, phase=phase)
        multitone += tone
        
    scalingFactor = float(peakAmplitude) / max(multitone)
    multitone *= scalingFactor
    return multitone    


def chirp(*args, **kwargs):
    """
    Generates a linear chirp audio source
    
    References: 
    http://en.wikipedia.org/wiki/Chirp#Linear_chirp
    http://en.wikipedia.org/wiki/Chirp#Exponential_chirp
    
    Arguments: 
    numSamples: total number of audio samples in output
    sample_rate: audio sample rate
    peakAmplitude: peak amplitude in output
    freqStart: starting frequency of chirp
    freqStop: ending frequency of chirp
    """
    numSamples = kwargs.get('nsamples', 1024*5)
    sampleRate = kwargs.get('sample_rate', 8000)
    peakAmplitude = kwargs.get('peakAmplitude', 0.8)
    freqStart = kwargs.get('freqStart', 100)
    freqStop = kwargs.get('freqStop', 4000)
    method = kwargs.get('method', 'linear')
    assert freqStart > 0
    assert freqStart < freqStop
    assert freqStop <= sampleRate / 2
    
    chirp = None
    if method is 'linear': 
        chirpRate = (freqStop - freqStart) / float(numSamples)
        samplePts = numpy.arange(numSamples, dtype=numpy.float64)
        f = freqStart + 0.5 * chirpRate * samplePts
        chirp = numpy.sin(2.0 * numpy.pi * (f * samplePts) / sampleRate)
    elif method is 'exponential': 
        chirpRate = (freqStop/freqStart)**(1 / float(numSamples))
        samplePts = numpy.arange(numSamples, dtype=numpy.float64)
        chirp = numpy.sin((2.0 * numpy.pi * freqStart / numpy.log(chirpRate)) * (chirpRate**samplePts - 1.0) / sampleRate)
    else: 
        raise Exception('Unkown chirp method')
        
    scalingFactor = float(peakAmplitude) / max(chirp)
    chirp *= scalingFactor
    return sampled_waveform(chirp, sample_rate=sampleRate, domain='time')

    
    
# ##############################################################################
# helper functions
# ##############################################################################

def eng_notation(range):
    """
    Helper function looks at range of values and returns a new range,
    Modified from: http://web.mit.edu/6.02/www/f2010/handouts/labs/lab7/waveforms.py
    
    Arguments: 
    range: Range of values
    
    Returns: 
    Engineering prefix and the scale factor
    """
    x = max(abs(range[0]),abs(range[1]))   # find largest value
    if x <= 1e-12:
        scale = 1e15
        units = "f"
    elif x <= 1e-9:
        scale = 1e12
        units = "p"
    elif x <= 1e-6:
        scale = 1e9
        units = "n"
    elif x <= 1e-3:
        scale = 1e6
        units = "u"
    elif x <= 1:
        scale = 1e3
        units = "m"
    elif x >= 1e9:
        scale = 1e-9
        units = "G"
    elif x >= 1e6:
        scale = 1e-6
        units = "M"
    elif x >= 1e3:
        scale = 1e-3
        units = "k"
    else:
        scale = 1
        units = ""
    return ((range[0]*scale,range[1]*scale),units,scale)

    
def calcRms(data): 
    """
    Function to calculate the RMS value of a dataset 
    """
    return numpy.sqrt(float(sum(numpy.power(x,2) for x in data))/len(data))

def calcCrestFactor(data): 
    """
    Function to calculate the RMS value of a dataset 
    """
    return max(data)/calcRms(data)
