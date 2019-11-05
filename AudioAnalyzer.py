import numpy
import scipy.io.wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt


class AudioAnalyzer():
    
    def __init__(self, sample_rate = 8000, pre_emphasis= 0.97, frame_size = 0.025, frame_stride = 0.01, NFFT = 512, nfilt = 40, num_ceps = 12):
        
        
        self.sample_rate = sample_rate
        self.pre_emphasis = pre_emphasis
        self.frame_size = frame_size
        self.frame_stride = frame_stride
        self.NFFT = NFFT
        self.nfilt = nfilt
        self.num_ceps = num_ceps
        
        
    def signal_to_array(self, signal_name):
        self.sample_rate, signal = scipy.io.wavfile.read(signal_name)
        return signal
    
    def pre_emphasis_signal(self, signal_name):
        '''
        PRE-EMPHASIS
        The first step is to apply a pre-emphasis filter on the signal to amplify the high frequencies. 
        A pre-emphasis filter is useful in several ways: 
        (1) balance the frequency spectrum since 
        high frequencies usually have smaller magnitudes compared to lower frequencies, 
        (2) avoid numerical problems during the Fourier transform operation and 
        (3) may also improve the Signal-to-Noise Ratio (SNR).
        '''
        self.sample_rate, signal = signal_to_array(signal_name)
        emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
        return emphasized_signal
    
    def framing(self, signal_name, frame_size = None, frame_stride = None):
        '''
        Returns a list of frames of a signal
        '''
        
        self.sample_rate, signal = scipy.io.wavfile.read(signal_name)
        emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
        
        if frame_size ==  None:
            frame_size = self.frame_size
        if frame_stride ==  None:
            frame_stride = self.frame_stride
            
        frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
        signal_length = len(emphasized_signal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

        pad_signal_length = num_frames * frame_step + frame_length
        z = numpy.zeros((pad_signal_length - signal_length))
        pad_signal = numpy.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

        indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = pad_signal[indices.astype(numpy.int32, copy=False)]
        
        return frames
    
    def HammingWindow(self, frames):
        '''
        Applies a Hamming window to a list of frames
        '''
        frame_length, frame_step = self.frame_size * self.sample_rate, self.frame_stride * self.sample_rate
        frames *= numpy.hamming(self.frame_length)
    
    def FFT(self, signal_name, frame_size = None, frame_stride = None):
        '''
        Applies a FFT to a signal in .wav format
        '''
        self.sample_rate, signal = scipy.io.wavfile.read(signal_name)
        emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
        
        if frame_size ==  None:
            frame_size = self.frame_size
        if frame_stride ==  None:
            frame_stride = self.frame_stride
            
        frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
        signal_length = len(emphasized_signal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

        pad_signal_length = num_frames * frame_step + frame_length
        z = numpy.zeros((pad_signal_length - signal_length))
        pad_signal = numpy.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

        indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = pad_signal[indices.astype(numpy.int32, copy=False)]
        
        
        frames *= numpy.hamming(frame_length)
        
        mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
        pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
        
        return pow_frames
    
    
    def filter_bank(self, signal_name, sample_rate = None, pre_emphasis = None, frame_size = None, frame_stride = None, NFFT = None):
       
        
        if sample_rate == None:
            sample_rate = self.sample_rate
        if pre_emphasis == None:
            pre_emphasis = self.pre_emphasis
        if frame_size == None:
            frame_size = self.frame_size
        if frame_stride == None:
            frame_stride = self.frame_stride
        if NFFT == None:
            NFFT = self.NFFT
        if nfilt == None:
            nfilt = self.nfilt
        
        sample_rate, signal = scipy.io.wavfile.read(signal_name)  # File assumed to be in the same directory
        signal = signal[0:int(3.5 * sample_rate)]  # Keep the first 3.5 seconds
        
        
        emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
        
        
        '''
        FRAMING
        Here we separate our signal into a list of frames
        '''
        frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
        signal_length = len(emphasized_signal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame
        
        pad_signal_length = num_frames * frame_step + frame_length
        z = numpy.zeros((pad_signal_length - signal_length))
        pad_signal = numpy.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
        
        indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = pad_signal[indices.astype(numpy.int32, copy=False)]
        
        
        '''
        WINDOWING
        A Hamming window is overlayed over each frame
        '''
        frames *= numpy.hamming(frame_length)
        # frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **
        
        
        '''
        FFT
        
        A Fast-Fourier-Transform is applied to each frame
        '''
        mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
        pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
        
        
        '''
        FILTER BANKS
        Here we apply the Mel scale filter to better approximate the effect
        sound has on the human hearing system. 
        '''
        
        low_freq_mel = 0
        high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
        mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
        bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)
        
        fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])   # left
            f_m = int(bin[m])             # center
            f_m_plus = int(bin[m + 1])    # right
        
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filter_banks = numpy.dot(pow_frames, fbank.T)
        filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
        filter_banks = 20 * numpy.log10(filter_banks)  # dB
        
        filter_banks -= (numpy.mean(filter_banks,axis=0) + 1e-8)
        return filter_banks
    
    
    def MFCC(self, signal_name, sample_rate = None, pre_emphasis = None, frame_size = None, frame_stride = None, NFFT = None, nfilt = None, num_ceps = None):
        
        
        if sample_rate == None:
            sample_rate = self.sample_rate
        if pre_emphasis == None:
            pre_emphasis = self.pre_emphasis
        if frame_size == None:
            frame_size = self.frame_size
        if frame_stride == None:
            frame_stride = self.frame_stride
        if NFFT == None:
            NFFT = self.NFFT
        if nfilt == None:
            nfilt = self.nfilt
        if num_ceps == None:
            num_ceps = self.num_ceps
        
        sample_rate, signal = scipy.io.wavfile.read(signal_name)  # File assumed to be in the same directory
        signal = signal[0:int(3.5 * sample_rate)]  # Keep the first 3.5 seconds
        
        
        emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
        
        
        '''
        FRAMING
        Here we separate our signal into a list of frames
        '''
        frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
        signal_length = len(emphasized_signal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame
        
        pad_signal_length = num_frames * frame_step + frame_length
        z = numpy.zeros((pad_signal_length - signal_length))
        pad_signal = numpy.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
        
        indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = pad_signal[indices.astype(numpy.int32, copy=False)]
        
        
        '''
        WINDOWING
        A Hamming window is overlayed over each frame
        '''
        frames *= numpy.hamming(frame_length)
        # frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **
        
        
        '''
        FFT
        
        A Fast-Fourier-Transform is applied to each frame
        '''
        mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
        pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
        
        
        '''
        FILTER BANKS
        Here we apply the Mel scale filter to better approximate the effect
        sound has on the human hearing system. 
        '''
        
        low_freq_mel = 0
        high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
        mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
        bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)
        
        fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])   # left
            f_m = int(bin[m])             # center
            f_m_plus = int(bin[m + 1])    # right
        
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filter_banks = numpy.dot(pow_frames, fbank.T)
        filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
        filter_banks = 20 * numpy.log10(filter_banks)  # dB
        
        filter_banks -= (numpy.mean(filter_banks,axis=0) + 1e-8)
        '''
        MFCC
        Here we caluclate the MFCC's
        '''
        mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
        
        return mfcc
        
        
    def plot(self, spectrum):
        plt.imshow(spectrum.T, cmap=plt.cm.jet, aspect='auto')
        ax = plt.gca()
        ax.invert_yaxis()
        plt.show()
            
        
        
        
