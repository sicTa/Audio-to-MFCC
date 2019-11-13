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

        '''
        ndarray
        An array object represents a multidimensional,
        homogeneous array of fixed-size items.
        '''
        return mfcc
        
        
    def plot(self, spectrum):
        plt.imshow(spectrum.T, cmap=plt.cm.jet, aspect='auto')
        ax = plt.gca()
        ax.invert_yaxis()
        plt.show()


    def feature_vector(self, signal_name, sample_rate = None, pre_emphasis = None, frame_size = None, frame_stride = None, NFFT = None, nfilt = None, num_ceps = None, N = 2):
        '''
        Forms a feature vector list from a mfcc list containg the mfcc and delta(dirst derivate)
        and delta-delta(second derivative)

        Each vector has 39 dimensions:
            12 mfcc coefficients and their mean
            12 delta coefficients and their mean
            12 delta-delta coefficients and their mean

        For more information visit
        http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
        for the formula
        '''


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

        number_of_vectors = numpy.size(mfcc, 0)
        
        
        
        feature = []  #makes an empty list
        #delta = numpy.zeros((number_of_vectors, 12))
        #delta_delta = numpy.zeros((number_of_vectors, 12))
        
        mfcc_list = []
        delta = [[0]*13 for i in range(number_of_vectors)]             #makes a matrix of zeros   num_of_vec X 13
        delta_delta = [[0]*13 for i in range(number_of_vectors)]       #makes a matrix of zeros   num_of_vec X 13

        denominator = 0
        for i in range(1, N + 1):
            denominator = denominator + i**2
        denominator*=2


        #now, on every mfcc vector we append the energy of a signal
        
        
        for i in range(number_of_vectors):
            energy = 0
            for j  in range(frame_length):
                energy += frames[i][j]**2
                
                
            if energy == 0:
                energy  = 1e-3
            energy = numpy.log(energy)
            curr_list = mfcc[i].tolist()
            curr_list.append(energy)
            feature.append(curr_list)
            mfcc_list.append(curr_list)
            
            

        
        for t in range(number_of_vectors):
            for n  in range(1, N + 1):  
                if t - n < 0:
                    c_minus = mfcc_list[t]
                else:
                    c_minus = mfcc_list[t - n]

                if t + n > number_of_vectors - 1:
                    c_plus = mfcc_list[t]
                else:
                    c_plus = mfcc_list[t + n]

                for j in range(13):     
                    delta[t][j] += n*(c_plus[j] + c_minus[j])/denominator

        for t in range(number_of_vectors):
            for n  in range(1, N + 1):  
                if t - n < 0:
                    c_minus = delta[t]
                else:
                    c_minus = delta[t - n]

                if t + n > number_of_vectors - 1:
                    c_plus = delta[t]
                else:
                    c_plus = delta[t + n]

                #we need to calculate this element by element
                for j in range(13):     
                    delta_delta[t][j] += n*(c_plus[j] + c_minus[j])/denominator
                    
                    
        #print(delta[100])
        #print(delta_delta[100])

        #now, calculate the mean for every vector and append it
        #every vector in the list of features
        #is of length 3 len of single mfcc vector


        for t in range(number_of_vectors):
            curr_mfcc =  mfcc_list[t]
            curr_delta = delta[t]
            curr_delta_delta = delta_delta[t]
            #feature.append([])
            #feature[t]  = []

            curr = []
            curr +=     (curr_mfcc +   curr_delta + curr_delta_delta)
            
            feature[t] = curr
            
            


        return feature
    

    def normalized_feature_vector(self, signal_name, sample_rate = None, pre_emphasis = None, frame_size = None, frame_stride = None, NFFT = None, nfilt = None, num_ceps = None, N = 2):
      
        features = self.feature_vector(signal_name, sample_rate, pre_emphasis, frame_size, frame_stride, NFFT, nfilt, num_ceps, N)
        
        

        num_of_vecs = len(features)
        vec_dim = len(features[0])
        

        for j in range(vec_dim):
            #normalazing the mean to zero
            mean = 0
            for i in range(num_of_vecs):
                mean += features[i][j]
            mean /= num_of_vecs
            for i in range(num_of_vecs):
                features[i][j] -= mean
                 
        #normalizing the standard deviation to one
            deviation = 0
            for i in range(num_of_vecs):
                deviation += (features[i][j])**2      #we do not subtract mean because mean is now 0
            deviation /= (num_of_vecs - 1)
            deviation = numpy.sqrt(deviation)
            for i in range(num_of_vecs):
                features[i][j] /= deviation
                
                
        #now, changing the range of all the variables to [0,1]
        
        for j in range(vec_dim):
            minimum = 50000
            maximum = -5000
            for i in range(num_of_vecs):
                if features[i][j] > maximum:
                    maximum = features[i][j]
                if features[i][j] < minimum:
                    minimum = features[i][j]
                    
            oldRange = maximum - minimum
            newRange = 1 - 0
            new_minimum = 0
           
            for i in range(num_of_vecs):
                features[i][j] = (features[i][j] - minimum)*newRange/oldRange + new_minimum
               
        return features
                       
        

        
        
        
        
        

            

            
                
             
            

        

        
        
        
        
            
        
        
        
