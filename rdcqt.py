# -*- coding: utf-8 -*-
"""
Constant-Q Transform using recursive downsampling method.

Written by kurene@https://www.wizard-notes.com

References:

  - Schörkhuber, Christian, and Anssi Klapuri. 
    "Constant-Q transform toolbox for music processing."
    7th Sound and Music Computing Conference, Barcelona, Spain. 2010.
"""
import numpy as np
from scipy.signal import butter, lfilter, bessel, filtfilt

        
class RDCQT():
    def __init__(self, 
        sr,
        n_octave=8,         # num. of octaves
        n_bpo=12,           # num. of bins per octave.
        f0=55/2,            # lowest center frequency
        window=np.hanning,  # window func.
        lpf_order=6,        # order of lowpass filter using recursive downsampling
        hop_length=128,
        winlenmax=None,
        q_rate=1.0,
        sp_thresh=0.0,
    ):
        # Params
        self.sr         = sr
        self.nyq        = sr / 2
        self.n_octave   = n_octave
        self.n_bpo      = n_bpo 
        self.hop_length = hop_length
        
        # calc. center freqs.
        self.cqt_freqs = self.compute_cqt_freqs(f0)
        self.n_pitch   = len(self.cqt_freqs)
        
        # calc. Q and Nk(winlens)
        Q            = q_rate / (2 **(1.0 / self.n_bpo) - 1.0)
        self.winlens = np.ceil(Q * sr / self.cqt_freqs).astype(np.int64)
        
        self.lpf_order = lpf_order
        #self.lpf_b, self.lpf_a = butter(self.lpf_order, 0.5, btype='low')
        self.lpf_b, self.lpf_a = bessel(self.lpf_order, 0.5, btype='low')


        # calc. kernel size
        self.n_fft_exp  = len(bin(int(self.winlens[0])))-2
        self.n_fft      = 2 ** (self.n_fft_exp)
        self.n_fft_1oct = 2 ** (self.n_fft_exp - (self.n_octave - 1))
        
        # 1-octave temporal/spectral kernel
        self.temporal_kernel = np.zeros((self.n_bpo, self.n_fft_1oct), 
                                         dtype=np.complex128)
        self.spectral_kernel = self.temporal_kernel.copy()


        # calc. 1-octave kernel
        coef = 2.0 * np.pi * 1j
        for m, k in enumerate(range(self.n_pitch-self.n_bpo, self.n_pitch)):
            Nk = self.winlens[k]
            Fk = self.cqt_freqs[k]
            
            #print(m, k, int(Nk), int(Fk), self.n_fft, self.n_fft_1oct)
            st = int((self.n_fft_1oct - Nk) / 2)
            en = Nk + st
            tmp_kernel = np.exp(coef * (Fk/sr) * np.arange(0, Nk))
            
            self.temporal_kernel[m, st:en] = tmp_kernel * (window(Nk) / Nk) 
            self.spectral_kernel[m]        = np.fft.fft(self.temporal_kernel[m])

        # Sparse thresholding 
        self.spectral_kernel[np.abs(self.spectral_kernel) <= sp_thresh] = 0.0

        self.spectral_kernel = self.spectral_kernel.conjugate() / self.n_fft
        
        # tmp variables in the transform
        self.x_list = np.zeros((self.n_octave, self.n_fft))
        self.tmp_x  = np.zeros(self.n_fft)
        self.xd_fft = np.zeros(self.n_fft_1oct, dtype=np.complex128)
        self.y      = np.zeros(self.n_pitch, dtype=np.complex128)
        
        
    def plot_kernel(self):
        import matplotlib.pyplot as plt
        plt.clf()
        plt.subplot(2,1,1)
        plt.imshow(np.abs(self.temporal_kernel), origin="lower", aspect="auto", cmap="jet")
        plt.colorbar()
        plt.title("Temporal kernel")
        plt.subplot(2,1,2)
        plt.imshow(np.abs(self.spectral_kernel), origin="lower", aspect="auto", cmap="jet")
        plt.colorbar()
        plt.title("Spectral kernel")
        plt.tight_layout()
        plt.show()  

    def compute_cqt_freqs(self, f0):
        cqt_freqs = []
        for m in range(self.n_octave):
            for n in range(self.n_bpo):
                k = m * self.n_bpo + n
                fk = f0 * 2 ** (k / self.n_bpo)
                cqt_freqs.append(fk)
        return np.array(cqt_freqs)

    def stcqt(self, x):
        """
            short-time constant q transform for offline proc.
        """
        n_frames = len(x) // self.hop_length
        x_tmp = np.r_[x, np.zeros(self.n_fft)]
        spec = np.zeros([n_frames, self.n_pitch], dtype=np.complex128)
        
        for k in range(n_frames):
            st = k * self.hop_length
            en = st + self.n_fft
            spec[k] = self.cqt(x_tmp[st:en])
        return spec.T
    
    def cqt(self, x):
        # zero clear
        self.x_list[:, :] = 0.0
        self.tmp_x[:]     = 0.0
        self.y[:]         = 0.0

        # set x
        length = len(x)
        if length < self.n_fft:
            self.x_list[0, 0:length] = x
            self.x_list[0, length::] = 0.0
        else:
            self.x_list[0, :] = x[0:self.n_fft]
            
        # subroutine
        _rdcqt(self.spectral_kernel, 
             self.xd_fft, 
             self.x_list, self.tmp_x, self.y,
             self.n_fft, self.n_fft_1oct, 
             self.n_bpo, self.n_octave, 
             self.lpf_b, self.lpf_a)

        return self.y[:]

            
def _rdcqt(spectral_kernel, 
         xd_fft, 
         x_list, tmp_x, y,
         n_fft, n_fft_1oct, 
         n_bpo, n_octave, 
         lpf_b, lpf_a):
         
    # recursive downsampling
    for k in range(1, n_octave):
        idx1 = n_fft // 2 ** (k-1)
        idx2 = idx1 // 2
        tmp_x[0:idx1]     = lfilter(lpf_b, lpf_a, x_list[k-1, 0:idx1]) #filtfilt
        #tmp_x[0:idx1]     = x_list[k-1, 0:idx1] # w/o lpf
        x_list[k, 0:idx2] = tmp_x[0:idx1][::2]

    # calc. cqt spectrum using Parseval's theorem
    for k in range(0, n_octave): 
        center = n_fft // 2 ** (k + 1)
        st = center - n_fft_1oct//2
        en = center + n_fft_1oct//2    
        st2 = n_bpo * (n_octave-k-1)
        en2 = st2 + n_bpo
        
        xd_fft[:] = np.fft.fft(x_list[k, st:en])
        y[st2:en2] = np.dot(spectral_kernel, xd_fft)

  
class MusicRDCQT(RDCQT):
    def __init__(self, 
        sr,
        min_pitch="A1",
        n_octave=6,
        tuning=440,
        add_bins_end=0,
        chroma_labels = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"],
        **kwargs
    ):
        n_chroma = len(chroma_labels)
        n_bins   = int(n_octave*n_chroma)
        # e.g. A0 (27.5Hz) is 0, A1(50Hz) is 12 (if tuning==440)
        calc_freq_from_pitch_index = lambda _k : tuning * 2 ** (_k/n_chroma - 4) 
        
        min_pitch_class = min_pitch[0].upper()
        min_pitch_num   = int(min_pitch[1])
        min_pitch_index = chroma_labels.index(min_pitch_class) + n_chroma * min_pitch_num
        
        f0   = calc_freq_from_pitch_index(min_pitch_index)
        
        super(MusicCQT, self).__init__(sr, n_octave=n_octave, f0=f0, n_bpo=n_chroma, **kwargs)
        self.min_pitch = min_pitch
        
        self.n_chroma = n_chroma
        self.labels = [ chroma_labels[k%n_chroma]+str(k//n_chroma) for k in range(min_pitch_index, min_pitch_index + n_bins)]
        

if __name__ == "__main__":
    import sys
    import time
    import librosa
    import matplotlib.pyplot as plt
    from cqt import CQT
    
    
    sr = 44100
    y, sr = librosa.load(sys.argv[1], sr=sr, mono=True)
    
    rdcqt     = RDCQT(sr)
    cqt       = CQT(sr, sparse_computation=True)
    
    start_time = time.time()
    spec_cqt = cqt.stcqt(y)
    print(f"proctime: {time.time() - start_time:1.3f}")
    
    start_time = time.time()
    spec_rdcqt = rdcqt.stcqt(y)
    print(f"proctime: {time.time() - start_time:1.3f}")
    
    amplitude_to_db = lambda x : np.sqrt(x)/np.max(np.sqrt(x)) #librosa.amplitude_to_db(x, ref=np.max)
    
    plt.clf()
    plt.subplot(1,2,1)
    plt.imshow(amplitude_to_db(np.abs(spec_cqt)),  origin="lower", aspect="auto", cmap="jet")
    plt.colorbar()
    plt.title("CQT: Sparse matrix computation")
    plt.subplot(1,2,2)
    plt.imshow(amplitude_to_db(np.abs(spec_rdcqt)), origin="lower", aspect="auto", cmap="jet")
    plt.title("CQT: Recursive downsampling method")
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()  

    import code
    console = code.InteractiveConsole(locals=locals())
    console.interact()