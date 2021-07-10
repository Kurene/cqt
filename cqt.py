# -*- coding: utf-8 -*-
"""
Written by kurene@https://www.wizard-notes.com

References:

  - Brown, Judith C. 
    "Calculation of a constant Q spectral transform." 
    The Journal of the Acoustical Society of America 89.1 (1991): 425-434.
  - Schörkhuber, Christian, and Anssi Klapuri. 
    "Constant-Q transform toolbox for music processing."
    7th Sound and Music Computing Conference, Barcelona, Spain. 2010.
"""

import sys
import time
import numpy as np
from numba import jit
from scipy.sparse import csr_matrix

    
class CQT():
    def __init__(self, 
        sr,
        n_octave=7,
        n_bpo=24,
        f1=55/2,
        fmax=None,
        window=np.hamming,
        sp_thresh=0.005,
        sparse_computation=True,
        hop_length=None,
        winlenmax=None,
        q_rate=1.0       # Schörkhuber, C., & Klapuri, A. , 2010.
    ):
        self.sparse_computation = sparse_computation
        
        self.sr       = sr
        self.nyq      = sr / 2
        self.fmax     = self.nyq if fmax is None else fmax
        self.n_octave = n_octave
        self.n_bpo    = n_bpo # the number of bins per octave.
        
        # calc. center freqs.
        self.cqt_freqs = self.compute_cqt_freqs(f1)
        self.n_pitch   = len(self.cqt_freqs)
        
        # calc. Q and Nk(winlens)
        Q            = q_rate / (2 **(1.0 / self.n_bpo) - 1.0)
        self.winlens = np.ceil(Q * sr / self.cqt_freqs).astype(np.int64)
        self.n_fft   = 2 ** (len(bin(int(self.winlens[0])))-2)
        if winlenmax is not None and self.n_fft > winlenmax:
            self.n_fft = winlenmax
            
        # calc. kernel
        self.a = np.zeros((self.n_pitch, self.n_fft), dtype=np.complex128)
        self.kernel = np.zeros(self.a.shape, dtype=np.complex128)
        coef = 2.0 * np.pi * 1j
        for k in range(self.n_pitch):
            Nk = self.winlens[k]
            Fk = self.cqt_freqs[k]
            st = int((self.n_fft - Nk) / 2)
            en = Nk + st
            print(k, Fk, sr, Fk/sr, Q, Nk*Fk/sr)
            #tmp_a = np.exp(coef * Q * np.arange(0, Nk)/Nk)   
            tmp_a = np.exp(coef * (Fk/sr) * np.arange(0, Nk))
            if winlenmax is not None and Nk > winlenmax:
                # truncate kernel
                tmp_a = tmp_a[-st:-st+self.n_fft]
                self.winlens[k] = winlenmax
                Nk = self.winlens[k]
                st, en = 0, self.n_fft

            self.a[k, st:en] = tmp_a * (window(Nk) / Nk) 
            self.kernel[k]   = np.fft.fft(self.a[k])
        
        # prepare sparse computation
        self.kernel[np.abs(self.kernel) <= sp_thresh] = 0.0
        if self.sparse_computation:
            self.kernel_sparse = csr_matrix(self.kernel)
        else:
            self.kernel_sparse = self.kernel
        self.kernel_sparse = self.kernel_sparse.conjugate() / self.n_fft
        
        # tmp variables for transform
        self.x = np.zeros(self.n_fft)
        self.y = np.zeros(self.n_pitch, dtype=np.complex128)
        self.hop_length = int(hop_length if hop_length is not None else 512)#sr * 0.1)

    def plot_kernel(self):
        import matplotlib.pyplot as plt
        plt.clf()
        plt.subplot(2,1,1)
        plt.imshow(np.abs(self.a), origin="lower", aspect="auto", cmap="jet")
        plt.colorbar()
        plt.subplot(2,1,2)
        plt.imshow(np.abs(self.kernel), origin="lower", aspect="auto", cmap="jet")
        plt.colorbar()
        plt.show()  
        
    def compute_cqt_freqs(self, f1):
        cqt_freqs = []
        for m in range(self.n_octave):
            for n in range(self.n_bpo):
                k = m * self.n_bpo + n
                fk = f1 * 2 ** (k / self.n_bpo)
                if fk <= self.fmax:
                    cqt_freqs.append(fk)
        return np.array(cqt_freqs)

    def stcqt(self, x):
        """
            short-time constant q transform for batch
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
        length = len(x)
        
        if length < self.n_fft:
            self.x[0:length] = x
            self.x[length::] *= 0.0
        else:
            self.x[:] = x[0:length]
            
        if self.sparse_computation:
            self.y[:] = np.fft.fft(self.x[:]) * self.kernel_sparse.T
        else:
            self.y[:] = np.dot(self.kernel_sparse, np.fft.fft(self.x[:]))
            
        return self.y


class MusicCQT(CQT):
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
        
        f1   = calc_freq_from_pitch_index(min_pitch_index)
        fmax = calc_freq_from_pitch_index(min_pitch_index + n_bins + add_bins_end)
        #print("f1, fmax", f1, fmax)
        
        super(MusicCQT, self).__init__(sr, n_octave=n_octave, f1=f1, fmax=fmax, n_bpo=n_chroma, **kwargs)
        self.min_pitch = min_pitch
        
        self.n_chroma = n_chroma
        self.labels = [ chroma_labels[k%n_chroma]+str(k//n_chroma) for k in range(min_pitch_index, min_pitch_index + n_bins)]
        

if __name__ == "__main__":
    import librosa
    import matplotlib.pyplot as plt
    sr = 44100
    y, sr = librosa.load(sys.argv[1], sr=sr, mono=True)
    
    cqt = MusicCQT(sr, sparse_computation=True, winlenmax=None)
    start_time = time.time()
    spec = cqt.stcqt(y)
    print(f"proctime: {time.time() - start_time:1.3f}")
    start_time = time.time()
    spec2 = librosa.cqt(y, sr=sr, hop_length=512, n_bins=72,
                        fmin=librosa.note_to_hz('A1'),
                        window="hamming", sparsity=0.005, scale=True)
    print(f"proctime: {time.time() - start_time:1.3f}")
    
    amplitude_to_db = lambda x : np.sqrt(x)
    #amplitude_to_db = lambda x : librosa.amplitude_to_db(x, ref=np.max)
    plt.clf()
    plt.subplot(1,2,1)
    plt.imshow(amplitude_to_db(np.abs(spec)),  origin="lower", aspect="auto", cmap="jet")
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(amplitude_to_db(np.abs(spec2)), origin="lower", aspect="auto", cmap="jet")
    plt.title("librosa.cqt")
    plt.colorbar()
    plt.show()  

    import code
    console = code.InteractiveConsole(locals=locals())
    console.interact()