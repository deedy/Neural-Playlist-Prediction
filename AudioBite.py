import matplotlib.mlab as mlab # Spectrograms
from PIL import Image # Matrix to images
from mpl_toolkits.mplot3d import Axes3D
from scipy.fftpack.realtransforms import dct #MFCC computation
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import time
import os
import cPickle as pickle

from IPython.core.debugger import Tracer

SAVE_DIR = 'processed'

SPECTROGRAM_BINS = 1024
DEFAULT_WINDOW_SIZE = 4096
DEFAULT_OVERLAP_RATIO = 0.5
DEFAULT_NUM_MEL_BANDS = 128

DEFAULT_MFCC_BANDS = 26
DEFAULT_NUM_MFCC_COEFFICIENTS = 13
DEFAULT_MFCC_DELTA_N = 2

MIN_MEL_FREQUENCY = 0
MAX_MEL_FREQUENCY = 14500

def hz2mel(hz):
    return 1127.0 * np.log(1 + hz / 700.0)

def mel2hz(mel):
    return (np.exp(mel / 1127.0) - 1) * 700

class AudioBite():
  def __init__(self, path, data, frequency):
    self.data = data
    self.num_samples = len(data)
    self.original_path = path
    _, ext  = os.path.splitext(path)
    self.filetype = ext
    self.frequency = frequency
    self.specgram_freq_bins = SPECTROGRAM_BINS

    spectrogram_start_time = time.time()
    print('Computing Spectrogram ...')
    self.specgram, self.spec_frequencies, self.frame_bins = self.__init_spectrogram()
    print('Done computing spectrogram in {0} seconds.\n'.format(time.time() - spectrogram_start_time))

    mfcc_start_time = time.time()
    print('Computing MFCCs ...\n')
    self.mfcc_cep, self.mfcc_delta, self.mfcc_delta_deltas = self.__init_mfcc()
    print('Done computing MFCC in {0} seconds.\n'.format(time.time() - mfcc_start_time))

    print('Computing Mel Spectrogram ...')
    mel_spectrogram_start_time = time.time()
    self.mel_specgram, self.mel_spec_frequencies = self.__init_mel_spectrogram()
    print('Done computing mel spectrogram in {0} seconds.\n'.format(time.time() - mel_spectrogram_start_time))


  def __init_spectrogram(self,
      nfft = DEFAULT_WINDOW_SIZE,
      overlap = DEFAULT_OVERLAP_RATIO):
    # matplotlib has 3 different specgram implementations - mlab, pyplot, and on
    # the axes:
    # 1. mlab - [https://github.com/matplotlib/matplotlib/blob/78f1942f5af063abacc3d5709912bde6bbb6ffec/lib/matplotlib/mlab.py]
    # 2. pyplot - [https://github.com/matplotlib/matplotlib/blob/e7717f71abae542ba0a5fea5b072bf52548d5ee3/lib/matplotlib/pyplot.py]
    # 3. axes - [https://github.com/matplotlib/matplotlib/blob/e7717f71abae542ba0a5fea5b072bf52548d5ee3/lib/matplotlib/axes/_axes.py]
    # 1 is the primary implementation. 2 calls 3 and 3 calls 1, but does the
    # added job of plotting after log scaling, and flipping the surface
    # matrix appropriately. We derive the plotting code from theirs.

    Pxx, freqs, bins = mlab.specgram(self.data, NFFT = nfft, Fs = self.frequency, pad_to = self.specgram_freq_bins, noverlap = (int(nfft*overlap)))
    return Pxx, freqs, bins

  def __init_mfcc(self,
      num_mel_bands = DEFAULT_MFCC_BANDS,
      num_mfcc = DEFAULT_NUM_MFCC_COEFFICIENTS,
      delta_N = DEFAULT_MFCC_DELTA_N):
    mel_bin_matrix, freqs = self.get_mel_binning_matrix(num_mel_bands)
    Pxx2 = np.dot(self.specgram.T, mel_bin_matrix)

    # Unlike the mlab implementation, we threshold and log our FFT magnitudes
    # before returning
    Pxx2[Pxx2 < 1e-10] = 1e-10
    Pxx2 = 10. * np.log10(Pxx2)
    Pxx2[Pxx2 <= 0.0] = 0.0

    # http://pydoc.net/Python/scikits.talkbox/0.2.4.dev/scikits.talkbox.features.mfcc/
    ceps = dct(Pxx2, type=2, norm='ortho', axis=-1)[:, :num_mfcc]
    ceps = np.flipud(ceps)
    deltas = np.zeros(ceps.shape)
    delta_deltas = np.zeros(ceps.shape)
    for cep_frame_i in xrange(len(ceps)):
      if cep_frame_i < delta_N:
        del_N = cep_frame_i
      elif cep_frame_i > len(ceps) - delta_N - 1:
        del_N = len(ceps) - cep_frame_i - 1
      else:
        del_N = delta_N
      if del_N == 0:
        continue
      deltas[cep_frame_i] = sum([n*(ceps[cep_frame_i + n] - ceps[cep_frame_i - n]) for n in xrange(1,del_N+1)]) / (2.0*sum([n**2 for n in xrange(1, del_N + 1)]))
    for cep_frame_i in xrange(len(deltas)):
      if cep_frame_i < delta_N:
        del_N = cep_frame_i
      elif cep_frame_i > len(ceps) - delta_N - 1:
        del_N = len(ceps) - cep_frame_i - 1
      else:
        del_N = delta_N
      if del_N == 0:
        continue
      delta_deltas[cep_frame_i] = sum([n*(deltas[cep_frame_i + n] - deltas[cep_frame_i - n]) for n in xrange(1,del_N+1)]) / (2.0*sum([n**2 for n in xrange(1, del_N + 1)]))
    ceps = ceps.T[1:]
    deltas = deltas.T[1:]
    delta_deltas = delta_deltas.T[1:]
    return ceps, deltas, delta_deltas

  def __init_mel_spectrogram(self,
      num_mel_bands = DEFAULT_NUM_MEL_BANDS):
    mel_bin_matrix, freqs = self.get_mel_binning_matrix(num_mel_bands)
    Pxx2 = np.dot(self.specgram.T, mel_bin_matrix)
    Pxx2 = Pxx2.T

    # Unlike the mlab implementation, we threshold and log our FFT magnitudes
    # before returning
    Pxx2[Pxx2 < 1e-10] = 1e-10
    Pxx2 = 10. * np.log10(Pxx2)
    Pxx2[Pxx2 <= 0.0] = 0.0
    return Pxx2, freqs

  def get_mel_binning_matrix(self, num_mel_bands):
    """
    function that returns a matrix that converts a regular DFT to a mel-spaced DFT,
    by binning coefficients.

    specgram_window_size: the window length used to compute the spectrograms
    sample_frequency: the sample frequency of the input audio
    num_mel_bands: the number of desired mel bands.

    The output is a matrix with dimensions (specgram_window_size/2 + 1, num_bands)
    """
    min_freq, max_freq = 0, self.frequency / 2
    if MAX_MEL_FREQUENCY:
      max_freq = min(max_freq, MAX_MEL_FREQUENCY)
    if MIN_MEL_FREQUENCY:
      min_freq = max(min_freq, MIN_MEL_FREQUENCY)
    min_mel = hz2mel(min_freq)
    max_mel = hz2mel(max_freq)
    num_specgram_components = self.specgram_freq_bins / 2 + 1
    freqs = np.linspace(min_mel, max_mel, num_mel_bands)
    m = np.zeros((num_specgram_components, num_mel_bands))

    r = np.arange(num_mel_bands + 2) # there are (num_mel_bands + 2) filter boundaries / centers

    # evenly spaced filter boundaries in the mel domain:
    mel_filter_boundaries = r * (max_mel - min_mel) / (num_mel_bands + 1) + min_mel

    def coeff(idx, mel): # gets the unnormalised filter coefficient of filter 'idx' for a given mel value.
        lo, cen, hi = mel_filter_boundaries[idx:idx+3]
        if mel <= lo or mel >= hi:
            return 0
        # linearly interpolate
        if lo <= mel <= cen:
            return (mel - lo) / (cen - lo)
        elif cen <= mel <= hi:
            return 1 - (mel - cen) / (hi - cen)

    for k in xrange(num_specgram_components):
        # compute mel representation of the given specgram component idx
        freq = k / float(num_specgram_components) * (self.frequency / 2)
        mel = hz2mel(freq)
        for i in xrange(num_mel_bands):
            m[k, i] = coeff(i, mel)

    # normalise so that each filter has unit contribution
    return (m / m.sum(0), freqs)


  def plot_mfcc(self):
    self._mfcc_plot_helper()
    plt.show()
    plt.close()

  def save_mfcc(self, format = '.png'):
    self._mfcc_plot_helper()
    filepath, _  = os.path.splitext(self.original_path)
    savepath = os.sep.join([SAVE_DIR] + filepath.split(os.sep)[1:])
    plt.savefig('{0} - mfcc{1}'.format(savepath, format))
    plt.close()

  def _mfcc_plot_helper(self):
    fig = plt.figure()
    xmin, xmax = 0, self.num_samples/self.frequency
    ymin, ymax = 1, DEFAULT_NUM_MFCC_COEFFICIENTS - 1
    extent = xmin, xmax, ymin, ymax
    ax = fig.add_subplot(311)
    ax.set_ylabel('MFCC number')
    ax.set_title('MFCCs')
    ax.imshow(self.mfcc_cep, extent=extent, aspect='auto')
    ax1 = fig.add_subplot(312)
    ax1.set_ylabel('MFCC number')
    ax1.set_title('Deltas')
    ax1.imshow(self.mfcc_delta, extent=extent, aspect='auto')
    ax2 = fig.add_subplot(313)
    ax2.set_xlabel('Time (in seconds)')
    ax2.set_ylabel('MFCC number')
    ax2.set_title('Delta-Deltas')
    ax2.imshow(self.mfcc_delta_deltas, extent=extent, aspect='auto')

  def plot_spectrogram(self):
    fig, ax = self._spectrogram_plot_helper()
    plt.show()
    plt.close()

  def save_spectrogram(self, plot_type = 'all', format = '.png'):
    fig, ax = self._spectrogram_plot_helper()
    filepath, _  = os.path.splitext(self.original_path)
    savepath = os.sep.join([SAVE_DIR] + filepath.split(os.sep)[1:])
    if plot_type == 'labels' or plot_type == 'all':
      plt.savefig('{0} - spec - labels{1}'.format(savepath, format))
    if plot_type == 'pure' or plot_type == 'all':
      plt.axis('off')
      extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
      plt.savefig('{0} - spec - pure{1}'.format(savepath, format), bbox_inches=extent)
    plt.close()

  def _spectrogram_plot_helper(self):
    Z = self.specgram
    Z[self.specgram < 1e-10] = 1e-10
    Z = 10. * np.log10(Z)
    Z = np.flipud(Z)
    xmin, xmax = 0, self.num_samples/self.frequency
    ymin, ymax = 0, self.frequency//2 # Nyquist sampling
    extent = xmin, xmax, ymin, ymax
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.imshow(Z, extent=extent, aspect='auto')
    plt.xlabel('Time (in seconds)')
    plt.ylabel('Frequency (in Hz)')
    plt.title('Hertz Spectrogram')
    return fig, ax

  def plot_mel_spectrogram(self):
    fig, ax = self._mel_spectrogram_plot_helper()
    plt.show()
    plt.close()

  def save_mel_spectrogram(self, plot_type = 'all', format = '.png'):
    fig, ax = self._mel_spectrogram_plot_helper()
    filepath, _  = os.path.splitext(self.original_path)
    savepath = os.sep.join([SAVE_DIR] + filepath.split(os.sep)[1:])
    if plot_type == 'labels' or plot_type == 'all':
      plt.savefig('{0} - mel - labels{1}'.format(savepath, format))
    if plot_type == 'pure' or plot_type == 'all':
      plt.axis('off')
      extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
      plt.savefig('{0} - mel - pure{1}'.format(savepath, format), bbox_inches=extent)
    if plot_type == 'input' or plot_type == 'all':
      im = Image.fromarray(self.mel_specgram.T)
      im.convert('RGB').save('{0} - mel - input{1}'.format(savepath, '.bmp'))
    plt.close()

  def _mel_spectrogram_plot_helper(self):
    Z = np.flipud(self.mel_specgram)
    xmin, xmax = 0, self.num_samples/self.frequency
    ymin, ymax = 0, hz2mel(self.frequency/2)
    extent = xmin, xmax, ymin, ymax
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.imshow(Z, extent=extent, aspect='auto')
    plt.xlabel('Time (in seconds)')
    plt.ylabel('Frequency (in mel)')
    plt.title('Mel-scaled Spectrogram')
    return fig, ax

  def plot_3d_spectrogram(self):
    fig, ax = self._3d_spectrogram_helper()
    plt.show()
    plt.close()

  def save_3d_spectrogram(self, format = '.png'):
    self._3d_spectrogram_helper()
    filepath, _  = os.path.splitext(self.original_path)
    savepath = os.sep.join([SAVE_DIR] + filepath.split(os.sep)[1:])
    plt.savefig('{0} - 3D{1}'.format(savepath, format))
    plt.close()

  def _3d_spectrogram_helper(self):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    B, F = np.meshgrid(self.frame_bins, self.mel_spec_frequencies)
    ax.plot_surface(B, F, self.mel_specgram, cstride=1, rstride=10, cmap=cm.gray, linewidth=0)
    plt.title('3D spectrogram')
    ax.set_xlabel('FFT bins')
    ax.set_ylabel('Frequency')
    ax.set_zlabel('Log Amplitude')
    return fig, ax

  def save(self):
    filepath, _  = os.path.splitext(self.original_path)
    savepath = os.sep.join([SAVE_DIR] + filepath.split(os.sep)[1:])
    pickle.dump(self, open('{0}.pik'.format(savepath), 'wb'))

