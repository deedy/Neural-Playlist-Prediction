from pylab import *
from mpl_toolkits.mplot3d import Axes3D

dt = 0.001
t = arange(0.0, 10.0*pi, dt)
s1 = sin(2*pi*200*t)
s2 = 2*sin(2*pi*400*t)
s3 = sin(2*pi*100*t)

# # create a transient "chirp"
mask = where(logical_and(t>5.0*pi, t<6.0*pi), 1.0, 0.0)
s2 = s2 * mask

mask = where(logical_and(t>3.0*pi, t<4.0*pi), 1.0, 0.0)
s3 = s3 * mask

# add some noise into the mix
nse = 0.01*randn(len(t))

x = s1 + s2 +s3 #+ nse # the signal
NFFT = 1024       # the length of the windowing segments
Fs = int(1.0/dt)  # the sampling frequency

# Pxx is the segments x freqs array of instantaneous power, freqs is
# the frequency vector, bins are the centers of the time bins in which
# the power is computed, and im is the matplotlib.image.AxesImage
# instance

ax1 = subplot(211)
plot(t, x)
subplot(212, sharex=ax1)
Pxx, freqs, bins, im = specgram(x, NFFT=NFFT, Fs=Fs, noverlap=0,
                                cmap=cm.gist_heat)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
B, F = meshgrid(bins, freqs)
ax.plot_surface(B, F, Pxx, cstride=1, rstride=10, cmap=cm.coolwarm)
show()


