import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift, ifftshift
import os

output_dir = '../figs/'

dt = 0.01
t = np.arange(-50, 50, dt)

# standard Gaussian pulse
x = np.sinc(t)

t_a = 2.0 
y = np.sinc(t-t_a) # y(t) = x(t - t_a)

n = len(t)
X = fftshift(fft(ifftshift(x)))
Y = fftshift(fft(ifftshift(y)))

freqs = fftshift(fftfreq(n, d=dt))

mag_X = np.abs(X)
mag_Y = np.abs(Y)

phase_X = np.angle(X)
phase_Y = np.angle(Y)

valid_idx = mag_X > 0.05*np.max(mag_X)

f_valid = freqs[valid_idx]
#ignore weak freqs instead of sharp jump to 0
phase_X_valid = np.unwrap(np.angle(X[valid_idx]))
phase_Y_valid = np.unwrap(np.angle(Y[valid_idx]))

center_idx = np.argmin(np.abs(f_valid)) # Set 0Hz to 0
phase_X_valid -= phase_X_valid[center_idx]
phase_Y_valid -= phase_Y_valid[center_idx]

plt.figure(figsize=(8, 5))
plt.plot(t, x, label='x(t) : Original Signal', color='blue')
plt.plot(t, y, label=f'y(t) : Delayed Signal (t_a={t_a})', color='red', linestyle='--')
plt.title('Time Domain')
plt.xlabel('Time (t)')
plt.ylabel('Amplitude')
plt.xlim(-5, 7)
plt.legend()
plt.grid(True)
plt.tight_layout()

time_domain_path = os.path.join(output_dir, 'time_domain_plot1.png')
plt.savefig(time_domain_path, dpi=300)
print(f"Saved: {time_domain_path}")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(freqs, mag_X, label='|X(ω)| : Magnitude of x(t)', color='blue', linewidth=3)
plt.plot(freqs, mag_Y, label='|Y(ω)| : Magnitude of y(t)', color='orange', linestyle='--', linewidth=2)
plt.title('Frequency Domain: Magnitude')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(-2, 2)
plt.legend()
plt.grid(True)
plt.tight_layout()

magnitude_path = os.path.join(output_dir, 'magnitude_spectrum_plot1.png')
plt.savefig(magnitude_path, dpi=300)
print(f"Saved: {magnitude_path}")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(f_valid, phase_X_valid, label=r'$\angle X(\omega)$ : Phase of x(t)', color='blue') # waaaow latex crajee
plt.plot(f_valid, phase_Y_valid, label=r'$\angle Y(\omega)$ : Phase of y(t)', color='red', linestyle='--')
plt.title('Frequency Domain: Phase')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (Radians)')
plt.xlim(-2, 2)
plt.legend()
plt.grid(True)
plt.tight_layout()

phase_path = os.path.join(output_dir, 'phase_spectrum_plot1.png')
plt.savefig(phase_path, dpi=300)
print(f"Saved: {phase_path}")
plt.close()
