import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

e_val = np.exp(1)

# start at 3 to avoid div by zero in the N/(N-1) bound
N = np.arange(3, 11)

errs = []
bounds = []

for n in N:
    k = np.arange(n)
    approx = np.sum(1 / factorial(k))
    
    errs.append(np.abs(e_val - approx))
    bounds.append((1 / factorial(n)) * (n / (n - 1)))

plt.figure(figsize=(9, 5))

plt.plot(N, errs, marker='o', label='Actual Error')
plt.plot(N, bounds, marker='s', ls='--', label='Theoretical Bound')

plt.axhline(y=0.0005, color='r', ls=':', label='Tolerance (0.0005)')
plt.axvline(x=7, color='g', ls=':', alpha=0.5)

plt.annotate('N=7', xy=(7, 0.0005), xytext=(7.3, 0.005),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

plt.yscale('log')
plt.xlabel('Number of terms (N)')
plt.ylabel('Error')
plt.title('Truncation Error for e^x at x=1')
plt.xticks(N)
plt.grid(True, ls="--", alpha=0.4)
plt.legend()

plt.tight_layout()
path='../figs/plot.png'
plt.savefig(path, dpi=300)
print(f"Saved: {path}")
plt.close()
