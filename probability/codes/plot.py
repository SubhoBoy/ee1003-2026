import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy import optimize
import os

# ---------------------------------------------------------
# Setup & Initializations
# ---------------------------------------------------------
output_dir = '../figs/'
os.makedirs(output_dir, exist_ok=True)

R_val = 25.0
true_root_pos = np.sqrt(R_val)
true_root_neg = -np.sqrt(R_val)

# Define initial guesses to explore behaviors
initial_guesses = {
    'Standard (x0=12)': 12.0,
    'Close to zero (x0=0.5)': 0.5,
    'Negative (x0=-12)': -12.0
}

n_iters = 8

# Analytical functions
def g(x): return 0.5 * (x + R_val / x)               # Iteration sequence
def dg(x): return 0.5 * (1 - R_val / x**2)           # Derivative for stability
def f(x): return x**2 - R_val                        # Inferred function (Method 2)
def df(x): return 2 * x                              # Derivative of inferred function

# ---------------------------------------------------------
# 1. Symbolic Verification
# ---------------------------------------------------------
print("--- SymPy Verification (Reverse-Engineering) ---")
x_sym, R_sym = sp.symbols('x R', real=True)
f_sym = x_sym**2 - R_sym
f_prime_sym = sp.diff(f_sym, x_sym)
nr_formula = x_sym - (f_sym / f_prime_sym)
simplified_nr = sp.simplify(nr_formula)
print(f"Standard N-R Formula: x - (x^2 - R)/(2x)")
print(f"Simplified: {simplified_nr}\n")

# ---------------------------------------------------------
# 2. Numerical Data Generation
# ---------------------------------------------------------
sequences = {}
errors = {}

for name, x0 in initial_guesses.items():
    seq = [x0]
    for _ in range(n_iters):
        seq.append(g(seq[-1]))
    sequences[name] = np.array(seq)
    
    # Error relative to the root it converges to (+5 or -5)
    target_root = true_root_pos if x0 > 0 else true_root_neg
    errors[name] = np.abs(sequences[name] - target_root)

# ---------------------------------------------------------
# Plot 1: Sequence Convergence (Multi-Path)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
colors = ['b', 'orange', 'r']
for (name, seq), color in zip(sequences.items(), colors):
    plt.plot(range(n_iters + 1), seq, marker='o', linestyle='-', color=color, label=f'{name}')

plt.axhline(y=true_root_pos, color='g', linestyle='--', label=f'+ Root ($\sqrt{{R}}$ = {true_root_pos})')
plt.axhline(y=true_root_neg, color='purple', linestyle='--', label=f'- Root ($-\sqrt{{R}}$ = {true_root_neg})')
plt.title('Convergence of Sequence for Different Initial Guesses')
plt.xlabel('Iteration ($n$)')
plt.ylabel('Value of $x_n$')
plt.xticks(range(n_iters + 1))
plt.grid(True, alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'plot1_convergence_multi.png'), dpi=300)
plt.close()

# ---------------------------------------------------------
# Plot 2: Error Analysis (Log Scale)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
for (name, err), color in zip(errors.items(), colors):
    plt.semilogy(range(n_iters + 1), err, marker='s', linestyle='-', color=color, label=f'{name}')

plt.title('Absolute Error vs Iteration (Log Scale)')
plt.xlabel('Iteration ($n$)')
plt.ylabel('Absolute Error $|x_n - Root|$')
plt.xticks(range(n_iters + 1))
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'plot2_error_multi.png'), dpi=300)
plt.close()

# ---------------------------------------------------------
# Plot 3: Geometric Interpretation of Newton-Raphson
# ---------------------------------------------------------
x_vals_geom = np.linspace(2, 13, 500)
y_vals_geom = f(x_vals_geom)

plt.figure(figsize=(10, 6))
plt.plot(x_vals_geom, y_vals_geom, label='$f(x) = x^2 - R$', color='black', linewidth=2)
plt.axhline(0, color='black', linewidth=1)

seq_std = sequences['Standard (x0=12)']
for i in range(3):
    xn = seq_std[i]
    yn = f(xn)
    xn_next = seq_std[i+1]
    
    plt.plot(xn, yn, marker='o', color='red')
    plt.vlines(xn, 0, yn, linestyles='dotted', colors='red', alpha=0.7)
    
    tangent_x = np.array([xn_next, xn])
    tangent_y = df(xn) * (tangent_x - xn) + yn
    plt.plot(tangent_x, tangent_y, color='red', linestyle='--', 
             label=f'Step {i+1}: $x_{i}={xn:.2f} \\rightarrow x_{i+1}={xn_next:.2f}$' if i==0 else "")

plt.plot(true_root_pos, 0, marker='*', color='gold', markersize=15, markeredgecolor='black', label='Root $\sqrt{R}$')
plt.title('Geometric Execution of Newton-Raphson (Standard Guess)')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'plot3_geometric_nr.png'), dpi=300)
plt.close()

# ---------------------------------------------------------
# Plot 4, 5, 6: Cobweb Plots for Different Initial Guesses
# ---------------------------------------------------------
def plot_cobweb(x0, name, filename, x_range, y_range):
    x_vals = np.linspace(x_range[0], x_range[1], 1000)
    with np.errstate(divide='ignore'):  # Ignore zero-division warnings locally
        y_vals = g(x_vals)
    
    plt.figure(figsize=(8, 8))
    plt.plot(x_vals, y_vals, label='$y = g(x)$', color='blue', linewidth=2)
    plt.plot(x_vals, x_vals, label='$y = x$', color='gray', linestyle='--')
    
    cx, cy = [x0], [0]
    xc = x0
    for _ in range(5):
        yc = g(xc)
        cx.extend([xc, yc])
        cy.extend([yc, yc])
        xc = yc
        
    plt.plot(cx, cy, color='red', marker='.', linestyle='-', linewidth=1.5, label='Path')
    
    target_root = true_root_pos if x0 > 0 else true_root_neg
    plt.plot(target_root, target_root, marker='*', color='gold', markersize=15, markeredgecolor='black', label=f'Root = {target_root}')
    
    plt.title(f'Cobweb Plot: {name}')
    plt.xlabel('$x_n$')
    plt.ylabel('$x_{n+1}$')
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

# Generate cobwebs
plot_cobweb(12.0, 'Standard Guess (x0=12)', 'plot4_cobweb_std.png', [2, 14], [2, 14])
plot_cobweb(0.5, 'Close to zero (x0=0.5)', 'plot5_cobweb_small.png', [0.1, 26], [0, 26])
plot_cobweb(-12.0, 'Negative Guess (x0=-12)', 'plot6_cobweb_neg.png', [-14, -2], [-14, -2])

# ---------------------------------------------------------
# Plot 7: Full Convergence Stability (Derivative Test)
# ---------------------------------------------------------
x_vals_stab = np.linspace(-15, 15, 1000)
x_vals_stab = x_vals_stab[np.abs(x_vals_stab) > 1.5] # Exclude small region around 0 
dy_vals = dg(x_vals_stab)

plt.figure(figsize=(10, 6))
plt.plot(x_vals_stab, dy_vals, label="$g'(x)$", color='purple', linewidth=2)
plt.axhline(1, color='red', linestyle=':', alpha=0.5, label='Stability Bound $\pm 1$')
plt.axhline(-1, color='red', linestyle=':', alpha=0.5)
plt.axhline(0, color='black', linewidth=1)

plt.plot(true_root_pos, dg(true_root_pos), marker='o', color='green', markersize=8, label=f"Positive Root (+5)")
plt.plot(true_root_neg, dg(true_root_neg), marker='o', color='blue', markersize=8, label=f"Negative Root (-5)")

plt.fill_between(x_vals_stab, -1, 1, color='green', alpha=0.1, label='Stable Region')
plt.title('Stability Analysis of $g(x)$ over Full Domain')
plt.xlabel('$x$')
plt.ylabel("$g'(x)$")
plt.ylim(-2, 1.5)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'plot7_stability.png'), dpi=300)
plt.close()

print("All unified plots generated and saved to '../figs/' successfully.")
