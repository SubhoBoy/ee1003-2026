import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import sympy as sp
import os

# Create the target directory for figures
os.makedirs('../figs', exist_ok=True)

# ==========================================
# PART 1: Symbolic Verification (SymPy)
# ==========================================
print("--- Symbolic Verification ---")

# Define symbols and matrices
s, t = sp.symbols('s t')
A_sym = sp.Matrix([[0, 1], [0, -2]])
B_sym = sp.Matrix([[0], [1]])
C_sym = sp.Matrix([[1, 0]])
I_sym = sp.eye(2)

# 1. Verify Transfer Function G(s) = C(sI - A)^-1 B
G_s = C_sym * (s * I_sym - A_sym).inv() * B_sym
G_s_simplified = sp.simplify(G_s[0])
print(f"1. Open-Loop Transfer Function G(s): {G_s_simplified}")

# 2. Verify Closed-Loop Matrix A_cl = A - BC
A_cl_sym = A_sym - B_sym * C_sym
print(f"2. Closed-Loop Matrix A_cl:\n{np.array(A_cl_sym.tolist())}")

# 3. Verify Exact Discretization Matrix (Matrix Exponential Phi_d)
Phi_d_sym = sp.simplify(sp.exp(A_cl_sym * t))
print(f"3. Matrix Exponential Phi_d(t) = e^(A_cl*t):\n{Phi_d_sym}")
print("-" * 40)


# ==========================================
# PART 2: Numerical System Definition
# ==========================================
print("--- Running Numerical Simulations ---")
# Explicitly float type to avoid internal truncation
A = np.array([[0, 1], [0, -2]], dtype=float)
B = np.array([[0], [1]], dtype=float)
C = np.array([[1, 0]], dtype=float)
D = np.array([[0]], dtype=float)
A_cl = A - B @ C

Ts = 0.5  
t_end = 8.0
t_samples = np.arange(0, t_end + Ts, Ts)


# ==========================================
# PART 3: Pure Analytical Continuous Solution
# ==========================================
# Using the exact mathematical derivation y(t) = 1 - (1+t)*e^(-t)
def exact_continuous(t):
    return 1.0 - (1.0 + t) * np.exp(-t)

# Smooth continuous curves for plotting
t_cont = np.linspace(0, t_end, 500)
y_cont = exact_continuous(t_cont)

t_dense = np.linspace(1.48, 1.52, 1000)
y_dense = exact_continuous(t_dense)

# True values exactly at sample times (for error calculation)
y_true_samples = exact_continuous(t_samples)


# ==========================================
# PART 4: Exact Discrete Method (Matrix Exponential)
# ==========================================
sys_cl_disc = signal.cont2discrete((A_cl, B, C, D), Ts, method='zoh')
_, y_disc = signal.dstep(sys_cl_disc, t=t_samples)
y_disc = np.squeeze(y_disc)


# ==========================================
# PART 5: RK4 Integration Method
# ==========================================
def rk4_step_response(A, B, C, Ts, t_end):
    t = np.arange(0, t_end + Ts, Ts)
    x = np.zeros((2, len(t)))
    for i in range(len(t) - 1):
        xi = x[:, i:i+1]
        k1 = A @ xi + B * 1.0
        k2 = A @ (xi + (Ts / 2) * k1) + B * 1.0
        k3 = A @ (xi + (Ts / 2) * k2) + B * 1.0
        k4 = A @ (xi + Ts * k3) + B * 1.0
        x[:, i+1:i+2] = xi + (Ts / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return t, (C @ x).flatten()

_, y_rk4 = rk4_step_response(A_cl, B, C, Ts, t_end)


# ==========================================
# PART 6: Error Calculations
# ==========================================
# Add an infinitesimally small number (1e-20) to avoid log(0) errors
error_disc = np.abs(y_true_samples - y_disc) + 1e-20
error_rk4 = np.abs(y_true_samples - y_rk4) + 1e-20


# ==========================================
# PART 7: Plotting and Saving Figures
# ==========================================

# --- FIGURE 1: Exact Discrete ---
plt.figure(figsize=(10, 6))
plt.plot(t_cont, y_cont, 'b-', linewidth=2, label='Continuous Analytical Curve')
plt.step(t_samples, y_disc, 'ro--', where='post', label=f'Exact Discrete ZOH ($T_s$={Ts}s)')
plt.axhline(1.0, color='k', linestyle=':', linewidth=2, label='Reference $r(t) = 1$')
plt.title('Figure 1: Continuous Analytical vs. Exact Discrete Method', fontsize=14)
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Output $y(t)$', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower right', fontsize=11)
plt.tight_layout()
plt.savefig('../figs/fig1_exact_discrete.png', dpi=300)
plt.close()

# --- FIGURE 2: RK4 Approximation ---
plt.figure(figsize=(10, 6))
plt.plot(t_cont, y_cont, 'b-', linewidth=2, label='Continuous Analytical Curve')
plt.plot(t_samples, y_rk4, 'gs-', markersize=6, alpha=0.8, label=f'RK4 Approximation ($h$={Ts}s)')
plt.axhline(1.0, color='k', linestyle=':', linewidth=2, label='Reference $r(t) = 1$')
plt.title('Figure 2: Continuous Analytical vs. RK4 Integration', fontsize=14)
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Output $y(t)$', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower right', fontsize=11)
plt.tight_layout()
plt.savefig('../figs/fig2_rk4_comparison.png', dpi=300)
plt.close()

# --- FIGURE 3: Microscopic Zoom at t = 1.5s ---
plt.figure(figsize=(10, 6))
plt.plot(t_dense, y_dense, 'b-', linewidth=2, label='Continuous Analytical Curve')
idx = 3 # Index corresponding to t=1.5s
plt.plot(t_samples[idx], y_disc[idx], 'ro', markersize=10, label='Exact Discrete Point')
plt.plot(t_samples[idx], y_rk4[idx], 'gs', markersize=10, label='RK4 Point')
plt.title('Figure 3: Microscopic Zoom at $t = 1.5$s', fontsize=14)
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Output $y(t)$', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower right', fontsize=11)
plt.tight_layout()
plt.savefig('../figs/fig3_zoomed.png', dpi=300)
plt.close()

# --- FIGURE 4: Error Magnitude (Log Scale) ---
plt.figure(figsize=(10, 6))
plt.semilogy(t_samples, error_disc, 'ro-', linewidth=2, label='Exact Discrete Error')
plt.semilogy(t_samples, error_rk4, 'gs-', linewidth=2, label='RK4 Truncation Error')
plt.axhline(1e-15, color='k', linestyle=':', linewidth=2, label='Machine Precision (~1e-15)')
plt.title('Figure 4: Absolute Error vs. True Analytical Solution', fontsize=14)
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Absolute Error $|y_{true} - y_{sim}|$ (Log Scale)', fontsize=12)
plt.grid(True, which="both", linestyle='--', alpha=0.5)
plt.legend(loc='center right', fontsize=11)
plt.tight_layout()
plt.savefig('../figs/fig4_error_log.png', dpi=300)
plt.close()

#print("All 4 figures generated and saved successfully!")
