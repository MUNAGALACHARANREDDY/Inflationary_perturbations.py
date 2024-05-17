import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the parameters and initial conditions
k = 0.05  # Example wavenumber
H = 1e-5  # Hubble parameter during inflation
eta0 = -1e2  # Initial conformal time
eta_end = -1e-2  # End of inflation (conformal time)
theta0 = 1e-2  # Small parameter

# Parameters for non-standard initial conditions
X = 0.1  # Example value for X
Y = 0.2  # Example value for Y

# Define the initial conditions for v_k and v_k'
v_k0 = (1 / np.sqrt(k)) * (1 + X + Y) * theta0 / 2
v_k0_prime = -1j * np.sqrt(k) * (1 + Y - X) * theta0 / 2

# Mukhanov-Sasaki equation
def ms_equation(eta, y, k, H):
    v_k, v_k_prime = y
    z_double_prime_over_z = 2 / eta**2  # Example for de Sitter space
    dvk_deta = v_k_prime
    dvk_prime_deta = - (k**2 - z_double_prime_over_z) * v_k
    return [dvk_deta, dvk_prime_deta]

# Solve the differential equation
solution = solve_ivp(
    ms_equation,
    [eta0, eta_end],
    [v_k0, v_k0_prime],
    args=(k, H),
    t_eval=np.linspace(eta0, eta_end, 1000),
    method='RK45'
)

# Extract the solution
eta_vals = solution.t
v_k_vals = solution.y[0]

# Calculate the power spectrum at the end of inflation
z_end = (H * eta_end)  # Simplified example for z at the end of inflation
P_R = (k**3 / (2 * np.pi**2)) * (np.abs(v_k_vals[-1]) / z_end)**2

# Plot the results
plt.plot(eta_vals, np.real(v_k_vals), label='Real part of $v_k$')
plt.plot(eta_vals, np.imag(v_k_vals), label='Imaginary part of $v_k$')
plt.xlabel('Conformal Time (η)')
plt.ylabel('$v_k$')
plt.legend()
plt.title('Evolution of $v_k$ with Non-Standard Initial Conditions')
plt.grid(True)
plt.show()

print(f'Power Spectrum at k={k}: {P_R:.3e}')
