import numpy as np
import sys
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the parameters and initial conditions
k = 0.05  # Example wavenumber
H = 1e-5  # Hubble parameter during inflation
eta0 = -1e2  # Initial conformal time
eta_end = -1e-2  # End of inflation (conformal time)
theta0 = 1e-2  # Small parameter

# Parameters for non-standard initial conditions


def calculate_XY(vacuum_prescription):
    if vacuum_prescription == "conventional":
        # Conventional vacuum prescription
        X = 0
        # 'p' should be defined or imported appropriately
        p = 0.1  # Example value for 'p'
        Y = 1j * (1 - 2 * p) * (1 - p)
    elif vacuum_prescription == "adiabatic":
        # Adiabatic vacuum prescription (≥ 1st order)
        X = 0
        Y = 0
    elif vacuum_prescription == "hamiltonian_diagonalization":
        # Hamiltonian diagonalization vacuum prescription
        X = 0
        Y = 0
    elif vacuum_prescription == "danielsson":
        # Danielsson vacuum prescription
        X = -1j
        Y = 1j
    else:
        print("Invalid vacuum prescription")
        sys.exit(1)
    return X, Y

# Check if command-line arguments for vacuum prescription are provided
if len(sys.argv) != 2:
    print("Usage: python inflationary_perturbations.py <vacuum_prescription>")
    sys.exit(1)

# Parse command-line argument for vacuum prescription
vacuum_prescription = sys.argv[1]

# Calculate X and Y based on the chosen vacuum prescription
X, Y = calculate_XY(vacuum_prescription)

# Define the initial conditions for v_k and v_k'
v_k0 = (1 / np.sqrt(k)) * (1 + X + Y) * theta0 / 2
v_k0_prime = -1j * np.sqrt(k) * (1 + Y - X) * theta0 / 2

# Mukhanov-Sasaki equation
def ms_equation(eta, y):
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
