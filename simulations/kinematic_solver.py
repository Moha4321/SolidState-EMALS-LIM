import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import cumulative_trapezoid

# --- SYSTEM PARAMETERS ---
# 7075-T6 Aluminum Properties
sigma = 1.8e7     # Electrical conductivity (Siemens/m)
rho = 2810.0      # Density (kg/m^3)

# PCB Stator Properties (Estimates for design)
f = 100.0         # AC Frequency of the inverter (Hz)
tau_p = 0.015     # Pole pitch / coil width on PCB (15 mm)
B_0 = 0.01        # Peak magnetic field from PCB traces (10 milliTesla)

# Calculated Synchronous Speed of the magnetic wave
v_s = 2 * f * tau_p

print(f"Synchronous Magnetic Wave Speed: {v_s:.2f} m/s")

# --- THE DIFFERENTIAL EQUATION ---
# F = ma -> m(dv/dt) = 0.5 * sigma * Volume * s * v_s * B_0^2
# Since m = rho * Volume, Volume cancels out!
# dv/dt = (sigma * B_0^2 / 2*rho) * (v_s - v)

def acceleration(v, t, sigma, rho, B_0, v_s):
    # This is our first-order ODE
    alpha = (sigma * B_0**2) / (2 * rho)
    dvdt = alpha * (v_s - v)
    return dvdt

# --- SIMULATION ---
# Time array: simulate for 1.5 seconds
t = np.linspace(0, 1.5, 1000)
v0 = 0.0  # Initial velocity

# Solve the ODE
v_solution = odeint(acceleration, v0, t, args=(sigma, rho, B_0, v_s))

# Calculate position by integrating velocity
x_solution = cumulative_trapezoid(v_solution.flatten(), t, initial=0)

# --- PLOTTING ---
plt.style.use('dark_background')
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:cyan'
ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('Velocity (m/s)', color=color, fontsize=12)
ax1.plot(t, v_solution, color=color, linewidth=2, label='Projectile Velocity')
ax1.axhline(v_s, color='white', linestyle='--', alpha=0.5, label='Magnetic Wave Speed ($v_s$)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='lower right')

ax2 = ax1.twinx()  
color = 'tab:orange'
ax2.set_ylabel('Position along PCB (m)', color=color, fontsize=12)  
ax2.plot(t, x_solution, color=color, linewidth=2, linestyle=':', label='Projectile Position')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Solid-State LIM: Kinematic Trajectory via ODE Integration', fontsize=14)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()