"""
Calculate Mean Square Displacement (MSD) and persistence length from FAABP simulation data.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

def load_simulation_data(data_file='./data/payload_numba_simulation_data_17.npz'):
    """Load simulation data from .npz file."""
    data = np.load(data_file)
    return {
        'positions': data['positions'],
        'orientations': data['orientations'],
        'velocities': data['velocities'],
        'payload_positions': data['payload_positions'],
        'payload_velocities': data['payload_velocities'],
        'params': data['params']
    }

def unravel_trajectory(traj, boundary_length):
    """Unravel a trajectory that was created with periodic boundary conditions."""
    # Track all jump locations
    jump_pos_lower = []
    jump_pos_upper = []
    
    # Go through trajectory in pairs
    for i, (c1, c2) in enumerate(zip(traj[:-1], traj[1:])):
        if np.abs(c1 - c2) > boundary_length / 2:
            # Determine where it jumped
            if c1 > c2:
                jump_pos_upper.append(i)
            else:
                jump_pos_lower.append(i)
    
    # Update values
    for i, c in enumerate(traj):
        # Jump at upper limit -> add boundary length
        if i in jump_pos_upper:
            traj[i+1:] += boundary_length
        # Jump at lower limit -> subtract boundary length
        elif i in jump_pos_lower:
            traj[i+1:] -= boundary_length
    
    return traj

def calculate_msd(trajectory, max_lagtime):
    """Calculate Mean Square Displacement from a trajectory."""
    T = len(trajectory)
    msd = np.zeros(min(max_lagtime, T-1))
    
    for dt in range(1, min(max_lagtime+1, T)):
        # Calculate displacements for this lag time
        displacements = trajectory[dt:] - trajectory[:-dt]
        # Calculate MSD
        msd[dt-1] = np.mean(np.sum(displacements**2, axis=1))
    
    return np.arange(1, len(msd)+1), msd

def fit_msd_abp(t, msd):
    """Fit MSD data to Active Brownian Particle model."""
    def msd_abp(dt, rot_diff, v):
        return (v**2 / (2 * rot_diff**2)) * (2 * rot_diff * dt + np.exp(-2 * rot_diff * dt) - 1)
    
    # Fit the curve
    popt, _ = curve_fit(msd_abp, t, msd, bounds=(0, [1, 5]))
    rot_diff, v = popt
    
    # Calculate persistence length
    tau_p = 1 / rot_diff
    l_p = v * tau_p
    
    return v, rot_diff, l_p, msd_abp(t, *popt)

def plot_msd(t, msd, fitted_msd, v, rot_diff, l_p):
    """Plot MSD data and fit."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data and fit
    ax.plot(t, msd, 'r-', alpha=0.7, label='Data')
    ax.plot(t, fitted_msd, 'k--', alpha=0.7, label='Fit')
    
    # Add text with parameters
    text = f'v = {v:.2f}\nD_r = {rot_diff:.2f}\nl_p = {l_p:.2f}'
    ax.text(0.02, 0.98, text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set labels and scale
    ax.set_xlabel('Lag time $t$')
    ax.set_ylabel(r'$\langle \Delta r^2 \rangle$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig

def main():
    # Load simulation data
    data = load_simulation_data()
    
    # Extract parameters
    box_size = data['params'][1]  # Assuming box_size is the second parameter
    max_lagtime = 1000  # Adjust as needed
    
    # Calculate MSD for payload
    payload_traj = data['payload_positions']
    
    # Unravel trajectories
    x_traj = unravel_trajectory(payload_traj[:, 0].copy(), box_size)
    y_traj = unravel_trajectory(payload_traj[:, 1].copy(), box_size)
    
    # Calculate MSD
    t, msd = calculate_msd(payload_traj, max_lagtime)
    
    # Fit MSD to ABP model
    v, rot_diff, l_p, fitted_msd = fit_msd_abp(t, msd)
    
    # Plot results
    fig = plot_msd(t, msd, fitted_msd, v, rot_diff, l_p)
    plt.savefig('visualizations/msd_analysis_17.png')
    plt.close()
    
    print(f"Analysis Results:")
    print(f"Self-propulsion speed (v): {v:.2f}")
    print(f"Rotational diffusion (D_r): {rot_diff:.2f}")
    print(f"Persistence length (l_p): {l_p:.2f}")

if __name__ == "__main__":
    main() 