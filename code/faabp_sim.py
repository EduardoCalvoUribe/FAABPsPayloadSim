"""
Simple Payload Transport Simulation with FAABPs

This script runs a simulation of a passive payload being transported by
Force-Aligning Active Brownian Particles (FAABPs) and creates a GIF animation.
The simulation uses Numba for improved performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.patches import Circle
import time
import os
import math
from numba import njit, float64, int64

# Create directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

##############################
# Physics utility functions  #
##############################

@njit
def normalize(v):
    """Normalize a vector to unit length."""
    norm = np.sqrt(np.sum(v**2))
    if norm > 0:
        return v / norm
    return v

@njit
def compute_minimum_distance(pos_i, pos_j, box_size):
    """Compute minimum distance vector considering periodic boundaries."""
    # Direct vector
    r_ij = pos_j - pos_i
    
    # Apply periodic boundary conditions to find minimum distance
    r_ij = r_ij - box_size * np.round(r_ij / box_size)
    
    return r_ij

@njit
def compute_repulsive_force(pos_i, pos_j, radius_i, radius_j, stiffness, box_size):
    """Compute repulsive force between two particles.
    
    Implements the equation:
    f_ij = { S_0 * (a+b-r_ij) * r_hat_ij, if r_ij <= a+b
           { 0,                           otherwise
    
    where:
    - S_0 is the stiffness
    - a, b are the particle radii
    - r_ij is the distance between particles
    - r_hat_ij is the unit vector from particle i to j
    """
    # Vector from particle i to j
    # r_ij = pos_j - pos_i
    r_ij = compute_minimum_distance(pos_i, pos_j, box_size)
    
    # Distance between particles
    dist = np.sqrt(np.sum(r_ij**2))
    
    # Avoid division by zero
    if dist < 1e-10:
        # Small displacement to avoid particles at exactly the same position
        r_ij = np.array([1e-5, 1e-5])  # Small fixed vector
        dist = np.sqrt(np.sum(r_ij**2))
    
    # Unit vector from particle i to j
    r_hat = r_ij / dist
    
    # Sum of radii (a+b in the equation)
    sum_radii = radius_i + radius_j
    
    # Compute repulsive force (only if particles overlap)
    if dist < sum_radii:
        # Force magnitude: S_0 * (a+b-r_ij)
        force_magnitude = stiffness * (sum_radii - dist)
        
        # Force direction: -r_hat (negative because it's repulsive)
        return -force_magnitude * r_hat
    
    # No force if particles don't overlap
    return np.zeros(2)

@njit
def create_cell_list(positions, box_size, cell_size, n_particles):
    """Create a cell list for efficient neighbor searching. 
    Uses a linked list implementation instead of a 2D array because it's faster with Numba!"""
    # Calculate number of cells
    n_cells = int(np.floor(box_size / cell_size)) # cell_size is at least 2*max_radius (particle-particle max interaction)
    
    # Initialize cell lists with -1 (empty indicator)
    head = np.ones(n_cells * n_cells, dtype=int64) * -1  # First particle in each cell
    list_next = np.ones(n_particles, dtype=int64) * -1   # Next particle in same cell
    # fails to work without int64 for some reason
    
    # Assign particles to cells
    for i in range(n_particles):
        # Get cell indices
        cell_x = min(int(positions[i, 0] / cell_size), n_cells - 1)
        cell_y = min(int(positions[i, 1] / cell_size), n_cells - 1)
        cell_id = cell_y * n_cells + cell_x 
        # Row-major ordering, converts 2D grid coords into a 1D index.
        # Every x, y pair maps to a different number, and it's reversible
        # x = cell_id % n_cells
        # y = cell_id // n_cells
        
        # Add particle to linked list
        list_next[i] = head[cell_id] 
        head[cell_id] = i
        
        # Example:
        
        # Initial state: Empty cell (cell_id = 5)
        # head[5] = -1
        # list_next = [-1, -1, -1, ...]

        # Add particle 10 to cell 5:
        # head[5] = 10
        # list_next[10] = -1

        # Add particle 7 to cell 5:
        # list_next[7] = 10  (7 points to 10)
        # head[5] = 7        (7 is new head)

        # Add particle 3 to cell 5:
        # list_next[3] = 7   (3 points to 7)
        # head[5] = 3        (3 is new head)
        
        # Later, when we want to find all particles in cell 5, we can start at head[5] and follow the links:
        # j = head[5]
        # while j != -1:
        #     # Do something with j
        #     j = list_next[j]
    
    return head, list_next, n_cells

##########################
# Main physics functions #
##########################

# Will remove this function later. Old version without cell list optimization
@njit
def compute_all_forces(positions, payload_pos, radii, payload_radius, stiffness, n_particles, box_size):
    """OLD: Compute all forces acting on particles and the payload."""
    # Initialize forces
    particle_forces = np.zeros((n_particles, 2))
    payload_force = np.zeros(2)
    
    # Compute forces between particles and payload
    for i in range(n_particles):
        # Force between particle and payload
        force_particle_payload = compute_repulsive_force(
            positions[i], payload_pos, radii[i], payload_radius, stiffness, box_size
        )
        
        # Apply force to particle
        particle_forces[i] += force_particle_payload
        
        # Apply opposite force to payload
        payload_force -= force_particle_payload
        
        # Repulsive forces from other particles
        for j in range(n_particles):
            if i != j:
                particle_forces[i] += compute_repulsive_force(
                    positions[i], positions[j], radii[i], radii[j], stiffness, box_size
                )
    
    return particle_forces, payload_force

@njit
def compute_all_forces_cell_list(positions, payload_pos, radii, payload_radius, stiffness, n_particles, box_size):
    """Compute all forces acting on particles and the payload using cell list optimization."""
    # Initialize forces
    particle_forces = np.zeros((n_particles, 2)) # Initialize force array for particles
    payload_force = np.zeros(2) # Initialize force array for payload
    
    # Determine maximum interaction distance (for cell size)
    max_radius = np.max(radii) # Takes maximum radius of all particles. (Because radius of particles is possibly heterogeneous)
    cell_size = 2 * max_radius  # For particle-particle interactions (not payload-particle)
    
    # Create cell list
    head, list_next, n_cells = create_cell_list(positions, box_size, cell_size, n_particles)
    
    # Compute forces between particles and payload (O(N))
    for i in range(n_particles):
        force_particle_payload = compute_repulsive_force( # Computes force between particle and payload
            positions[i], payload_pos, radii[i], payload_radius, stiffness, box_size
        )
        particle_forces[i] += force_particle_payload # Applies force to particle
        payload_force -= force_particle_payload  # Applies opposite force to payload
    
    # Compute forces between particles using cell list (now O(N))
    # For each particle
    for i in range(n_particles):
        # Find which cell it belongs to
        cell_x = min(int(positions[i, 0] / cell_size), n_cells - 1)
        cell_y = min(int(positions[i, 1] / cell_size), n_cells - 1)
        
        # Check neighboring cells (including own cell)
        for dx in range(-1, 2):  # -1, 0, 1
            for dy in range(-1, 2):  # -1, 0, 1
                # Get neighboring cell (periodic boundaries)
                neigh_x = (cell_x + dx) % n_cells
                neigh_y = (cell_y + dy) % n_cells
                neigh_cell_id = neigh_y * n_cells + neigh_x # create_cell_list() uses this 1D format because it's faster with Numba
                
                # Get the first particle in the neighboring cell
                j = head[neigh_cell_id]
                
                # Looping through all particles in this cell
                while j != -1:
                    if i != j:  # Skips self-interaction
                        particle_forces[i] += compute_repulsive_force(
                            positions[i], positions[j], radii[i], radii[j], stiffness, box_size
                        )
                    # Move to next particle in this cell
                    j = list_next[j] # Check create_cell_list() for more details
    
    return particle_forces, payload_force


@njit
def update_orientation_vectors(orientations, forces, curvity, dt, rot_diffusion, n_particles):
    """Update all particle orientations based on forces and rotational diffusion.
    The torque is calculated as:
    torque = k * (n × F)
    (this is equivalent to k(e x (v x e)) from paper)
    
    Orientation update is:
    dn/dt = torque * (n × z) + noise
    
    Where:
    - n is the orientation vector
    - F is the net force
    - k is curvity
    - z is the unit vector pointing out of the 2D plane (implicitly used in the cross product calculation)
    """
    
    new_orientations = np.zeros_like(orientations)
    
    for i in range(n_particles):
        # Calculate torque: τ = curvity * (n × F)
        # n × F = n_x*F_y - n_y*F_x
        cross_product = orientations[i, 0] * forces[i, 1] - orientations[i, 1] * forces[i, 0]
        torque = curvity[i] * cross_product
        
        # Calculate orientation change: dn/dt = torque * (n × z)
        # n × z = (-n_y, n_x)
        n_cross_z = np.array([-orientations[i, 1], orientations[i, 0]])
        orientation_change = torque * n_cross_z * dt
        
        # Add rotational diffusion as a random perpendicular vector
        if rot_diffusion[i] > 0:
            # Generate noise using normal distribution
            noise_magnitude = np.sqrt(2 * rot_diffusion[i] * dt)
            noise_x = np.random.normal(0, noise_magnitude)
            noise_y = np.random.normal(0, noise_magnitude)
            noise_vector = np.array([noise_x, noise_y])
            
            # Project noise to be perpendicular to orientation using cross product
            # (n × (noise × n)) = noise - (noise·n)n
            noise_dot_n = noise_vector[0] * orientations[i, 0] + noise_vector[1] * orientations[i, 1]
            noise_perp = np.array([
                noise_vector[0] - noise_dot_n * orientations[i, 0],
                noise_vector[1] - noise_dot_n * orientations[i, 1]
            ])
            
            orientation_change += noise_perp
        
        # Update orientation and normalize
        new_orientations[i] = normalize(orientations[i] + orientation_change)
    
    return new_orientations

@njit
def update_curvity(positions, i, goal_position, payload_pos, payload_radius):
    """Update the curvity of a particle. Returns True if there is line of sight"""
    # I think I can vectorize this implementation to make it faster w/ Numba
    # but I dont know if the gains are actually worth the effort.
    # Would be less readable too
    
    # Points
    x_i, y_i = positions[i]
    x_goal, y_goal = goal_position
    x_p, y_p = payload_pos
    
    # Direction vector: particle to goal
    dx, dy = x_goal - x_i, y_goal - y_i
    
    # Direction vector: particle to payload
    fx, fy = x_p - x_i, y_p - y_i
    
    # Coefficients of quadratic equations
    a = dx**2 + dy**2
    b = 2 * (fx * dx + fy * dy)
    c = fx**2 + fy**2 - payload_radius**2
    
    # Discriminant
    discriminant = b**2 - 4 * a * c
    
    if discriminant < 0:
        return True
    else:
        sqrt_discriminant = math.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2 * a)
        t2 = (-b + sqrt_discriminant) / (2 * a)
        
        if (t1 <= 0) or (t2 <= 0): #(0 <= t1 <= 1) or (0 <= t2 <= 1): # Was wrong
            return False
        else:
            return True
    
    # For testing: if the particle is in the top half of the box, it has positive curvity. Otherwise, negative
    # if positions[i, 1] > box_size / 2:
    #     return 0.4
    # else:
    #     return -0.4
    
@njit
def simulate_single_step(positions, orientations, velocities, payload_pos, payload_vel, 
                         radii, v0s, mobilities, payload_mobility, curvity, 
                         stiffness, box_size, payload_radius, dt, rot_diffusion, n_particles):
    """Simulate a single time step"""
    # Compute forces on particles and payload
    particle_forces, payload_force = compute_all_forces_cell_list(
        positions, payload_pos, radii, payload_radius, stiffness, n_particles, box_size
    )
    
    # Update particle orientations
    orientations = update_orientation_vectors(
        orientations, particle_forces, curvity, dt, rot_diffusion, n_particles
    )
    
    # Update particle positions
    for i in range(n_particles):
        
        # Update curvity
        goal_position = [4*(box_size / 5), 4*(box_size / 5)] # static location for now, top right corner.
        if update_curvity(positions, i, goal_position, payload_pos, payload_radius):
            curvity[i] = 0.3
        else:
            curvity[i] = -0.3
        
        # Self-propulsion velocity with particle-specific v0
        self_propulsion = v0s[i] * orientations[i]
        
        # Force-induced velocity with particle-specific mobility
        force_velocity = mobilities[i] * particle_forces[i]
        
        # Total velocity
        velocities[i] = self_propulsion + force_velocity
        
        # Update position
        positions[i] += velocities[i] * dt
    
    # Update payload
    payload_vel = payload_mobility * payload_force
    payload_pos += payload_vel * dt
    
    # Apply periodic boundary conditions
    positions = positions % box_size
    payload_pos = payload_pos % box_size
    
    return positions, orientations, velocities, payload_pos, payload_vel

#####################################################
# Main simulation runner functions                  #
#####################################################

def run_payload_simulation(params):
    """Run the complete payload transport simulation."""
    print(f"Running payload transport simulation with {params['n_particles']} particles for {params['n_steps']} steps...")
    # print(f"Curvity range: {np.min(params['curvity'])} to {np.max(params['curvity'])}")
    # print(f"Payload radius: {params['payload_radius']}")
    # print(f"Rotational diffusion: {params['rot_diffusion']}, Self-propulsion speed: {np.mean(params['v0'])}")
    
    # Initialize arrays
    n_particles = params['n_particles']
    box_size = params['box_size']
    n_steps = params['n_steps']
    save_interval = params['save_interval']
    
    # Initialize particle positions, orientations, and velocities
    positions = np.random.uniform(0, box_size, (n_particles, 2))
    orientations = np.zeros((n_particles, 2))
    velocities = np.zeros((n_particles, 2))
    
    # Initialize random orientations
    for i in range(n_particles):
        angle = np.random.uniform(0, 2*np.pi)
        orientations[i] = np.array([np.cos(angle), np.sin(angle)])
    
    # Initialize payload location. Bottom left corner for now
    payload_pos = np.array([box_size/4, box_size/4 - params['payload_radius']/4])
    payload_vel = np.zeros(2)
    
    # Pre-allocate arrays for storing simulation data
    n_saves = n_steps // save_interval + 1
    saved_positions = np.zeros((n_saves, n_particles, 2))
    saved_orientations = np.zeros((n_saves, n_particles, 2))
    saved_velocities = np.zeros((n_saves, n_particles, 2))
    saved_payload_positions = np.zeros((n_saves, 2))
    saved_payload_velocities = np.zeros((n_saves, 2))
    saved_curvity = np.zeros((n_saves, n_particles))
    
    # Store initial state
    saved_positions[0] = positions.copy()
    saved_orientations[0] = orientations.copy()
    saved_velocities[0] = velocities.copy()
    saved_payload_positions[0] = payload_pos.copy()
    saved_payload_velocities[0] = payload_vel.copy()
    saved_curvity[0] = params['curvity'].copy()
    
    # Run simulation
    start_time = time.time()
    save_idx = 1
    
    for step in range(1, n_steps + 1):
        # Unified simulation step
        positions, orientations, velocities, payload_pos, payload_vel = simulate_single_step(
            positions, orientations, velocities, payload_pos, payload_vel, 
            params['particle_radius'], params['v0'], params['mobility'], params['payload_mobility'], 
            params['curvity'], params['stiffness'], params['box_size'], params['payload_radius'], 
            params['dt'], params['rot_diffusion'], n_particles
        )
        
        # Save data at specified intervals
        if step % save_interval == 0:
            saved_positions[save_idx] = positions
            saved_orientations[save_idx] = orientations
            saved_velocities[save_idx] = velocities
            saved_payload_positions[save_idx] = payload_pos
            saved_payload_velocities[save_idx] = payload_vel
            saved_curvity[save_idx] = params['curvity'].copy()
            save_idx += 1
            
            # Report progress periodically
            if step % (save_interval * 10) == 0:
                print(f"Step {step}:")
                payload_displacement = np.sqrt(np.sum((saved_payload_positions[save_idx-1] - saved_payload_positions[0])**2))
                print(f"  Payload displacement from start: {payload_displacement:.3f}")
    
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    # Calculate payload displacement
    total_payload_displacement = np.sqrt(np.sum((saved_payload_positions[-1] - saved_payload_positions[0])**2))
    print(f"Total payload displacement: {total_payload_displacement:.3f}")
    
    return (
        saved_positions, 
        saved_orientations, 
        saved_velocities, 
        saved_payload_positions, 
        saved_payload_velocities,
        saved_curvity
    )

def create_payload_animation(positions, orientations, velocities, payload_positions, params, 
                            curvity_values, output_file='visualizations/payload_animation_00.mp4'):
    """Create an animation of the payload transport simulation."""
    print("Creating animation...")
    
    # Extract parameters
    box_size = params['box_size']
    payload_radius = params['payload_radius']
    n_particles = params['n_particles']
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set axis limits
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_title('FAABP Cooperative Transport Simulation')
    ax.grid(True, alpha=0.3)
    
    # Map color of particle based on curvity sign
    def get_particle_color(curvity_value):
        return 'red' if curvity_value > 0 else 'darkblue'
    
    # Using scatter for particles instead of individual circles
    particle_colors = [get_particle_color(curvity_values[0, i]) for i in range(n_particles)]
    scatter = ax.scatter(
        positions[0, :, 0], 
        positions[0, :, 1],
        s=np.pi * (params['particle_radius'] * 2)**2,  # Area of circle
        c=particle_colors,
        alpha=0.7
    )
    
    # Removed arrows for now
    # quiver = ax.quiver(
    #     positions[0, :, 0],
    #     positions[0, :, 1],
    #     orientations[0, :, 0],
    #     orientations[0, :, 1],
    #     scale=0.1,
    #     width=0.002,
    #     alpha=0.2,
    #     color='red'
    # )
    
    # Create payload
    payload = Circle(
        (payload_positions[0, 0], payload_positions[0, 1]),
        radius=payload_radius,
        color='gray',
        alpha=0.7
    )
    ax.add_patch(payload)
    
    # TEMPORARY: Static goal indication
    goal_position = [4*(box_size / 5), 4*(box_size / 5)] 
    goal = Circle(
        (goal_position[0], goal_position[1]),
        radius=2,
        color='green'
    )
    ax.add_patch(goal)
    
    # Create payload trajectory
    trajectory, = ax.plot(
        payload_positions[0:1, 0], 
        payload_positions[0:1, 1], 
        'k--', 
        alpha=0.5, 
        linewidth=1.0
    )
    
    # Add parameters text
    params_text = ax.text(-0.02, -0.065, f'n_particles: {n_particles}, curvity range: [{np.min(params["curvity"])}, {np.max(params["curvity"])}], payload radius: {payload_radius}', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top')
    params_text_2 = ax.text(-0.02, -0.093, f'orientational noise: {params["rot_diffusion"][0]}, particle mobility: {params["mobility"][0]}, payload mobility: {params["payload_mobility"]}', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top')
    
    # Add time counter
    time_text = ax.text(0.02, 0.98, 'Frame: 0', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top')
    
    def init():
        """Initialize the animation."""
        return [scatter, payload, trajectory, time_text, params_text, params_text_2]
    
    def update(frame):
        """Update the animation for each frame."""
        # Update time counter
        time_text.set_text(f'Frame: {frame}')
        
        # Report progress periodically
        if frame % 50 == 0:
            print(f"Progress: Frame {frame}")
        
        # Update payload
        payload.center = (payload_positions[frame, 0], payload_positions[frame, 1])
        
        # Update payload trajectory
        trajectory_end = min(frame + 1, len(payload_positions))
        trajectory.set_data(
            payload_positions[:trajectory_end, 0],
            payload_positions[:trajectory_end, 1]
        )
        
        # Particle positions & colors update
        scatter.set_offsets(positions[frame])
        scatter.set_color([get_particle_color(cv) for cv in curvity_values[frame]])

        return [scatter, payload, trajectory, time_text]
    
    # Create animation
    n_frames = positions.shape[0]
    
    sim_seconds_per_real_second = 75 # Increase frame skip for fewer frames to render if its too slow
    target_fps = 15
    
    # Calculate frame skip to maintain consistent sim-time to real-time ratio
    skip = max(1, int(sim_seconds_per_real_second / target_fps))
    
    # Create sequence of frames to include
    frames = range(0, n_frames, skip)
    print(f"Number of frames: {n_frames}")
    
    plt.rcParams['savefig.dpi'] = 170  # Lower dpi for faster rendering
    
    anim = FuncAnimation(
        fig, 
        update, 
        frames=frames, 
        init_func=init, 
        blit=True, 
        interval=120  # Increased from 50
    )
    
    #writer = PillowWriter(fps=target_fps) # for gifs, but its slower
    writer = FFMpegWriter(fps=target_fps, bitrate=1000) # mp4
    
    anim.save(output_file, writer=writer)
    plt.close()
    print(f"Animation saved as '{output_file}'")

#####################################################
# Simulation configuration functions                #
#####################################################

def default_payload_params(n_particles=500):
    """Return default parameters for payload transport simulation"""
    return {
        # Global parameters
        'n_particles': n_particles,    
        'box_size': 350,               
        'dt': 0.01,                  
        'n_steps': 1000,               
        'save_interval': 10,            # Interval for saving data
        'payload_radius': 20.0,        
        'payload_mobility': 0.05,        # Manually kept to 1/r
        'stiffness': 50.0,              
        # Particle-specific parameters
        'v0': np.ones(n_particles) * 3.75,           
        'mobility': np.ones(n_particles) * 1,    # Manually kept to 1/r
        'curvity': np.ones(n_particles) * -0.3,     
        'particle_radius': np.ones(n_particles) * 1, 
        'rot_diffusion': np.ones(n_particles) * 0.05, 
    }
    

# Currently unused
def heterogeneous_curvity(params):
    """Make half of the particles have a positive curvity (default is currently all of them negative)."""
    n_particles = params['n_particles']
    half_particles = n_particles // 2
    curvity = np.ones(n_particles) * params['curvity'][0]  # Use first value as baseline
    curvity[:half_particles] = 0.4 # Make half of the particles have opposite curvity
    params['curvity'] = curvity
    return params

def save_simulation_data(filename, positions, orientations, velocities, payload_positions, 
                        payload_velocities, params, curvity_values):
    """Save simulation data including individual particle parameters."""
    np.savez(
        filename,
        # Frame-specific data
        positions=positions,
        orientations=orientations,
        velocities=velocities,
        payload_positions=payload_positions,
        payload_velocities=payload_velocities,
        curvity_values=curvity_values,
        # Parameters
        # params['curvity'] accessible through curvity_values[-1]
        v0=params['v0'],
        mobility=params['mobility'],
        particle_radius=params['particle_radius'],
        payload_mobility=params['payload_mobility'],
        payload_radius=params['payload_radius'],
        box_size=params['box_size'],
        dt=params['dt'],
        stiffness=params['stiffness']
    )

#####################
# Main execution    #
#####################

if __name__ == "__main__":
    # Set simulation parameters
    params = default_payload_params()
    # params = heterogeneous_curvity(params) # This randomly sets half of the particles to a positive curvity. Just for some early testing

    # Run simulation
    positions, orientations, velocities, payload_positions, payload_velocities, curvity_values = run_payload_simulation(params)
    
    # Save timestamp to use in filenames
    T = int(time.time())
    
    # Save simulation data
    save_simulation_data(
        # Include timestamp in filename
        'data/payload_simulation_data_{timestamp}.npz'.format(timestamp=T),
        positions, orientations, velocities, payload_positions, payload_velocities, params,
        curvity_values
    )
    
    # Create animation with frame-specific curvity values
    create_payload_animation(positions, orientations, velocities, payload_positions, params, 
                             curvity_values, 'visualizations/payload_animation_{timestamp}.mp4'.format(timestamp=T))
    
    print("Payload simulation and animation completed successfully!")