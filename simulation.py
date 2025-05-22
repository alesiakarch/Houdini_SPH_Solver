import numpy as np
import hou

node = hou.pwd()
geo = node.geometry()
vdb_geo = node.inputs()[1].geometry()

# Find the VDB primitive
vdb_prim = next((prim for prim in vdb_geo.prims() if isinstance(prim, hou.VDB)), None)
if vdb_prim is None:
    raise ValueError("No VDB primitive found in input 2!")

# Parameters
fps = 24.0
timestep = 1.0 / fps
gravity = hou.Vector3(0.0, -9.8, 0.0)
particle_radius = 0.002
smoothing_length = 0.06
damping = 0.9

rest_density = 1000.0
gas_stiffness = 2000.0
epsilon = 1e-6  # small value to avoid divide by zero

# SPH kernels
def poly6(r, h):
    if 0 <= r <= h:
        return (315.0 / (64.0 * np.pi * h**9)) * (h*h - r*r)**3
    return 0.0

def spiky_grad(r_vec, h):
    r = r_vec.length()
    if 0 < r <= h:
        return (-45.0 / (np.pi * h**6)) * ((h - r)**2) * (r_vec.normalized())
    return hou.Vector3(0, 0, 0)

points = geo.points()

# Cache positions and velocities for neighbor search and updates
positions = [hou.Vector3(p.attribValue("last_pos")) if hou.frame() != 1 else p.position() for p in points]

velocities = [hou.Vector3(p.attribValue("last_vel")) if hou.frame() != 1 else hou.Vector3(p.attribValue("v")) for p in points]

for i, point in enumerate(points):
    print(f"Particle {i} start pos: {point.position()}")
    if hou.frame() == 1: # at frame 1 do:
        pos = point.position()
        vel = hou.Vector3(point.attribValue("v"))
        point.setAttribValue("last_pos", pos)
        point.setAttribValue("last_vel", vel)
        
    else:
        pos = positions[i]
        velocity = velocities[i]

        substeps = 15
        sub_timestep = timestep / substeps

        for _ in range(substeps):
            # Gravity
            velocity += gravity * sub_timestep

            # Density estimation weighted by mass
            print(f"Particle {i} pos: {pos}")
            density = 0.0
            for j, pj in enumerate(points):
                #print(f"Particle {j} pos: {positions[j]}")
                r_vec = pos - positions[j]
                r = r_vec.length()
                #print(f"distance between particle i and neighbor j: {r}, with smooth_length {smoothing_length}")
                if r <= smoothing_length:
                    m_j = pj.attribValue("mass")
                    print(f"Mass of the neighbor {pj} = {m_j}")
                    density += m_j * poly6(r, smoothing_length)
                    print(f"density after the initial calculation: {density}")
            density = max(density, epsilon)
            point.setAttribValue("rho", density)

            
            # # Color based on density
            # if density <= epsilon * 1.1:  # particles with very low density (close to epsilon)
            #     color = hou.Vector3(1, 0, 0)  # red
            # else:
            #     max_density = 2000.0
            #     t = min(density / max_density, 1.0)
            #     color = hou.Vector3(0, t, 1 - t)  # blue to green gradient
            
            # Pressure
            pressure = gas_stiffness * max(density - rest_density, 0.0)
            point.setAttribValue("p", pressure)

            # Pressure force calculation
            pressure_force = hou.Vector3(0, 0, 0)
            for j, pj in enumerate(points):
                if i == j:
                    continue
                r_vec = pos - positions[j]
                r = r_vec.length()
                if r > smoothing_length or r == 0:
                    continue
                m_j = pj.attribValue("mass")
                rho_j = max(pj.attribValue("rho"), epsilon)
                p_j = pj.attribValue("p")
                #pressure_term = (pressure / (density * density) + p_j / (rho_j * rho_j)) * m_j
                pressure_term = ((pressure + p_j) / (2 * rho_j)) * m_j
                pressure_force += -pressure_term * spiky_grad(r_vec, smoothing_length)

            # Update velocity with pressure acceleration
            acceleration = pressure_force
            velocity += acceleration * sub_timestep
            
            # Color based on pressure
            max_pressure = 50000.0  # adjust depending on your gas_stiffness
            t = min(pressure / max_pressure, 1.0)
            color = hou.Vector3(t, 0, 1 - t)  # red = high pressure, blue = low
            
            # Viscosity force (optional but recommended for realism)
            viscosity_force = hou.Vector3(0, 0, 0)
            viscosity_coef = 0.01  # tweak this
            for j, pj in enumerate(points):
                if i == j:
                    continue
                r_vec = pos - positions[j]
                r = r_vec.length()
                if r > smoothing_length or r == 0:
                    continue
                v_j = velocities[j]
                m_j = pj.attribValue("mass")
                rho_j = max(pj.attribValue("rho"), epsilon)
                viscosity_force += (v_j - velocity) * (m_j / rho_j) * poly6(r, smoothing_length)
            viscosity_force *= viscosity_coef
            velocity += viscosity_force * sub_timestep

            # Predict new position
            predicted_pos = pos + velocity * sub_timestep
            print(f"Particle {i} predicted pos: {predicted_pos}")
            
            print(f"Particle {i} updated pos: {pos}")

            # Collision with collider (VDB)
            # dist_to_surface = vdb_prim.sample(predicted_pos)
            # if dist_to_surface < particle_radius:
            #     velocity = -velocity * damping
            #     predicted_pos = pos
            #     break
            
            # dist_to_surface = vdb_prim.sample(predicted_pos)
            # if dist_to_surface < particle_radius:
            #     normal = vdb_prim.gradient(predicted_pos).normalized()
            #     correction = (particle_radius - dist_to_surface) * normal
            #     predicted_pos += correction
            #     velocity -= velocity.dot(normal) * normal  # Remove velocity into the surface
            #     velocity *= damping
            max_correction_iters = 1
            for _ in range(max_correction_iters):
                dist_to_surface = vdb_prim.sample(predicted_pos)
                if dist_to_surface >= particle_radius:
                    break
                normal = vdb_prim.gradient(predicted_pos)
                if normal.length() == 0:
                    break  # avoid dividing by zero
                normal = normal.normalized()
                correction = (particle_radius - dist_to_surface) * normal
                predicted_pos += correction
                velocity -= velocity.dot(normal) * normal
                velocity *= damping
            pos = predicted_pos
            print(f"Particle {i} after collision pos: {pos}")
            
                    # === Sanity checks ===
            max_pos = 10.0  # adjust to your scene scale
            if (abs(predicted_pos[0]) > max_pos or
                abs(predicted_pos[1]) > max_pos or
                abs(predicted_pos[2]) > max_pos):
                print(f"Warning: particle {i} position out of bounds: {predicted_pos}")
                predicted_pos = hou.Vector3(0, 1.5, 0)  # or last valid position
                velocity = hou.Vector3(0, 0, 0)
            
            vel_mag = velocity.length()
            if vel_mag > 100:
                print(f"Warning: particle {i} velocity too large: {velocity}")
                velocity = hou.Vector3(0, 0, 0)

        # Commit updates
        point.setPosition(pos)
        point.setAttribValue("v", velocity)
        point.setAttribValue("last_pos", pos)
        point.setAttribValue("last_vel", velocity)
        point.setAttribValue("Cd", color)

        positions[i] = pos
        velocities[i] = velocity
        
        print(f"Particle {i} position: {pos}")
        print(f"Particle {i} velocity: {velocity}")
