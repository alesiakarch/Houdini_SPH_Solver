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
particle_radius = 0.03
smoothing_length = 5 * particle_radius
damping = 0.98
spacing = 0.06

rest_density = 800.0
gas_stiffness = 500.0
epsilon = 1e-6  # small value to avoid divide by zero
particle_mass = rest_density * (spacing ** 3)

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

# cel function
def get_cell(pos, cell_size):
    return (int(pos[0] // cell_size),
            int(pos[1] // cell_size),
            int(pos[2] // cell_size))

def get_neighbor_cells(cell):
    neighbors = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                neighbors.append((cell[0] + dx, cell[1] + dy, cell[2] + dz))
    return neighbors

points = geo.points()
for p in geo.points():
    if not p.attribValue("last_pos_initialized"):
        p.setAttribValue("last_pos", p.position())
        p.setAttribValue("last_vel", hou.Vector3(p.attribValue("v")))
        p.setAttribValue("last_pos_initialized", 1)

# Cache positions and velocities for neighbor search and updates
positions = [hou.Vector3(p.attribValue("last_pos")) if hou.frame() != 1 else p.position() for p in points]

velocities = [hou.Vector3(p.attribValue("last_vel")) if hou.frame() != 1 else hou.Vector3(p.attribValue("v")) for p in points]

for i, point in enumerate(points):
    point.setAttribValue("mass", particle_mass)
    if hou.frame() == 1: # at frame 1 do:
        pos = point.position()
        vel = hou.Vector3(point.attribValue("v"))
        point.setAttribValue("last_pos", pos)
        point.setAttribValue("last_vel", vel)
        
    else:
    
        # Build neighbor grid
        cell_size = smoothing_length
        grid = {}
        for j, pos_j in enumerate(positions):
            cell = get_cell(pos_j, cell_size)
            if cell not in grid:
                grid[cell] = []
            grid[cell].append(j)

        pos = positions[i]
        velocity = velocities[i]

        substeps = 20
        sub_timestep = timestep / substeps

        for _ in range(substeps):
            # Gravity
            velocity += gravity * sub_timestep
           
            # Density estimation weighted by mass
            density = 0.0
            neighbours_count = 0
            my_cell = get_cell(pos, cell_size)

            for neighbor_cell in get_neighbor_cells(my_cell):
                if neighbor_cell not in grid:
                    continue
    
                for j in grid[neighbor_cell]:
                    r_vec = pos - positions[j]
                    r = r_vec.length()
                    if r <= smoothing_length:
                        m_j = points[j].attribValue("mass")
                        density += m_j * poly6(r, smoothing_length)
               
                        neighbours_count += 1
            density = max(density, epsilon)           
       
            point.setAttribValue("rho", density)
            point.setAttribValue("neighbours", neighbours_count)
   
            # Pressure

            pressure = gas_stiffness * max(density - rest_density, 0.0)
            pressure = min(pressure, 5000.0)  # cap it
            point.setAttribValue("press", pressure)

            # Pressure force calculation
            pressure_force = hou.Vector3(0, 0, 0)
            for neighbor_cell in get_neighbor_cells(my_cell):
                if neighbor_cell not in grid:
                    continue
                for j in grid[neighbor_cell]:
                    if i == j:
                        continue
                    r_vec = pos - positions[j]
                    r = r_vec.length()
                    if r > smoothing_length or r == 0:
                        continue
                    m_j = points[j].attribValue("mass")
                    rho_j = max(points[j].attribValue("rho"), epsilon)
                    p_j = points[j].attribValue("press")
                    pressure_term = m_j * ((pressure / (density * density)) + (p_j / (rho_j * rho_j)))
                    pressure_force += -pressure_term * spiky_grad(r_vec, smoothing_length)


            # Update velocity with pressure acceleration
            acceleration = pressure_force
            velocity += acceleration * sub_timestep
            
            # Color based on pressure
            max_pressure = 5000.0  # adjust depending on your gas_stiffness
            t = min(pressure / max_pressure, 1.0)
            color = hou.Vector3(t, 0, 1 - t)  # red = high pressure, blue = lo

            
            # Viscosity force (optional but recommended for realism)
            viscosity_force = hou.Vector3(0, 0, 0)
            viscosity_coef = 0.15  # tweak this
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
            
            # Surface tension force
            surface_tension_force = hou.Vector3(0, 0, 0)
            tension_coef = 1.2  
            surface_threshold = 20 

            if neighbours_count < surface_threshold:
                for neighbor_cell in get_neighbor_cells(my_cell):
                    if neighbor_cell not in grid:
                        continue
                    for j in grid[neighbor_cell]:
                        if i == j:
                            continue
                        r_vec = pos - positions[j]
                        r = r_vec.length()
                        if r > smoothing_length or r == 0:
                            continue
                        m_j = points[j].attribValue("mass")
                        rho_j = max(points[j].attribValue("rho"), epsilon)
                        direction = r_vec.normalized()
                        attraction = -tension_coef * poly6(r, smoothing_length)
                        surface_tension_force += (m_j / rho_j) * attraction * direction
        
                velocity += surface_tension_force * sub_timestep



            # Predict new position
            predicted_pos = pos + velocity * sub_timestep
      
            collided = False
            max_correction_iters = 3
            for _ in range(max_correction_iters):
                dist_to_surface = vdb_prim.sample(predicted_pos)
                if dist_to_surface >= particle_radius:
                    break
                normal = vdb_prim.gradient(predicted_pos)
                if normal.length() == 0:
                    break  # avoid dividing by zero
                normal = normal.normalized()
                correction = (particle_radius - dist_to_surface) * normal
                correction *= 0.5  
            
                predicted_pos += correction
                collided = True
            
            if collided:
                # Push velocity along surface to avoid bounce
                normal = vdb_prim.gradient(predicted_pos).normalized()
                velocity -= velocity.dot(normal) * normal  
                velocity *= 0.5  
            
                # Optional: kill tiny jiggles
                if velocity.length() < 0.01:
                    velocity = hou.Vector3(0, 0, 0)
            else:
                velocity *= 0.98  # global drag (optional)

            pos = predicted_pos
            # XSPH smoothing (velocity blending)
            xsph_coef = 0.2
            xsph_correction = hou.Vector3(0, 0, 0)
            for neighbor_cell in get_neighbor_cells(my_cell):
                if neighbor_cell not in grid:
                    continue
                for j in grid[neighbor_cell]:
                    if i == j:
                        continue
                    r_vec = pos - positions[j]
                    r = r_vec.length()
                    if r > smoothing_length or r == 0:
                        continue
                    v_j = velocities[j]
                    m_j = points[j].attribValue("mass")
                    rho_j = max(points[j].attribValue("rho"), epsilon)
                    xsph_correction += (v_j - velocity) * (m_j / rho_j) * poly6(r, smoothing_length)
            velocity += xsph_coef * xsph_correction

            
            # Sanity checks
            max_pos = 10.0  # adjust to your scene scale
            if (abs(predicted_pos[0]) > max_pos or
                abs(predicted_pos[1]) > max_pos or
                abs(predicted_pos[2]) > max_pos):
                print(f"Warning: particle {i} position out of bounds: {predicted_pos}")
                predicted_pos = hou.Vector3(0, 1.5, 0)  # or last valid position
                velocity = hou.Vector3(0, 0, 0)
            
            vel_mag = velocity.length()
            if vel_mag > 20:
                print(f"Warning: particle {i} velocity too large: {velocity}")
                velocity = hou.Vector3(0, 0, 0)
                velocity = velocity.normalized() * 10
            if neighbours_count < 8:
                point.setAttribValue("Cd", hou.Vector3(1, 1, 0))  # yellow: under-sampled
            elif pressure > 3000:
                point.setAttribValue("Cd", hou.Vector3(1, 0, 0))  # red: high pressure


        # Commit updates
        point.setPosition(pos)
        point.setAttribValue("v", velocity)
        point.setAttribValue("last_pos", pos)
        point.setAttribValue("last_vel", velocity)
        point.setAttribValue("Cd", color)

        positions[i] = pos
        velocities[i] = velocity
        
