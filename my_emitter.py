import hou
import numpy as np

node = hou.pwd()
geo = node.geometry()
incoming_geo = node.inputs()[0].geometry()

geo.clear()
geo.merge(incoming_geo)

emit_rate = 5
emit_radius = 0.01  
emit_center = hou.Vector3(0, 1.5, 0)
initial_velocity = hou.Vector3(0, -1.0, 0) 

for i in range(emit_rate):
    # Small circular emission offset
    angle = np.random.uniform(0, 2 * np.pi)
    radius = np.random.uniform(0, emit_radius)
    offset_x = np.random.uniform(-emit_radius, emit_radius)
    offset_z = np.random.uniform(-emit_radius, emit_radius)
    #offset_x = np.cos(angle) * radius
    #offset_z = np.sin(angle) * radius
    offset_y = 0  # no vertical offset

    position = emit_center + hou.Vector3(offset_x, offset_y, offset_z)
    
    velocity = initial_velocity  # no randomness

    p = geo.createPoint()
    p.setPosition(position)
    p.setAttribValue("v", velocity)
    p.setAttribValue("last_pos", position)
    p.setAttribValue("last_vel", velocity)
    p.setAttribValue("mass", 1.0)
    p.setAttribValue("rho", 1000.0)
    p.setAttribValue("press", 0.0)
    p.setAttribValue("Cd", hou.Vector3(0, 0, 1)) 
