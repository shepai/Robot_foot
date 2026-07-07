import numpy as np
import matplotlib.pyplot as plt

def generate_dome(R=1.0, n_layers=20, n_total=300,
                  tip_layer_density=2.0, min_pts=6,
                  remove_bottom_layers=1):
    layers = []  # each element is (Ni x 3)
    actual_layer_counts = []

    # -------------------------
    # Z layer generation
    # -------------------------
    t = np.linspace(0, 1, n_layers)
    z_vals = R * (np.sin(t * np.pi / 2) ** (1 / tip_layer_density))

    circumferences = 2 * np.pi * np.sqrt(np.maximum(R**2 - z_vals**2, 0))
    weights = circumferences + 1.0
    weights = weights / weights.sum()

    layer_counts = (weights * n_total).astype(int)

    # -------------------------
    # Build layers
    # -------------------------
    for z, n_pts in zip(z_vals, layer_counts):

        r = np.sqrt(max(R**2 - z**2, 0))

        # apex handling
        if r < 1e-8:
            layers.append(np.array([[0.0, 0.0, R]]))
            actual_layer_counts.append(1)
            continue

        n_pts = max(min_pts, n_pts)

        theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        layer = np.column_stack([x, y, np.full_like(x, z)])

        layers.append(layer)
        actual_layer_counts.append(n_pts)

    # -------------------------
    # REMOVE TOP LAYERS (CORRECT)
    # -------------------------
    if remove_bottom_layers > 0:
        layers = layers[remove_bottom_layers:]
        actual_layer_counts = actual_layer_counts[remove_bottom_layers:]

    # -------------------------
    # FLATTEN FINAL POINTS
    # -------------------------
    points = np.vstack(layers)

    return points / 10.0, actual_layer_counts
def generate_xml(points, num,stiff=300,damp=20):
    if sum(num) != len(points):
        raise ValueError(
            f"Topology mismatch: sum(num)={sum(num)} vs len(points)={len(points)}"
        )
    xml = f"""<mujoco model="flexible_structure" >
     <option integrator="implicitfast" timestep="0.001"/> 
     <asset>
         <material name="black_mesh_mat" rgba="0.05 0.05 0.05 1" shininess="0.1"/>
    </asset>
       <worldbody>
        <body name="flexible_structure" pos="0 0 0" quat="0 0.7071 0.7071 0">
            <inertial pos="0 0 0" mass="0.1" diaginertia="0.0005 0.0005 0.0005"/>
            """
    xml += f"""
        <body name="cylinder_mount" pos="0 0 -0.02">
            <geom type="cylinder" size="0.04 0.02" mass="0.05" rgba="1 1 1 1" />
            
            </body>
            <camera name="sensor_cam" pos="0 0 -0.002" fovy="120" zaxis="0 0 -1" />

            <light name="sensor_light" pos="0 0 -0.1" dir="0 0 -1" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2" directional="true" />
    
    """
    layers = []
    idx = 0
    for n in num:
        layers.append(list(range(idx, idx + n)))
        idx += n
    # -------------------------
    # create nodes
    # -------------------------
    allnames=""
    for i in range(len(points)):
        point = points[i]
        inner_pos = point * -0.2
        xml += f"""
        <body name="node_{i}" pos="{point[0]} {point[1]} {point[2]}">
            <joint type="slide" springref="0" limited="true" range="-0.02 0.02" axis="1 0 0" name="j_c{i}_x" stiffness="{stiff}" damping="{damp}"  armature="0.001"/>
            <joint type="slide" springref="0" limited="true" range="-0.02 0.02" axis="0 1 0" name="j_c{i}_y" stiffness="{stiff}" damping="{damp}"  armature="0.001"/>
            <joint type="slide" springref="0" limited="true" range="-0.02 0.02" axis="0 0 1" name="j_c{i}_z" stiffness="{stiff}" damping="{damp}"  armature="0.001"/>
            <geom type="sphere" size="0.001" rgba="1 1 1 0" mass="0.001" condim="3" contype="1" conaffinity="1" group="1"/>
            <geom pos="{inner_pos[0]} {inner_pos[1]} {inner_pos[2]}" type="sphere" size="0.002" rgba="1 1 1 1" 
            
                group="1" 
                contype="0" 
                conaffinity="0" 
                mass="0" 
                density="0"/>
            <site name="s_c{i}" pos="0 0 0" size="0.002"/>
        </body>
        """
        allnames+= f"node_{i} "
    xml += """
        </body>
</worldbody>
    <tendon>
    """
    
    for i, layer in enumerate(layers):
        n = len(layer)

        for j in range(n):
            if n != 1:
                a = layer[j]
                b = layer[(j + 1) % n]
                xml += f"""
                <spatial width="0.0015" name="t_h_{i}_{j}" 
                        damping="2"
                        solreflimit="0.008 1" solimplimit="0.95 0.99 0.001">
                    <site site="s_c{a}"/>
                    <site site="s_c{b}"/>
                </spatial>"""

    for i in range(len(layers) - 1):
        curr = layers[i]
        nxt = layers[i + 1]

        n_curr = len(curr)
        n_nxt = len(nxt)

        for j, a in enumerate(curr):
            k = int(j * n_nxt / n_curr)
            k = min(max(k, 0), n_nxt - 1)
            b = nxt[k]
            if a == b:
                continue

            xml += f"""
            <spatial width="0.0015" name="t_v_{i}_{j}" solreflimit="0.008 1" solimplimit="0.95 0.99 0.001">
                <site site="s_c{a}"/>
                <site site="s_c{b}"/>
            </spatial>"""
    center = layers[0][0]
    spoke_stiff = 1500
    spoke_damp = 15
    for i in range(len(layers)):
        curr = layers[i]

        for j in range(0, len(curr), 3):  
            a = curr[j]

            if a == center:
                continue

            xml += f"""
            <spatial width="0.0005"
                    rgba="0 1 0 0"
                    name="t_spoke_{i}_{j}"
                    stiffness="{spoke_stiff}"
                    damping="{spoke_damp}"
                    solreflimit="0.005 1"
                    solimplimit="0.9 0.95 0.001">
                <site site="s_c{center}"/>
                <site site="s_c{a}"/>
            </spatial>"""
    elements = []
    for i in range(len(layers) - 1):
        curr = layers[i]
        nxt = layers[i + 1]
        n_curr = len(curr)
        n_nxt = len(nxt)

        if n_curr == 1: # Apex layer fanning out
            apex = curr[0]
            for j in range(n_nxt):
                elements.append(f"{apex} {nxt[j]} {nxt[(j + 1) % n_nxt]}")
        else: # Standard concentric rings stitching together
            for j in range(n_curr):
                c1, c2 = curr[j], curr[(j + 1) % n_curr]
                k1 = min(max(int(j * n_nxt / n_curr), 0), n_nxt - 1)
                k2 = min(max(int(((j + 1) % n_curr) * n_nxt / n_curr), 0), n_nxt - 1)
                
                # Draw the structural triangle grids between the layers
                elements.append(f"{c1} {nxt[k1]} {c2}")
                if k1 != k2:
                    elements.append(f"{c2} {nxt[k1]} {nxt[k2]}")

    element_str = "   ".join(elements)
    body_str = " ".join([f"node_{i}" for i in range(len(points))])
    vertex_str = "   ".join(["0 0 0" for _ in range(len(points))])
    xml += f"""
    </tendon>
    <deformable>
    <flex name="black_skin" 
              material="black_mesh_mat" 
              dim="2"
              body="{body_str}"
              vertex="{vertex_str}"
              element="{element_str}"/>
    </deformable>
    """
    xml += "<equality>\n"
    for i in range(num[0]):
        xml += f'  <joint joint1="j_c{i}_x" polycoef="0 1 0 0 0"/>\n'
        xml += f'  <joint joint1="j_c{i}_y" polycoef="0 1 0 0 0"/>\n'
        xml += f'  <joint joint1="j_c{i}_z" polycoef="0 1 0 0 0"/>\n'
    xml += """</equality>\n
    </mujoco>
    """
    
    return xml
# ---- generate + plot ----
pts,ln = generate_dome(R=0.35, n_layers=5, n_total=80,remove_bottom_layers=0)
xml=generate_xml(pts,ln,stiff=150,damp=3) 
with open("/home/dexter/Documents/GitHub/Robot_foot/Mujoco_work/tactip/generating/generated.xml","w") as file:
    file.write(xml)
import seperate
"""fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=10)

ax.set_title("Dome Point Grid")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()"""