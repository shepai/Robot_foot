import numpy as np
import matplotlib.pyplot as plt

def generate_dome(R=1.0, n_layers=20, n_total=300,
                  top_bias_strength=2.0, min_pts=6):

    points = []
    actual_layer_counts = []

    z_vals = np.linspace(0, R, n_layers)

    circumferences = 2 * np.pi * np.sqrt(
        np.maximum(R**2 - z_vals**2, 0)
    )

    top_bias = (1 - z_vals / R) ** top_bias_strength + 0.3

    weights = circumferences * top_bias
    weights = weights / weights.sum()

    layer_counts = np.maximum(
        min_pts,
        (weights * n_total).astype(int)
    )

    for z, n_pts in zip(z_vals, layer_counts):

        r = np.sqrt(max(R**2 - z**2, 0))

        # -------------------------
        # apex layer
        # -------------------------
        if r < 1e-8:
            points.append((0.0, 0.0, R))
            actual_layer_counts.append(1)
            continue

        theta = np.linspace(0, 2*np.pi, n_pts, endpoint=False)

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        for xi, yi in zip(x, y):
            points.append((xi, yi, z))

        actual_layer_counts.append(n_pts)

    return np.array(points) / 10.0, actual_layer_counts

def generate_xml(points, num,stiff=300,damp=20):
    if sum(num) != len(points):
        raise ValueError(
            f"Topology mismatch: sum(num)={sum(num)} vs len(points)={len(points)}"
        )
    xml = """<mujoco model="tactip_stable_octagonal_vault">
     <option timestep="0.008" gravity="0 0 -9.81" integrator="implicitfast" solver="Newton" tolerance="1e-8"/>
     <asset>
         <material name="black_mesh_mat" rgba="0.05 0.05 0.05 1" shininess="0.1"/>
    </asset>
    <worldbody>
        <light pos="0 0 4" dir="0 0 -1"/>
        <geom name="floor" type="plane" size="2 2 .1" rgba=".8 .8 .8 1"/>
        <body name="dropping_box" pos="0 0 0.4">
            <freejoint name="box_free"/>

            <geom type="box" size="0.04 0.04 0.04" rgba="0.8 0.5 0.1 1" mass="0.2" friction="1 0.005 0.005" condim="3" contype="1" conaffinity="1"/>
        </body>
        <body name="dropping_box2" pos="0.02 0.04 0.4">
            <freejoint name="box_free2"/>

            <geom type="box" size="0.04 0.04 0.04" rgba="0.8 0.5 0.1 1" mass="0.2" friction="1 0.005 0.005" condim="3" contype="1" conaffinity="1"/>
        </body>
        <body name="flexible_structure" pos="0 0 0.8" quat="0 1 0 0">
            <freejoint/>
            <inertial pos="0 0 0.03" mass="0.1" diaginertia="0.0005 0.0005 0.0005"/>
    """

    # -------------------------
    # create nodes
    # -------------------------
    allnames=""
    for i, point in enumerate(points):
        xml += f"""
        <body name="node_{i}" pos="{point[0]} {point[1]} {point[2]}">
            <joint type="slide" axis="1 0 0" name="j_c{i}_x" stiffness="{stiff}" damping="{damp}"/>
            <joint type="slide" axis="0 1 0" name="j_c{i}_y" stiffness="{stiff}" damping="{damp}"/>
            <joint type="slide" axis="0 0 1" name="j_c{i}_z" stiffness="{stiff}" damping="{damp}"/>
            <geom type="sphere" size="0.009" rgba="1 1 1 1" mass="0.01" condim="3" contype="1" conaffinity="1"/>
            <site name="s_c{i}" pos="0 0 0" size="0.002"/>
        </body>
        """
        allnames+= f"node_{i} "
    xml += """
        </body>
    </worldbody>
    <tendon>
    """

    # -------------------------
    # build layer structure
    # -------------------------
    layers = []
    idx = 0
    for n in num:
        layers.append(list(range(idx, idx + n)))
        idx += n
    for i, layer in enumerate(layers):
        n = len(layer)

        for j in range(n):
            if n != 1:
                a = layer[j]
                b = layer[(j + 1) % n]

                xml += f"""
                <spatial name="t_h_{i}_{j}" solreflimit="0.008 1" solimplimit="0.95 0.99 0.001">
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
            <spatial name="t_v_{i}_{j}" solreflimit="0.008 1" solimplimit="0.95 0.99 0.001">
                <site site="s_c{a}"/>
                <site site="s_c{b}"/>
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
    </mujoco>
    """

    return xml
# ---- generate + plot ----
pts,ln = generate_dome(R=2.0, n_layers=10, n_total=250)
xml=generate_xml(pts,ln,stiff=150,damp=3) 
with open("/home/dexter/Documents/GitHub/Robot_foot/Mujoco_work/tactip/generated.xml","w") as file:
    file.write(xml)
"""fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=10)

ax.set_title("Dome Point Grid")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()"""