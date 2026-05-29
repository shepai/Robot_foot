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

def generate_xml(points, num,stiff=300,damp=10):
    if sum(num) != len(points):
        raise ValueError(
            f"Topology mismatch: sum(num)={sum(num)} vs len(points)={len(points)}"
        )
    xml = """<mujoco model="tactip_stable_octagonal_vault">
    <option timestep="0.0002" gravity="0 0 -9.81"/>
    
    <worldbody>
        <light pos="0 0 4" dir="0 0 -1"/>
        <geom name="floor" type="plane" size="2 2 .1" rgba=".8 .8 .8 1"/>
        <body name="dropping_box" pos="0 0 1.2">
            <freejoint name="box_free"/>

            <geom type="box" size="0.04 0.04 0.04" rgba="0.8 0.5 0.1 1" mass="0.2" friction="1 0.005 0.005"/>
        </body>
        <body name="flexible_structure" pos="0 0 0.8">
            <freejoint/>
            <inertial pos="0 0 0.03" mass="0.1" diaginertia="0.0005 0.0005 0.0005"/>
    """

    # -------------------------
    # create nodes
    # -------------------------
    for i, point in enumerate(points):
        xml += f"""
        <body name="node_{i}" pos="{point[0]} {point[1]} {point[2]}">
            <joint type="slide" axis="1 0 0" name="j_c{i}_x" stiffness="5" damping="3"/>
            <joint type="slide" axis="0 1 0" name="j_c{i}_y" stiffness="5" damping="3"/>
            <joint type="slide" axis="0 0 1" name="j_c{i}_z" stiffness="5" damping="3"/>
            <geom type="sphere" size="0.005" rgba="0.0 0.6 0.0 1" mass="0.01"/>
            <site name="s_c{i}" pos="0 0 0" size="0.002"/>
        </body>
        """

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
                <spatial name="t_h_{i}_{j}" stiffness="{stiff}" damping="{damp}" >
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
            <spatial name="t_v_{i}_{j}" stiffness="{stiff}" damping="{damp}">
                <site site="s_c{a}"/>
                <site site="s_c{b}"/>
            </spatial>"""

    xml += """
    </tendon>
    </mujoco>
    """

    return xml
# ---- generate + plot ----
pts,ln = generate_dome(R=2.0, n_layers=10, n_total=200)
xml=generate_xml(pts,ln) 
with open("/home/dexter/Documents/GitHub/Robot_foot/Mujoco_work/tactip/generated.xml","w") as file:
    file.write(xml)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=10)

ax.set_title("Dome Point Grid")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()