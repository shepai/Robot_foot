import trimesh

# 1. Generate an icosphere 
# subdivisions=2 creates exactly 42 vertices and 80 faces (perfect for fast physics)
# subdivisions=3 creates exactly 162 vertices and 320 faces (smoother, still ultra-fast)
low_poly_ball = trimesh.creation.icosphere(subdivisions=2, radius=0.1) # 10cm radius

# 2. Run standard cleanup just to guarantee topological compliance
low_poly_ball.merge_vertices()
low_poly_ball.process(validate=True)

# 3. Save the clean asset for your MuJoCo environment
output_path = '/home/dexter/Documents/GitHub/Robot_foot/Mujoco_work/tactip/clean_ball.obj'
low_poly_ball.export(output_path)

print(f"Generated ball successfully!")
print(f"Total Vertices: {len(low_poly_ball.vertices)}")
print(f"Total Faces: {len(low_poly_ball.faces)}")