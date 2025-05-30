from Robot_foot import opticalSensor, keys
import numpy as np

#load in the sensor model
tactip = opticalSensor()
data=np.random.random((1,4,110*120)) #needs at least 4 frames of video
print("Classification:",keys[tactip.predict_texture(data)])
data=np.random.random((1,4,110*120)) #needs at least 10 frames of video
print("Friction:",tactip.predict_friction(data))
