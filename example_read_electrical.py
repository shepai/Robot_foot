from Robot_foot import PressTipSensor, keys
import numpy as np

#load in the sensor model
presstip = PressTipSensor()
data=np.random.random((1,10,4)) #needs 10 frames
print("Friction:",presstip.predict_friction(data))
data=np.random.random((1,10,4)) #needs 10 frames of video
print("Classification:",keys[presstip.predict_texture(data)])