from Robot_foot import PressTipSensor
import numpy as np

#load in the sensor model
presstip = PressTipSensor()
data=np.random.random((1,64))
print("Friction:",presstip.predict_friction(data))
data=np.random.random((1,10,4))
print("Classification:",presstip.predict_texture(data))