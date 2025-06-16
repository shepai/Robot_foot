from Robot_foot import opticalSensor
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib
matplotlib.use('TkAgg')  
#load in the sensor model
tactip = opticalSensor()
#load in image 
image=cv2.imread("assets/example_image.png",cv2.IMREAD_GRAYSCALE)
coord=tactip.predict_points(image.reshape((1,-1))) #preduct points

plt.imshow(image,cmap="gray")
plt.scatter(coord[:,0],coord[:,1],c="y")
plt.axis("off")
plt.tight_layout()
plt.savefig("assets/predictions.pdf")
plt.show()