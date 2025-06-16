from Robot_foot import opticalSensor, keys, ImageDataPreprocessor
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
#load in the sensor model
tactip = opticalSensor()
processor=ImageDataPreprocessor()
#load in image 
image=cv2.imread("assets/example_image.png",cv2.IMREAD_GRAYSCALE)
#image=processor.sobel(image)
coord=tactip.predict_points(image.reshape((1,-1))) #preduct points

plt.imshow(image,cmap="gray")
plt.scatter(coord[:,0],coord[:,1],c="y")
plt.axis("off")
plt.tight_layout()
plt.savefig("assets/predictions.pdf")
plt.show()