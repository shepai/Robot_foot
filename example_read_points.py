from Robot_foot import opticalSensor, keys, ImageDataPreprocessor
import numpy as np
import matplotlib.puplot as plt
import cv2

#load in the sensor model
tactip = opticalSensor()
processor=ImageDataPreprocessor()
#load in image 
image=cv2.imread("assets/example_image.png")
image=processor.process_raw_image(image)
coord=tactip.predict_points(image) #preduct points

plt.imshow(image)
plt.scatter(coord[:,0],coord[:,1])
plt.axis("off")
plt.tight_layout("assets/predictions.pdf")
plt.savefig()
plt.show()