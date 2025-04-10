"""
Code for different feet (electrical or optical tactile sensors).
This codwe provides an interface with the pretrained models, as well as the option to further train the models

Code by Dexter R. Shepherd, PhD student in Artificial intellgience at the University of Sussex
https://www.linkedin.com/in/dexter-shepherd-1a4a991b8/
https://github.com/shepai

"""
import numpy as np
import cv2

class dataPreprocessor:
    """
    Class to take in a dataset and try make it look more like the existing data that the odel was trained on
    """
    def __init__(self):
        self.h=
        self.w=
    def process_raw_image(self,image):
        image=cv2.resize(image,(self.h,self.w),interpolation=cv2.INTER_AREA) #resize 
        #apply Sobel filter in x-direction
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  #ksize=3 for a 3x3 Sobel kernel
        #apply Sobel filter in y-direction
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        #convert the results back to uint8
        sobel_x = np.uint8(np.absolute(sobel_x))
        sobel_y = np.uint8(np.absolute(sobel_y))
        #combine the results to get the final edge-detected image
        sobel_combined = cv2.bitwise_or(sobel_x, sobel_y)
        return sobel_combined
    def process_data(self,dataset):
        X=[]
        for i in range(len(dataset)):
            t=[]
            for t in range(dataset[0]):
                t.append(self.process_raw_image(dataset[i][t]))
            X.append(t)
        return np.array(X)

class opticalSensor:
    def __init__(self):
        pass


class TactileSensor:
    def __init__(self):
        pass