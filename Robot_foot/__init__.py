"""
Code for different feet (electrical or optical tactile sensors).
This code provides an interface with the pretrained models, as well as the option to further train the models

Code by Dexter R. Shepherd, PhD student in Artificial intellgience at the University of Sussex
https://www.linkedin.com/in/dexter-shepherd-1a4a991b8/
https://github.com/shepai

"""
import numpy as np
import cv2
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
base_path = os.path.dirname(__file__)  # path of the current .py file
model_dir = os.path.join(base_path, 'models')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    

class dataPreprocessor:
    """
    Class to take in a dataset and try make it look more like the existing data that the odel was trained on
    You could do this yourself but this feels neater and a bit more 'idiot-proof'
    """
    def __init__(self):
        self.h=110
        self.w=120
    def process_raw_image(self,image): #takes in the image you wish to process
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
    def process_data(self,dataset): #takes in the dataset you wish to process
        X=[]
        for i in range(len(dataset)):
            t=self.process_video(dataset[i])
            X.append(t)
        return np.array(X)
    def process_video(self,video): #takes in the frames of images
        t=[]
        for t in range(len(video)):
            t.append(self.process_raw_image(video[t]))
        return np.array(video)

class opticalSensor:
    def __init__(self): #load in the model
        self.model = SimpleLSTM(110*120,350,15,3).to(device)
        self.model.load_state_dict(torch.load(os.path.join(model_dir,"models/mymodel_lstm_augment")))
        self.friction=joblib.load(os.path.join(model_dir,'models/random_forest_model.pkl'))
    def predict_texture(self,images):
        if type(images)!=type(torch.tensor([])): #ensure that the data is correct format
            images=torch.tensor(images).to(device)
        images=images.reshape((1,len(images),110*120))
        return self.model(images).cpu().detach().numpy()[0]
    def predict_friction(self,images):
        if type(images)==type(torch.tensor([])): #ensure that the data is correct format
            images=images.cpu().detach().numpy()
        images=images.reshape((1,len(images)*110*120))
        return self.friction.predict(images)

class PressTipSensor:
    def __init__(self): #load in the model
        self.model = joblib.load(os.path.join(model_dir, 'random_forest_classifier_elec.pkl'))
        self.friction = joblib.load(os.path.join(model_dir, 'random_forest_model.pkl'))
    def predict_texture(self,data):
        if type(data)==type(torch.tensor([])): #ensure that the data is correct format
            data=data.cpu().detach().numpy()
        data=data.reshape((1,-1))
        return self.model.predict(data)[0]
    def predict_friction(self,data):
        if type(data)==type(torch.tensor([])): #ensure that the data is correct format
            data=data.cpu().detach().numpy()
        data=data.reshape((1,-1))
        return self.friction.predict(data)[0]

if __name__=="__main__":
    presstip = PressTipSensor()
    data=np.random.random((1,10,4))
    print("Friction:",presstip.predict_friction(data))
    print("Classification:",presstip.predict_texture(data))