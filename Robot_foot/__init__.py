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

######################
#variabes
######################

keys=['Leather', 'Cork', 'wool', 'LacedMatt', 'Gfoam', 'Plastic', 'Carpet', 'bubble', 'Efoam', 'cotton', 'LongCarpet', 'Flat', 'felt', 'Jeans', 'Ffoam']

######################
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
    

class ImageDataPreprocessor:
    """
    Class to take in a dataset and try make it look more like the existing data that the odel was trained on
    You could do this yourself but this feels neater and a bit more 'idiot-proof'
    """
    def __init__(self):
        self.h=110
        self.w=120
    def process_raw_image(self,image): #takes in the image you wish to process
        crop=[60,180,40,150]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        compression_dim=0.4
        new_dim=(int(640*compression_dim),int(480*compression_dim))#cv2.resize(image,(self.h,self.w),interpolation=cv2.INTER_AREA) #resize 
        image = cv2.resize(image,new_dim,interpolation=cv2.INTER_AREA)[crop[2]:crop[3],crop[0]:crop[1]]
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
        t_=[]
        for t in range(len(video)):
            t_.append(self.process_raw_image(video[t]))
        return np.array(t_)

class opticalSensor:
    def __init__(self): #load in the model
        self.model = SimpleLSTM(110*120,350,15,3).to(device)
        self.model.load_state_dict(torch.load(os.path.join(model_dir,"mymodel_lstm_augment")))
        self.friction=joblib.load(os.path.join(model_dir,'random_forest_model_optical.pkl'))
    def predict_texture(self,images):
        if type(images)!=type(torch.tensor([])): #ensure that the data is correct format
            images=torch.tensor(images).to(device)
        images=images.reshape((1,images.shape[0],110*120)).to(torch.float32)
        return np.argmax(self.model(images).cpu().detach().numpy()[0])
    def predict_texture_multi(self,images):
        if type(images)!=type(torch.tensor([])): #ensure that the data is correct format
            images=torch.tensor(images).to(device)
        images=images.reshape((images.shape[0],images.shape[1],110*120)).to(torch.float32)
        return np.argmax(self.model(images).cpu().detach().numpy(),axis=1)
    def predict_friction(self,images):
        if type(images)==type(torch.tensor([])): #ensure that the data is correct format
            images=images.cpu().detach().numpy()
        images=images.reshape((1,len(images[0])*110*120))
        return self.friction.predict(images)[0]

class PressTipSensor:
    def __init__(self): #load in the model
        self.model = joblib.load(os.path.join(model_dir, 'random_forest_classifier_elec.pkl'))
        self.friction = joblib.load(os.path.join(model_dir, 'random_forest_model_electric_friction.pkl'))
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
    def predict_texture_multi(self,images):
        if type(data)==type(torch.tensor([])): #ensure that the data is correct format
            data=data.cpu().detach().numpy()
        data=data.reshape((1,-1))
        data=data.reshape((len(data),1)).to(torch.float32)
        return self.model.predict(data)

if __name__=="__main__":
    presstip = PressTipSensor()
    data=np.random.random((1,64))
    print("Friction:",presstip.predict_friction(data))
    data=np.random.random((1,10,4))
    print("Classification:",presstip.predict_texture(data))
