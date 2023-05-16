import torch
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
corp_size=256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings("ignore")

from timmML_025.models.factory import create_model

model = create_model('efficientnet_lite0') #this is modified lite0.
PATH_model="student_025.pt"

model.load_state_dict(torch.load(PATH_model))
model.eval()
img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN_STD[0],MEAN_STD[1])
    ])
    

path = 'path_to_dataset/jhu_crowd_v2.0/test'

targetfile_path=path+"/image_labels.txt"
pd_target=pd.read_csv(targetfile_path,header=None,dtype=str)

diffs=[]
i=0
for file_id,count,scene,weather,dis in zip(pd_target[0],pd_target[1],pd_target[2],pd_target[3],pd_target[4]):
    count=float(count)
    filename = path+"/images/"+file_id+'.jpg'
    img = Image.open(filename).convert('RGB')
    
    c=0
    c_size=256#240 for lite1, 256 for lite2
    
    if img.size[0]<c_size*2 and c==0:
        img=img.resize((c_size*2,int(c_size*2*img.size[1]/img.size[0])))
        c=1
    if img.size[1]<c_size*2 and c==0:
        img=img.resize((int(c_size*2*img.size[0]/img.size[1]),c_size*2))
        c=1
    
    kk=14 #14 is the bestor
    if img.size[0]>c_size*kk and c==0:
        img=img.resize((c_size*kk,int(c_size*kk*img.size[1]/img.size[0])))
        c=1
    if img.size[1]>c_size*kk and c==0:
        img=img.resize((int(c_size*kk*img.size[0]/img.size[1]),c_size*kk))
        #c=1  
    
    input_tensor = img_transform(img)
    input_batch = input_tensor.unsqueeze(0) 
    input_batch = input_batch.to('cuda')
    model.to('cuda')
    
    with torch.no_grad():
        output,features = model(input_batch)
    
    pred_count= torch.sum(output).item()
   # print(pred_count)
    diffs.append([file_id,count,scene,weather,dis,abs(pred_count-count)])
    print(str(file_id),count,pred_count,img.size)


results=pd.DataFrame(diffs)
results.columns = ["id", "counting", "scene", "weather","dis","error"]
print("result of ICOS 025")
errors_all=np.array(results["error"])
mse_all = np.sqrt(np.mean(np.square(errors_all)))
mae_all = np.mean(np.abs(errors_all))
print("mae_all:",mae_all,"mse_all:",mse_all)

errors_low=np.array(results[results['counting']<=50]["error"])
errors_mid=np.array(results[(results['counting']>50) & (results['counting']<=500)]["error"])
errors_high=np.array(results[results['counting']>500]["error"])

mse_low = np.sqrt(np.mean(np.square(errors_low)))
mae_low = np.mean(np.abs(errors_low))
print("mae_low:",mae_low,"mse_low:",mse_low)

mse_mid = np.sqrt(np.mean(np.square(errors_mid)))
mae_mid = np.mean(np.abs(errors_mid))
print("mae_mid:",mae_mid,"mse_mid:",mse_mid)

mse_high = np.sqrt(np.mean(np.square(errors_high)))
mae_high = np.mean(np.abs(errors_high))
print("mae_high:",mae_high,"mse_high:",mse_high)

errors_weather=np.array(results[results['weather']!='0']["error"])
mse_weather = np.sqrt(np.mean(np.square(errors_weather)))
mae_weather = np.mean(np.abs(errors_weather))
print("mae_weather:",mae_weather,"mse_weather:",mse_weather)
print(PATH_model,"_total_parameter:", sum(p.numel() for p in model.parameters()))
