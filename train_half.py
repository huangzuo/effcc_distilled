print("start...")
from PIL import Image,ImageOps
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as VF
from torchvision import transforms
import random
import numpy as np
import pandas as pd
np.seterr(divide='ignore', invalid='ignore')

print("loading classes")
def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w


def cal_innner_area(c_left, c_up, c_right, c_down, bbox):
    inner_left = np.maximum(c_left, bbox[:, 0])
    inner_up = np.maximum(c_up, bbox[:, 1])
    inner_right = np.minimum(c_right, bbox[:, 2])
    inner_down = np.minimum(c_down, bbox[:, 3])
    inner_area = np.maximum(inner_right-inner_left, 0.0) * np.maximum(inner_down-inner_up, 0.0)
    return inner_area

def cal_box_area(bbox):
    box_area = np.maximum(bbox[:, 2]-bbox[:, 0], 0.0) * np.maximum(bbox[:, 3]- bbox[:, 1], 0.0)
    return box_area

class Crowd(data.Dataset):
    def __init__(self, root_path, crop_size,
                 downsample_ratio, is_gray=False,
                 method='train'):

        self.root_path = root_path
        targetfile_path=self.root_path+"/image_labels.txt"
        self.pd_target=pd.read_csv(targetfile_path,header=None,dtype=str)
        
        self.im_list = list(self.pd_target[0])
        if method not in ['train', 'val']:
            raise Exception("not implement")
        self.method = method

        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio

        if is_gray:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.root_path+"/images/"+self.im_list[item]+".jpg"
        gt_path = img_path.replace('jpg', 'txt').replace("images","gt")
        img = Image.open(img_path).convert('RGB')
        if self.method == 'train':
            wd, ht = img.size
            gt=pd.read_csv(gt_path,delim_whitespace=True,header=None).to_numpy()
            points=[]
            bboxs=[]
            if len(gt)>0:
                for p in gt:
                    if p[0]<0 or p[1]<0 or p[2]<0 or p[3]<0 or p[4]<0 or p[5]<0:
                        continue
                    points.append([p[0],p[1]]) #convert (col,row) to (row,col)
                    left_up=[max(p[0]-int(p[2]/2),0),max(p[1]-int(p[3]/2),0)]
                    right_down=[min(p[0]+int(p[2]/2),wd),min(p[1]+int(p[3]/2),ht)]
                    bboxs.append(left_up+right_down)
            keypoints = np.array(points)
            #print("k:",keypoints)
            bboxs = np.array(bboxs)
            #print("bbox",bboxs)
            return self.train_transform(img, keypoints,bboxs)



    def train_transform(self, img, keypoints,bbox):
   # def train_transform(self, img,den, keypoints,bbox):
        """random crop image patch and find people in it"""
        wd, ht = img.size
        st_size = min(wd, ht)

        if st_size < self.c_size:
            delta_w = self.c_size - wd
            delta_h = self.c_size - ht
            padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
            img = ImageOps.expand(img, padding)
            wd, ht = img.size
        
        if len(keypoints)==0:
            #no head in the data
            i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
            img = VF.crop(img, i, j, h, w)
          #  dens = VF.crop(dens, i, j, h, w)
            keypoints=np.array([])
            bbox=np.array([])
            target=np.array([])
            return self.trans(img), torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(bbox.copy()).float(),\
               torch.from_numpy(target.copy()).float(), st_size
        
       
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = VF.crop(img, i, j, h, w)
        

        inner_area = cal_innner_area(j, i, j+w, i+h, bbox)

        origin_area = cal_box_area(bbox)
        ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
        mask = (ratio >= 0.3)

        target = ratio[mask]
        keypoints = keypoints[mask]
        bbox=(bbox[mask]-[j,i,j,i]).clip(min=0,max=self.c_size)
        keypoints = keypoints[:, :2] - [j, i]  # change coodinate
        #print(keypoints)
        #print(bbox)
        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(bbox.copy()).float(),\
               torch.from_numpy(target.copy()).float(), st_size
    
#------------
print("loading bayeslian loss")
from torch.nn import Module
class Post_Prob(Module):
    def __init__(self, sigma, c_size, stride, background_ratio, use_background, device):
        super(Post_Prob, self).__init__()
        assert c_size % stride == 0

        self.sigma = sigma
        self.bg_ratio = background_ratio
        self.device = device
        # coordinate is same to image space, set to constant since crop size is same
        self.cood = torch.arange(0, c_size, step=stride,
                                 dtype=torch.float32, device=device) + stride / 2
        self.cood.unsqueeze_(0)
        self.softmax = torch.nn.Softmax(dim=0)
        self.use_bg = use_background

    def forward(self, points, st_sizes):
        num_points_per_image = [len(points_per_image) for points_per_image in points]
        all_points = torch.cat(points, dim=0)

        if len(all_points) > 0:
            x = all_points[:, 0].unsqueeze_(1)
            y = all_points[:, 1].unsqueeze_(1)
            x_dis = -2 * torch.matmul(x, self.cood) + x * x + self.cood * self.cood
            y_dis = -2 * torch.matmul(y, self.cood) + y * y + self.cood * self.cood
            y_dis.unsqueeze_(2)
            x_dis.unsqueeze_(1)
            dis = y_dis + x_dis
            dis = dis.view((dis.size(0), -1))

            dis_list = torch.split(dis, num_points_per_image)
            prob_list = []
            for dis, st_size in zip(dis_list, st_sizes):
                if len(dis) > 0:
                    if self.use_bg:
                        min_dis = torch.clamp(torch.min(dis, dim=0, keepdim=True)[0], min=0.0)
                        d = st_size * self.bg_ratio
                        bg_dis = (d - torch.sqrt(min_dis))**2
                        dis = torch.cat([dis, bg_dis], 0)  # concatenate background distance to the last
                    dis = -dis / (2.0 * self.sigma ** 2)
                    prob = self.softmax(dis)
                else:
                    prob = None
                prob_list.append(prob)
        else:
            prob_list = []
            for _ in range(len(points)):
                prob_list.append(None)
        return prob_list
#-------------------------
class Bay_Loss(Module):
    def __init__(self, use_background, device):
        super(Bay_Loss, self).__init__()
        self.device = device
        self.use_bg = use_background

    def forward(self, prob_list, target_list, pre_density):
        loss = 0
        for idx, prob in enumerate(prob_list):  # iterative through each sample
            if prob is None:  # image contains no annotation points
                pre_count = torch.sum(pre_density[idx])
                target = torch.zeros((1,), dtype=torch.float32, device=self.device)
            else:
                N = len(prob)
                if self.use_bg:
                    target = torch.zeros((N,), dtype=torch.float32, device=self.device)
                    target[:-1] = target_list[idx]
                else:
                    target = target_list[idx]
                pre_count = torch.sum(pre_density[idx].view((1, -1)) * prob, dim=1)  # flatten into vector

            loss += torch.sum(torch.abs(target - pre_count))
        loss = loss / len(prob_list)
        return loss

#-------------
print("loading avemeter")
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
#####################
import torch.nn as nn
import time
import torch.optim as optim
print("loading trainer")
def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]# the number of points is not fixed, keep it as a list of tensor
    bboxs=transposed_batch[2]
    targets = transposed_batch[3]
    st_sizes = torch.FloatTensor(transposed_batch[4])
    return images, points, bboxs,targets, st_sizes


import torch.nn.functional as F


def cal_dense_msm(features,weight): #dense fsp function
        msm = []
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                x = features[i]
                y = features[j]
                msm.append(cal_msm(x,y,weight))
        #print(msm)
        return msm

def cal_msm(x,y,weight):
    x = torch.flatten(x,start_dim=1)
    y = torch.flatten(y,start_dim=1)
    w_sum=torch.mul(x,y)#[16,1112121]
    w_sum=torch.sum(w_sum,1)#[16,]
    w_norm =torch.mul(torch.norm(x,dim=1),torch.norm(y,dim=1))
    cos=w_sum/w_norm
    ang=torch.acos(cos)*180/3.14159
    ang=torch.mul(weight,ang)
    return ang
########################
class Trainer():
    def __init__(self,teacher,student, train_loader,val_path,criterion, optimizer,device,scale=1,
                   lr=1e-5,max_epoch=300,path="test.pt"):
        self.device=device
        self.post_prob = Post_Prob(8,#sigma
                                   256,#crop
                                   8,#args.downsample_ratio,
                                   0.1,#args.background_ratio,
                                   True,#
                                   self.device)
        self.loss = Bay_Loss(True, self.device)
        self.best_mae = 200
        self.best_mse = 500
        self.best_count = 0
        self.student = student.to(device)
        self.teacher = teacher.to(device)
        self.train_loader = train_loader
        self.val_path = val_path
        self.mae_list=[]
        self.mse_list=[]
        self.loss_list=[]
        self.max_epoch=max_epoch
        self.path=path
        self.scale=scale
        self.criterion=criterion
        self.optimizer=optimizer
        self.teacher_loss=Bay_Loss(True, self.device)

    
    def forward(self):
        for epoch in range(0,self.max_epoch):
            self.epoch = epoch            
            # training    
            self.train()

    def train(self): # training for all datasets
        self.teacher.eval()
        self.student.train()

        total_loss=0

        losses_h = AverageMeter()
        losses_s = AverageMeter()
        losses_msm_dense = AverageMeter()
        losses_msm = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()
        pre_count=0
        gt_count=0

        for i_batch, sample_batched in enumerate(self.train_loader):
            img,points,bboxs,targets,st_sizes=sample_batched
            points = [p.to(self.device) for p in points]
            bboxs = [b.to(self.device) for b in bboxs]
            targets = [t.to(self.device) for t in targets]
            targets_count=torch.stack([torch.sum(t) for t in targets],dim=0)
            st_sizes = st_sizes.to(self.device)
            img = img.to(self.device,dtype=torch.float)
            gt_count = gt_count+np.sum(np.array([len(p) for p in points], dtype=np.float32))

            with torch.no_grad():
                teacher_output,teacher_features = self.teacher(img)
                teacher_count=torch.sum(teacher_output,(3,2,1))
                teacher_loss=torch.abs(teacher_count-targets_count)
                #print("teacher_loss",teacher_loss)
            with torch.set_grad_enabled(True):

                student_output,student_features = self.student(img)
                
                student_count=torch.sum(student_output,(3,2,1))
                student_loss=torch.abs(student_count.detach()-targets_count)
                outlier=student_loss-teacher_loss #nagive means out-lier.
                outlier=torch.where(outlier<0,torch.tensor(0.).cuda(),torch.tensor(1.).cuda())
                #outlier=F.softmax(outlier)
                #student_features=[torch.mul(outlier,f) for f in student_features]
                #teacher_features=[torch.mul(outlier,f) for f in teacher_features]                
                #print(teacher_features[0])
                
                prob_list = self.post_prob(points,st_sizes)
                loss_h = self.loss(prob_list, targets, student_output)
                
                teacher_msm = cal_dense_msm(teacher_features,outlier)
                student_msm = cal_dense_msm(student_features,outlier)

                #student_output=torch.mul(outlier,student_output)
                #teacher_output=torch.mul(outlier,teacher_output)
                #print(teacher_output)
 
                loss_s = torch.sum(criterion(student_output, teacher_output),dim=(1,2,3))
                loss_s = torch.mul(outlier,loss_s)
                loss_s=torch.mean(loss_s)
                loss_msm_dense = torch.tensor([0.], dtype=torch.float).cuda()

                loss_f = []
                assert len(teacher_msm) == len(student_msm)
                for t in range(len(teacher_msm)):
                    loss_f.append(criterion(teacher_msm[t], student_msm[t]))
                loss_msm_dense = torch.mean(sum(loss_f))
  
                
                loss_msm = torch.tensor([0.], dtype=torch.float).cuda()
                loss_c = []
                for t in range(len(student_features) - 1):
                    loss_c.append(torch.sum(cal_msm(student_features[t], teacher_features[t],outlier)))
                loss_msm = sum(loss_c)/len(loss_c)
                
                loss=loss_h + loss_s + loss_msm_dense + loss_msm
                losses_h.update(loss_h.item(), img.size(0))
                losses_s.update(loss_s.item(), img.size(0))
                losses_msm_dense.update(loss_msm_dense.item(), img.size(0))
                losses_msm.update(loss_msm.item(), img.size(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_time.update(time.time() - end)
                end = time.time()
                if i_batch % 100 == (100 - 1):
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.avg:.3f}  '
                          'Data {data_time.avg:.3f}  '
                          'Loss_h {loss_h.avg:.4f}  '
                          'Loss_s {loss_s.avg:.4f}  '
                          'Loss_msm_dense {loss_msm_dense.avg:.4f}  '
                          'Loss_msm_inter {loss_msm.avg:.4f}  '
                        .format(
                        self.epoch, i_batch, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss_h=losses_h, loss_s=losses_s,
                        loss_msm_dense=losses_msm_dense, loss_msm=losses_msm))


            pre_count=pre_count+torch.sum(student_output).item()
            total_loss=total_loss+loss.item()

        print(self.epoch,":OUT_no weight half full trained:",total_loss,pre_count,gt_count)
        torch.save(self.student.state_dict(), self.path)
        #validation
        
        self.student.eval()
        targetfile_path=self.val_path+"/image_labels.txt"
        pd_target=pd.read_csv(targetfile_path,header=None,dtype=str)
        diffs=[]
        i=0
        for file_id,count,scene,weather,dis in zip(pd_target[0],pd_target[1],pd_target[2],pd_target[3],pd_target[4]):
            count=float(count)
            filename = self.val_path+"/images/"+file_id+'.jpg'
            img = Image.open(filename).convert('RGB')

            c=0
            c_size=256#240 for lite1, 256 for lite2

            if img.size[0]<c_size*2 and c==0:
                img=img.resize((c_size*2,int(c_size*2*img.size[1]/img.size[0])))
                c=1
            if img.size[1]<c_size*2 and c==0:
                img=img.resize((int(c_size*2*img.size[0]/img.size[1]),c_size*2))
                c=1

            kk=12 #12 is the best for lite0 rightnow
            if img.size[0]>c_size*kk and c==0:
                img=img.resize((c_size*kk,int(c_size*kk*img.size[1]/img.size[0])))
                c=1
            if img.size[1]>c_size*kk and c==0:
                img=img.resize((int(c_size*kk*img.size[0]/img.size[1]),c_size*kk))
                #c=1

            input_tensor = img_transform(img)
            input_batch = input_tensor.unsqueeze(0)
            input_batch = input_batch.to('cuda')
            with torch.no_grad():
                output,features = self.student(input_batch)

            pred_count= torch.sum(output).item()
            diffs.append(abs(pred_count-count))

        diffs = np.array(diffs)

        mse = np.sqrt(np.mean(np.square(diffs)))
        mae = np.mean(np.abs(diffs))
       
        log_str = ':epoch test: mae {}, mse {}'.format(mae, mse)

        print(self.epoch,log_str)
        self.mae_list.append(mae)
        self.mse_list.append(mse)
        if mse<self.best_mse:
            self.best_mse=mse
            print("get a better model, saving....")
            torch.save(self.student.state_dict(), "student_out2_half_full_best.pt")
            
#############################
print("starting")
from torch.utils.data import Dataset, DataLoader

train_path = 'path_to_dataset/jhu_crowd_v2.0/train'
val_path = 'path_to_dataset/jhu_crowd_v2.0/val'
PATH_model="student_out2_half_full.pt"

train_set = Crowd(train_path,256, 8)  
#val_set = Crowd(val_path,256, 8)   
train_loader=DataLoader(train_set,collate_fn=train_collate,batch_size=16,shuffle=True,num_workers=4)
#val_loader=DataLoader(val_set,collate_fn=train_collate,batch_size=1,shuffle=True,num_workers=1)

img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

print(len(train_loader))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
from timmML2.models.factory import create_model
model_teacher = create_model('efficientnet_lite2')
PATH_model_teacher="teacher.pt"
model_teacher.load_state_dict(torch.load(PATH_model_teacher))
print("teacher2 loaded")

from timmML_half.models.factory import create_model
model_student = create_model('efficientnet_lite0')
#model_student.load_state_dict(torch.load(PATH_model))
print("student loaded")
criterion = nn.MSELoss(reduction='none').cuda()
optimizer = torch.optim.Adam(model_student.parameters(), lr=1e-5, weight_decay=1e-4)



cc_trainer = Trainer(teacher=model_teacher,student=model_student,
                     train_loader=train_loader,val_path=val_path,
                     criterion=criterion,optimizer=optimizer,
                     device=device,lr=1e-5,scale=1,path=PATH_model)
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
cc_trainer.forward()
