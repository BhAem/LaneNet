from torch.utils.data import Dataset
import os
from torchvision import transforms
import cv2
import torch
import numpy as np

class TusimpleData(Dataset):
    def __init__(self,root_dir,rescale=(512,256)):
        super(TusimpleData,self).__init__()
        self.root_dir=root_dir
        self.rescale=rescale
        file_names=os.listdir(os.path.join(self.root_dir,'gt_image'))
        name_map={}
        for idx,file_name in enumerate(file_names):
            name_map[idx]=file_name.split('.')[0]
        self.name_map=name_map

    def __len__(self):
        return len(os.listdir(os.path.join(self.root_dir,'gt_image')))
    
    def __getitem__(self,index):
        lane_image=cv2.imread(os.path.join(self.root_dir,'gt_image',self.name_map[index]+'.jpg'),cv2.IMREAD_UNCHANGED)
        binary_label=cv2.imread(os.path.join(self.root_dir,'gt_binary_image',self.name_map[index]+'.png'),cv2.IMREAD_UNCHANGED)
        instance_label=cv2.imread(os.path.join(self.root_dir,'gt_instance_image',self.name_map[index]+'.png'),cv2.IMREAD_UNCHANGED)
        lane_image=cv2.cvtColor(lane_image,cv2.COLOR_BGR2RGB)

        if self.rescale:
            lane_image=cv2.resize(lane_image,self.rescale,cv2.INTER_CUBIC)
       
        binary_label=binary_label/255
        lane_image=np.transpose(lane_image,(2,0,1))

        lane_image=torch.tensor(lane_image,dtype=torch.float)/255
        instance_label=torch.tensor(instance_label,dtype=torch.float)
        return lane_image,binary_label,instance_label

'''
For future new datasets
'''
class NewData(Dataset):
    pass







