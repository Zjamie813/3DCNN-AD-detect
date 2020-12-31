import random
import torch
import SimpleITK as sitk
from nipype.interfaces.ants import N4BiasFieldCorrection
from cfg import config
import os
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
from matplotlib import pylab as plt
from skimage import transform
from torchvision.transforms import transforms
from nibabel.viewers import OrthoSlicer3D

def gen_data(path,label):
    '''

    :param path: ad or cm path
    :return: a file with datapath,label,83
    '''
    exit_Y1 = False
    with open(config.data_file, 'a', encoding='utf-8') as f:
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            f.write(file_path + ',' + str(label) + '\n')
            # nii_file_path = os.path.join(file_path, 'T1')
            # for nii_file in os.listdir(nii_file_path):
            #     if 'Y1' in nii_file:
            #         exit_Y1 = True
            #         nii_file_name = os.path.join(nii_file_path, nii_file)
            #         f.write(nii_file_name + ',' + str(label)+'\n')
            # if exit_Y1 == False:
            #     print(nii_file_path)

class AD_data(Dataset):
    def __init__(self,mode='train'):
        super(AD_data).__init__()
        self.filenames,self.labels = self.get_fn_label()
        ran_list = list(zip(self.filenames,self.labels)) #打包
        random.shuffle(ran_list)
        self.filenames,self.labels = zip(*ran_list) #解压
        train_len = int(len(self.labels) * 0.8)
        if mode=='train':
            self.filenames = self.filenames[:train_len]
            self.labels = self.labels[:train_len]
        else:
            self.filenames = self.filenames[train_len:]
            self.labels = self.labels[train_len:]

    def normalize_data(self,data, mean, std):
        # data:[1,144,144,144]
        data -= mean[:, np.newaxis, np.newaxis, np.newaxis]
        data /= std[:, np.newaxis, np.newaxis, np.newaxis]
        return data


    def get_fn_label(self):
        fns = []
        lbs = []
        with open(config.data_file,'r',encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                fn, label = line.split(',')
                fns.append(fn)
                lbs.append(int(label))
        return fns,lbs



    def __len__(self):
        return len(self.labels)



    def __getitem__(self, index):
        fpath = self.filenames[index]
        lab = self.labels[index]
        #print(torch.tensor(nib.load(fpath).get_data().astype(np.float32)).shape)
        img_arr = torch.tensor(nib.load(fpath).get_data().astype(np.float32)).permute(3,2,1,0) # [[1, 256, 240, 176]]

        img_arr = transform.resize(img_arr,(1,75,95,79))
        mean = img_arr.mean(axis=(1, 2, 3))
        std = img_arr.std(axis=(1, 2, 3))
        img_norm_arr = self.normalize_data(img_arr, mean, std)

        return img_norm_arr,lab


if __name__ == '__main__':
    ad_path = config.ad_path
    cm_path = config.cm_path

    gen_data(ad_path,1)
    gen_data(cm_path,0)

