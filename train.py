from tqdm import tqdm
import torch.nn as nn
from cfg import config
from model import Base3DModel
import torch
from torch import optim
from dataset import AD_data
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score,accuracy_score
import os
from gradcam import GradCam,GuidedBackpropReLUModel,show_3d_cam_on_image
import nibabel as nib
import numpy as np
from skimage import transform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"



if __name__ == '__main__':
    train_data = AD_data(mode='train')
    test_data = AD_data(mode='test')

    train_load = DataLoader(dataset=train_data,batch_size=config.bs,shuffle=True)
    test_load = DataLoader(dataset=test_data,batch_size=2,shuffle=False)

    model = Base3DModel(config.num_class).to(device)
    optimize = optim.Adam(model.parameters(),lr=1e-4)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimize, gamma=0.9)

    one_hot = lambda x: torch.eye(config.num_class)[x,:].long()
    criterion = nn.CrossEntropyLoss()
    #criterion = lambda x, y: (((x - y) ** 2).sum(1).sqrt()).mean()

    min_f1 = 0
    train_min_f1 = 0
    for epoch in range(config.epochs):
        print('epoch:{}'.format(epoch))
        bar = tqdm(train_load)
        train_true_lab = []
        train_pre_lab = []
        for (image,label) in bar:
            image = image.to(device)
            label = label.to(device) # [5,1,79,95,79]
            optimize.zero_grad()

            pre_label = model(image)
            loss = criterion(pre_label,label) # pre是二维，而label是一维即可

            loss.backward()
            optimize.step()

            bar.set_description(str(loss.item()))
            _, class_lab = torch.max(pre_label, dim=1)

            train_true_lab.extend(label.cpu().detach().numpy().tolist())
            train_pre_lab.extend(class_lab.cpu().detach().numpy().tolist())
        #scheduler.step()

        acc = accuracy_score(train_true_lab,train_pre_lab)
        train_f1 = f1_score(train_true_lab, train_pre_lab)
        if train_f1 > train_min_f1:
            print('save train model..')
            checkpointer = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optim': optimize.state_dict()
            }
            torch.save(checkpointer, 'base_train_3dmodel.pth')
        train_min_f1 = train_f1


        # valid
        test_bar = tqdm(test_load)
        test_true_labels = []
        test_pre_labels = []
        for (img,label) in test_bar:
            model.eval()
            img = img.to(device)
            label = label.to(device)
            pre_lab = model(img)
            _,class_lab = torch.max(pre_lab,dim=1)

            test_true_labels.extend(label.cpu().detach().numpy().tolist())
            test_pre_labels.extend(class_lab.cpu().detach().numpy().tolist())

        print(test_true_labels)
        print(test_pre_labels)

        f1 = f1_score(test_true_labels,test_pre_labels)

        print('epoch:{}, loss:{},train_accu:{},train_f1:{},test_f1:{}'.format(epoch,loss.item(),acc,train_f1,f1))

        if f1 > min_f1:
            config.savepath = 'base_test_3dmodel.pth'
            print('save model..')
            checkpointer = {
                'epoch':epoch,
                'model':model.state_dict(),
                'optim':optimize.state_dict()
            }
            torch.save(checkpointer,config.savepath)
        min_f1 = f1

        if (epoch+1) % 10 ==0:
            print('save interval model..')
            config.savepath = 'base_interval_3dmodel.pth'
            checkpointer = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optim': optimize.state_dict()
            }
            torch.save(checkpointer, config.savepath)


        ##grad cam
        # grad_cam = GradCam(model=model,feature_module=model.conv,\
        #         #                  target_layer_names=["conv13"],use_cuda=True) # 卷积层的最后一层卷积
        #         #
        #         # img_arr = nib.load(config.example_path).get_data()
        #         # img_arr = torch.tensor(img_arr.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device) #[1,1,75,95,79]]
        #         #
        #         # target_index=None
        #         # mask = grad_cam(img_arr,target_index)
        #         #
        #         # img_arr = img_arr.squeeze()
        #         # mask = mask.squeeze()
        #         # img_arr = img_arr.cpu().data.numpy()
        #         #
        #         # show_3d_cam_on_image(img_arr,mask)

        












