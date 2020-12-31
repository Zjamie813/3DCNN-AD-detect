from tqdm import tqdm
import torch.nn as nn
from cfg import config
from model import Base3DModel
import torch
from torch import optim
from dataset import AD_data
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
import os
from gradcam import GradCam, GuidedBackpropReLUModel, show_3d_cam_on_image
import nibabel as nib
import numpy as np
from medical_model import generate_model
from skimage import transform
from nilearn import plotting

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_VISIBLE_DEVICES"] = "3,2"


def normalize_data(data, mean, std):
    # data:[1,144,144,144]
    data -= mean[:, np.newaxis, np.newaxis, np.newaxis]
    data /= std[:, np.newaxis, np.newaxis, np.newaxis]
    return data


if __name__ == '__main__':
    # ex_nii = 'e2.nii.gz'
    # outfile = 'e2.png'
    # img_arr = nib.load(config.example_path).get_data().transpose(3,2,1,0)
    # img_arr = transform.resize(img_arr, (1, 75, 95, 79))
    # mean = img_arr.mean(axis=(1, 2, 3))
    # std = img_arr.std(axis=(1, 2, 3))
    # norm_data = normalize_data(img_arr, mean, std)[0]
    #
    # new_image = nib.Nifti1Image(norm_data, np.eye(4))
    # nib.save(new_image, ex_nii)
    # plotting.plot_stat_map(ex_nii, output_file=outfile)
    # plotting.show()

    config.savepath= 'AD_medical_checkpoint.pth'
    train_data = AD_data(mode='train')
    test_data = AD_data(mode='test')

    train_load = DataLoader(dataset=train_data, batch_size=config.bs, shuffle=True)
    test_load = DataLoader(dataset=test_data, batch_size=2, shuffle=False)

    model, parameters = generate_model(config)
    model.to(device)

    params = [
        {'params': parameters['base_parameters'], 'lr': config.lr},
        {'params': parameters['new_parameters'], 'lr': config.lr * 2.6}
    ]
    optimize = optim.Adam(params,weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimize, gamma=0.9)

    one_hot = lambda x: torch.eye(config.num_class)[x, :].long()
    criterion = nn.CrossEntropyLoss()
    #criterion = lambda x, y: (((x - y) ** 2).sum(1).sqrt()).mean()

    # train
    min_f1 = 0
    train_min_f1 = 0
    for epoch in range(config.epochs):
    #     print('epoch:{}'.format(epoch))
    #     bar = tqdm(train_load)
    #     train_true_lab = []
    #     train_pre_lab = []
    #     for (image, label) in bar:
    #         image = image.to(device)
    #         ini_label = label
    #         label = label.to(device)  # [5,1,79,95,79]
    #         #label = one_hot(label).to(device)
    #         optimize.zero_grad()
    #
    #         pre_label = model(image)
    #         loss = criterion(pre_label, label)  # pre是二维，而label是一维即可
    #
    #         loss.backward()
    #         optimize.step()
    #
    #         bar.set_description(str(loss.item()))
    #         _, class_lab = torch.max(pre_label, dim=1)
    #
    #         train_true_lab.extend(ini_label.cpu().detach().numpy().tolist())
    #         train_pre_lab.extend(class_lab.cpu().detach().numpy().tolist())
    #
    #
    #     scheduler.step()
    #
    #     acc = accuracy_score(train_true_lab, train_pre_lab)
    #     train_f1 = f1_score(train_true_lab, train_pre_lab)
    #     if train_f1 > train_min_f1:
    #         print('save train model..')
    #         checkpointer = {
    #             'epoch': epoch,
    #             'model': model.state_dict(),
    #             'optim': optimize.state_dict()
    #         }
    #         torch.save(checkpointer, config.save_train_model)
    #     train_min_f1 = train_f1
    #
    #     #valid
    #     test_bar = tqdm(test_load)
    #     test_true_labels = []
    #     test_pre_labels = []
    #     for (img, label) in test_bar:
    #         model.eval()
    #         img = img.to(device)
    #         test_ini_label = label
    #         label = label.to(device)
    #         label = one_hot(label).to(device)
    #         pre_lab = model(img)
    #         _, class_lab = torch.max(pre_lab, dim=1)
    #
    #         test_true_labels.extend(test_ini_label.cpu().detach().numpy().tolist())
    #         test_pre_labels.extend(class_lab.cpu().detach().numpy().tolist())
    #
    #
    #     print(test_true_labels)
    #     print(test_pre_labels)
    #
    #     f1 = f1_score(test_true_labels, test_pre_labels)
    #
    #     print('epoch:{}, loss:{},train_accu:{},train_f1:{},test_f1:{}'.format(epoch, loss.item(), acc, train_f1, f1))
    #
    #     if f1 > min_f1:
    #         print('save model..')
    #         checkpointer = {
    #             'epoch': epoch,
    #             'model': model.state_dict(),
    #             'optim': optimize.state_dict()
    #         }
    #         torch.save(checkpointer, config.savepath)
    #     min_f1 = f1
    #
    #     if (epoch+1) % config.epoch_interval:
    #         config.savepath = '/data/zjm/AD/checkpoint/ep{}_test_3dmodel.pth'.format(epoch+1)
    #         checkpointer = {
    #             'epoch': epoch,
    #             'model': model.state_dict(),
    #             'optim': optimize.state_dict()
    #         }
    #         torch.save(checkpointer, config.savepath)


        ##grad cam
        if (epoch+1) == config.epoch_interval:
            #print(model._modules.items())
            model = Base3DModel(config.num_class).to(device)
            config.save_train_model = '/data/zjm/AD/base_train_3dmodel.pth'
            # '/data/zjm/AD/base_train_3dmodel.pth'
            # '/data/zjm/AD/checkpoint/train_3dmodel.pth'
            if config.resume:
                print('=>load train model')
                ckp = torch.load(config.save_train_model)
                model.load_state_dict(ckp['model'])
                #optimizer.load_state_dict(ckp['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(config.save_train_model, ckp['epoch']))

            grad_cam = GradCam(model=model, feature_module=model.conv, \
                               target_layer_names=["conv9"], use_cuda=True)  # 卷积层的最后一层卷积
            #layer2,["0"]
            #conv,["conv9"]
            img_arr = nib.load(config.example_path).get_data().transpose(3,2,1,0)
            img_arr = transform.resize(img_arr, (1, 75, 95, 79))
            mean = img_arr.mean(axis=(1, 2, 3))
            std = img_arr.std(axis=(1, 2, 3))
            norm_data = normalize_data(img_arr, mean, std)

            tensor_data = torch.tensor(norm_data.astype(np.float32)).unsqueeze(0).to(device)

            #img_arr = transform.resize(nib.load(config.example_path).get_data(),(1,75,95,79))
            #img_arr = torch.tensor(img_arr.astype(np.float32)).unsqueeze(1).to(device)
            #img_arr = torch.tensor(img_arr.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,75,95,79]]

            target_index = None
            mask = grad_cam(tensor_data, target_index)

            img_3dim = tensor_data.squeeze()
            mask = mask.squeeze()
            img_3dim = img_3dim.cpu().data.numpy()

            show_3d_cam_on_image(img_3dim, mask)














