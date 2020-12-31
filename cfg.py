
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ad_path',type=str,default='/data/zjm/AD/ADnii')
parser.add_argument('--cm_path',type=str,default='/data/zjm/AD/CNnii')
parser.add_argument('--data_file',type=str,default='niiAD_path_label.txt')
parser.add_argument('--example_path',type=str,default='/data/zjm/AD/AD/006_S_4153/T1/ADNI2_Y1_ADNI_006_S_4153_MR_MPRAGE_br_raw_20120918161251818_97_S167974_I335308.nii')
parser.add_argument('--num_class',type=int,default=2)
parser.add_argument('--epochs',type=int,default=100)
parser.add_argument('--epoch_interval',type=int,default=1)
parser.add_argument('--bs',type=int,default=5)
parser.add_argument('--lr',type=float,default=1e-5)
parser.add_argument('--pretrain_path',type=str,default='/data/zjm/AD/pretrain_model/pretrain/resnet_10.pth')
parser.add_argument('--model_depth',type=int,default=10)
parser.add_argument('--resnet_shortcut',type=str,default='A')
parser.add_argument('--input_D',type=int,default=75)
parser.add_argument('--input_H',type=int,default=95)
parser.add_argument('--input_W',type=int,default=79)
parser.add_argument('--no_cuda',type=bool,default=False)
parser.add_argument('--gpu_id',type=list,default=[3])
parser.add_argument('--new_layer_names',type=str,default='fc') #在resnet10中删了conv-seg新增了fc层
parser.add_argument('--model',type=str,default='resnet')
parser.add_argument('--savepath',type=str,default='/data/zjm/AD/checkpoint/test_3dmodel.pth')
parser.add_argument('--save_train_model',type=str,default='/data/zjm/AD/checkpoint/train_3dmodel.pth')
parser.add_argument('--resume',type=bool,default=False)

config = parser.parse_args()