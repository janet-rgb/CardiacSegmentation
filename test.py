#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt
import cv2

from utils.dataset_synapse import CT_dataset, hist_filter
from utils.utils import test_single_volume

from lib.networks import MIST_CAM


# In[2]:


import gc
gc.collect()
torch.cuda.empty_cache()
os.environ['CUDA_VISIBLE_DEVICES']='0, 1'


# In[3]:



parser = argparse.ArgumentParser(description='Searching longest common substring. '
                    'Uses Ukkonen\'s suffix tree algorithm and generalized suffix tree. '
                    'Written by Ilya Stepanov (c) 2013')
parser.add_argument('strings', metavar='STRING', nargs='*', help='String for searching',)

parser.add_argument('--volume_path', type=str,
                    default='C:/Users/AIMEDIC/Desktop/LALV/rawdata_LA', help='root dir for validation volume data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')

parser.add_argument('--test_save_dir', type=str, default='predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0001, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=2222, help='random seed')


parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=10, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=3, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of network input') #224
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')

args = parser.parse_args("AAA".split())
# In[4]:

args

#if(args.num_classes == 14):
#    classes = ['spleen', 'right kidney', 'left kidney', 'gallbladder', 'esophagus', 'liver', 'stomach', 'aorta', 'inferior vena cava', 'portal vein and splenic vein', 'pancreas', 'right adrenal gland', 'left adrenal gland']
#else:
#    classes = ['spleen', 'right kidney', 'left kidney', 'gallbladder', 'pancreas', 'liver', 'stomach', 'aorta']
classes = ['LA', 'bg']
def inference(args, model, test_save_path=None): # INFERENCE(ARGS, NET, TEST_SAVE_PATH)
    db_test = args.Dataset(base_dir=args.volume_path, split="test", list_dir=args.list_dir, filter = hist_filter, nclass=args.num_classes)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1) # BATCH SIZE = 1
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    i = 0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)): # BATCH SIZE = 1
        #h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0] # IMAGE, LABEL OF (VOL SIZE, 512, 512) SHAPE
        print(i_batch, case_name)
        # full volume
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, save_img = True, case=case_name,z_spacing=1)
        #print(metric_i)
        metric_list += np.array(metric_i)
        #print(metric_list.shape)
        #print(metric_list[0])
        
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1], np.mean(metric_i, axis=0)[2], np.mean(metric_i, axis=0)[3]))
        #if i_batch % 50 == 0:
        for i in range(1,args.num_classes):
            logging.info('Mean class (%d) %s mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (i, classes[i-1], metric_list[i-1][0]/(i_batch+1), metric_list[i-1][1]/(i_batch+1), metric_list[i-1][2]/(i_batch+1), metric_list[i-1][3]/(i_batch+1)))

    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class (%d) %s mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (i, classes[i-1], metric_list[i-1][0], metric_list[i-1][1], metric_list[i-1][2], metric_list[i-1][3]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    mean_jacard = np.mean(metric_list, axis=0)[2]
    mean_asd = np.mean(metric_list, axis=0)[3]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f, mean_jacard : %f mean_asd : %f' % (performance, mean_hd95, mean_jacard, mean_asd))
    return "Testing Finished!"


# In[5]:


import time

if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = 'CHD' #'Synapse' #args.dataset
    dataset_config = {
        'CHD': {
            'Dataset': CT_dataset,
            'root_path': '/content/drive/ImageCHD_dataset',
            'volume_path': '/content/drive/ImageCHD_dataset',
            'list_dir': '/content/drive/lists',
            'num_classes': 2,
            'z_spacing': 1,
        },

    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.base_lr = 0.0001

    args.is_pretrain = True
    args.sum = False #True  # sum prediction heads?
    args.filter = 'hist' # "log": log_filter # "hist": hist_filter
 
    args.resume_train = True
    args.prev_train = None
    args.epo = 0
    args.epoch = -1
    args.it_num = 0
    args.retrain = 0

    args.exp = 'MIST_' + dataset_name + str(args.img_size)
    if args.mix_atten == False:
        args.exp = args.exp + '_noSSAM' 
    if args.sum != False:
        args.exp = args.exp+ '_sum' 
    
    snapshot_path = "/content/drive/MyDrive/lab_internship/code/MIST/model_pth/{}".format(args.exp)
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snap = '' #str('011635')
    snapshot_path = snapshot_path + "_run" + snap
    #net = MERIT_Parallel_Modified3(n_class=args.num_classes, img_size_s1=(args.img_size,args.img_size), img_size_s2=(224,224), model_scale='small', decoder_aggregation='additive', interpolation='bilinear').cuda()
    net = MIST_CAM(n_class=args.num_classes, img_size_s1=(args.img_size,args.img_size), img_size_s2=(512,512), model_scale='small', decoder_aggregation='additive', interpolation='bilinear')
    if torch.cuda.is_available():
        net = net.cuda()

    snapshot_name = ''
    snapshot = os.path.join(snapshot_path, snapshot_name+'.pth').replace('\\','/')
    if not os.path.exists(snapshot): 
        snapshot = snapshot.replace(snapshot_name,'last')
    net.load_state_dict(torch.load(snapshot))

    snapshot_name = snapshot_name
    log_folder = '/content/drive/MIST/results/' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    args.test_save_dir = os.path.join(log_folder, 'predictions')
    test_save_path = os.path.join(args.test_save_dir, snapshot_name)
    os.makedirs(test_save_path, exist_ok=True)
    inference(args, net, test_save_path=test_save_path)


# In[ ]:




