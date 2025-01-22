#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import logging
import os
import time
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from lib.networks import MIST_CAM

from trainer import trainer_CT

from torchsummaryX import summary
from ptflops import get_model_complexity_info


# In[2]:


import gc
gc.collect()
torch.cuda.empty_cache()
os.environ['CUDA_VISIBLE_DEVICES']='0, 1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# In[3]:


parser = argparse.ArgumentParser(description='Searching longest common substring. '
                    'Uses Ukkonen\'s suffix tree algorithm and generalized suffix tree. '
                    'Written by Ilya Stepanov (c) 2013')
parser.add_argument('strings', metavar='STRING', nargs='*', help='String for searching',)

parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=10, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=3, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of network input') #224
parser.add_argument('--seed', type=int,
                    default=2222, help='random seed')          
args = parser.parse_args("AAA".split())


# In[4]:


args


# In[ ]:


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
    args.base_lr = 0.0001

    args.is_pretrain = True
    args.sum = True #True  # sum prediction heads
    args.mix_atten = True # attention mixer
    args.filter = 'hist' # "log": log_filter # "hist": hist_filter
    args.resume_train = False
    args.epoch = -1

    args.exp = 'MIST_' + dataset_name + str(args.img_size)
    if args.mix_atten == False:
        args.exp = args.exp + '_noSSAM' 
    if args.sum != False:
        args.exp = args.exp+ '_sum' 

    snapshot_path = "/content/drive/MIST/model_pth/{}".format(args.exp)
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    #snapshot_path = snapshot_path + '_'+str(args.img_size)
    current_time = time.strftime("%H%M%S")
    print("The current time is", current_time)
    snapshot_path = snapshot_path +'_run'+current_time
    
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    #net = MERIT_Parallel_Modified3(n_class=args.num_classes, img_size_s1=(args.img_size,args.img_size), img_size_s2=(512,512), model_scale='small', decoder_aggregation='additive', interpolation='bilinear')
    net = MIST_CAM(n_class=args.num_classes, img_size_s1=(args.img_size,args.img_size), img_size_s2=(512,512), model_scale='small', decoder_aggregation='additive', interpolation='bilinear')
    
    print('Model %s created, param count: %d' %
                     ('MIST_CAM: ', sum([m.numel() for m in net.parameters()])))
    if args.resume_train:
        snapshot = ""
        snap = ""
        snapshot = os.path.join(snapshot, "epo%diter%d.pth"%(0, 1740)).replace("\\","/")
        if not os.path.exists:
            snapshot = snapshot.replace(snap, 'last')
        checkpoint = torch.load(snapshot)
        if os.path.exists(snapshot): net.load_state_dict(checkpoint)
        args.epoch = -1 
    net = net.cuda()
   
    #macs, params = get_model_complexity_info(net, (3, args.img_size, args.img_size), as_strings=True,
                                           #print_per_layer_stat=False, verbose=True)
    #print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    trainer = {'CHD': trainer_CT}
    trainer[dataset_name](args, net, snapshot_path)
    ''''''


# In[ ]:




