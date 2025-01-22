import argparse
import logging
import os
import random
import sys
import time
import numpy as np
from torch.cuda.graphs import CUDAGraph
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CTCLoss, CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast

from utils.dataset_synapse import CT_dataset, RandomGenerator
from utils.dataset_synapse import log_filter, hist_filter

from utils.utils import powerset
from utils.utils import one_hot_encoder
from utils.utils import DiceLoss
from utils.utils import val_single_volume

#def worker_init_fn(worker_id):
#        sd = 2222
#        random.seed(sd)
#        np.random.seed(sd)
#        torch.manual_seed(sd)
#        torch.cuda.manual_seed(sd)
#        return random.seed(sd + worker_id) 
           
def inference(args, model, best_performance):
    db_test = CT_dataset(base_dir=args.volume_path, split="val_vol", list_dir=args.list_dir, nclass=args.num_classes)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = val_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
    metric_list = metric_list / len(db_test)
    performance = np.mean(metric_list, axis=0)
    logging.info('Testing performance in val model: mean_dice : %f, best_dice : %f' % (performance, best_performance))
    return performance

def trials(args):
    args.filter = hist_filter
    db_train = CT_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train", nclass=args.num_classes, filter=args.filter,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    trainloader = DataLoader(db_train, batch_size=3, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label'].squeeze(1)
            #image_batch, label_batch = image_batch.cuda(), label_batch.squeeze(1).cuda()
            image_batch, label_batch = image_batch.cpu(), label_batch.squeeze(1).cpu()
            unique_labels = torch.unique(label_batch)
            print('label contains',unique_labels)
            if unique_labels.min() < 0 or unique_labels.max() >= 2:
                print(sampled_batch['case_name'], i_batch, "Invalid label values: {unique_labels}")
    
def save_train(snapshot, iter_num, image, output, label):
    save_dir = os.path.join(snapshot, 'training_img')
    os.makedirs(save_dir, exist_ok=True)
    img = image[1, 0:1, :, :]
    img = (img - img.min()) / (img.max() - img.min())
    img_norm = img.squeeze().cpu().numpy()
    output = torch.argmax(torch.softmax(output, dim=1), dim=1, keepdim=True)
    pred_norm = (output[1, ...].squeeze().cpu().numpy() * 50).astype(np.uint8)  # Scale prediction
    label = label[1, ...].unsqueeze(0)
    label_norm = (label.squeeze().cpu().numpy() * 50).astype(np.uint8)  # Scale label
    print(img_norm.shape, pred_norm.shape, label_norm.shape)
    combined = np.concatenate((img_norm.squeeze(), label_norm, pred_norm), axis=1)
    combined = combined.astype(np.uint8)
    combined_path = os.path.join(save_dir, f"iter_{iter_num:04d}.bmp")
    Image.fromarray(combined).save(combined_path)

def write_train(args, iter_num, image_batch, output, label_batch, writer):
    image = image_batch[1, 0:1, :, :]
    image = (image - image.min()) / (image.max() - image.min())
    writer.add_image('train/Image', image, iter_num)
    output = torch.argmax(torch.softmax(output, dim=1), dim=1, keepdim=True)
    writer.add_image('train/Prediction', output[1, ...] * 50, iter_num)
    labs = label_batch[1, ...].unsqueeze(0) * 50
    writer.add_image('train/GroundTruth', labs, iter_num)   

def trainer_CT(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    if args.filter == 'hist': args.filter = hist_filter
    if args.filter == 'log': args.filter = log_filter
    db_train = CT_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train", nclass=args.num_classes, filter=args.filter,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))

    print("The length of train set 1 is: {}".format(len(db_train)))
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    #trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    print('dataloader ready')
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    #ce_loss = CrossEntropyLoss()
    class_weights = torch.tensor([1.0, 9.0], dtype=torch.float32).cuda()
    ce_loss = CrossEntropyLoss(weight=class_weights)
    dice_loss = DiceLoss(num_classes, class_weights = class_weights)
    #optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    #iterator = tqdm(range(max_epoch), ncols=70)
    leng = len(trainloader)//10

    for epoch_num in range(args.epoch+1, args.max_epochs):

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label'].squeeze(1)
            image_batch, label_batch = image_batch.cuda(), label_batch.squeeze(1).cuda()

            P = model(image_batch)   # p is tuple of each prediction head 

            if isinstance(P, tuple): 
                if args.sum == True: 
                    output = 0.2*P[1] + 0.3*P[2] + 0.5*P[3]
                else: 
                    output = P[3]
            if isinstance(P, torch.Tensor): 
                output = P

            loss = 0.0
            lc1, lc2 = 0.3, 0.7 #0.3, 0.7
            dice = 0.0
            ce = 0.0

            loss_ce = ce_loss(output, label_batch[:].long())
            loss_dice = dice_loss(output, label_batch, softmax=True) #output = torch tensor([8,2,512,512])
            loss += (lc1 * loss_ce + lc2 * loss_dice) #0.3 CE, 0.7 DICE
            dice += loss_dice
            ce += loss_ce
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #if iter_num > 1000:
            #    lr_ = base_lr * (1.0 - (iter_num-1000) / max_iterations) ** 0.9 # we did not use this
            lr_ = base_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)                
            logging.info('iteration %d, epoch %d : loss : %f, ce: %f dice: %f, lr: %f' % (iter_num, epoch_num, loss.item(),ce, dice,  lr_))

            with open(snapshot_path+"/log.txt", "a") as file:
                file.write('iteration %d : loss : %f, loss_ce: %f ' % (iter_num, loss.item(), loss_ce.item()))
                file.write("\n")
            #print('iteration %d : loss : %f, loss_dice:%f, loss_ce: %f' % (iter_num, loss.item(), loss_dice.item(), loss_ce.item()))
            if iter_num % 20 == 1:
                save_train(snapshot_path, iter_num, image_batch, output, label_batch)
            if iter_num % leng == 0:
                save_mode_path = os.path.join(snapshot_path, 'epo%diter%d.pth'%(epoch_num, iter_num))
                torch.save(model.state_dict(), save_mode_path)    
                logging.info('model saved to epo%diter%d.pth'%(epoch_num, iter_num))       

        #performance = inference(args, model, best_performance)
        #if(best_performance <= performance):
        #    best_performance = performance
        #    save_mode_path = os.path.join(snapshot_path, 'best.pth')
        #    torch.save(model.state_dict(), save_mode_path)
        #    logging.info("save model to {}".format(save_mode_path))
            
        save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        torch.save(model.state_dict(), save_mode_path)
        logging.info("save model to {}".format(save_mode_path))
        
        if epoch_num >= max_epoch - 1:
            break
    writer.close()
    return "Training Finished!"
