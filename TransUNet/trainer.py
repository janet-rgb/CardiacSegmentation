import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
import logging

def trainer_CT(args, model, snapshot_path):
    from datasets.dataset_synapse import CT_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    logging.disable(logging.WARNING)

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = CT_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train", divide = 1,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    db_train2 = CT_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train", divide = 2,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    db_train3 = CT_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train", divide = 3,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set 1 is: {}".format(len(db_train)))
    print("The length of train set 2 is: {}".format(len(db_train2)))
    print("The length of train set 3 is: {}".format(len(db_train3)))
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    trainloader2 = DataLoader(db_train2, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    trainloader3 = DataLoader(db_train3, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    #print('resume training in batch %d with iter num %d'%(args.retrain+1, args.it_num))
    leng = len(trainloader)//10
    
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            outputs = model(image_batch)

            loss_ce = ce_loss(outputs, label_batch[:].long())

            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))
            with open(snapshot_path+"/log.txt", "a") as file:
                file.write('iteration %d : loss : %f, loss_ce: %f ' % (iter_num, loss.item(), loss_ce.item()))
                file.write("\n")
            print('iteration %d : loss : %f, loss_dice:%f, loss_ce: %f' % (iter_num, loss.item(), loss_dice.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[0, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[0, ...] * 50, iter_num)
                labs = label_batch[0, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
            if iter_num % leng == 0:
                save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) +'iter_'+ \
                str(iter_num)+ '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                print("save model to {}".format(save_mode_path))

        save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        torch.save(model.state_dict(), save_mode_path)
        logging.info("save model to {}".format(save_mode_path))
        with open(snapshot_path+"/log.txt", "a") as file:
            file.write("save model to {}".format(save_mode_path))
            file.write("\n")
            file.close()

        if epoch_num >= max_epoch - 1:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"
