import os
import random
from cv2.gapi import imgproc
import h5py
import numpy as np
from numpy.core.multiarray import packbits
import torch
import cv2
import nibabel as nib
from nibabel.imageglobals import LoggingOutputSuppressor
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
def imsave(im, lbl,id):
    color1 = np.array([0,255,0], dtype='uint8')
    masked = np.where(lbl[...,None], color1, im[...,None])
    img = masked
    #im = im[:,:, np.newaxis]
    #im = np.concatenate((im,im,im), axis = 2).squeeze()
    #img = np.concatenate((im, masked), axis=1)
    cv2.imwrite('img%d.bmp'%id,img)
    
def hist_filter(sample):
    image, label = sample['image'], sample['label']
    image = (image * 255).astype('uint8')  # If image is normalized to [0, 1]
    eq_img = cv2.equalizeHist(image)
    sample = {'image': eq_img, 'label': label}

    return sample

def normalize(image, label):
    image = image/(image.max()-image.min())
    return image, label

def random_rot_flip(image, label): #normal
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k) 
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label): #normal

    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        image, label = normalize(image, label)

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        
        if random.random() > 0.5:
            image, label = random_rotate(image, label)

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            print('false')
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}

        return sample

class CT_dataset(Dataset):
    def __init__(self, base_dir, list_dir, nclass, split, filter=None, divide = None, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.vol_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.divide = divide
        self.nclass = nclass
        self.filter = filter
        ls = []
        with LoggingOutputSuppressor():
            for i in range(len(self.vol_list)):
                name = self.vol_list[i].strip('\n')+'_label.nii.gz'
                vol = nib.load(os.path.join(self.data_dir, name))
                if len(vol.shape) != 3:
                    print(name, vol.shape)
                for j in range(vol.shape[2]):
                    ls.append(self.vol_list[i]+'_slice'+str(j))
        self.sample_list = ls
        self.leng = len(ls)

    def __len__(self):
        if self.split == 'test':
            return len(self.vol_list)
        elif self.divide == 1 or self.divide == 2:
            return (self.leng)//3
        elif self.divide == 3:
            return (self.leng - 2*self.leng//3)
        else:
            return self.leng

    def __getitem__(self, idx):
        if self.split == "train":

            if self.divide == 2: idx += (self.leng)//3
            elif self.divide ==3: idx += 2*(self.leng)//3

            slice_name = self.sample_list[idx].strip('\n')
            path1 = os.path.join(self.data_dir, slice_name.split('_slice')[0].strip('\n')+'_image.nii.gz')
            path2 = os.path.join(self.data_dir, slice_name.split('_slice')[0].strip('\n')+'_label.nii.gz')
            with LoggingOutputSuppressor():

                image_ = nib.load(path1)#.get_fdata()
                label_ = nib.load(path2)#.get_fdata()

                hd=image_.header
                hd['pixdim'][0] = 1
                hd=label_.header
                hd['pixdim'][0] = 1

                image_ = image_.get_fdata()
                label_ = label_.get_fdata()

            val = [0,3]
            idx = int(slice_name.split('_slice')[1].strip('\n'))
            image = image_[:,:,idx ]#*255
            label = label_[:,:,idx]
            #label = np.where(np.isin(label_, val), label_, 0)
            #label[label == 3] = 1

        else:
            vol_name = self.vol_list[idx].strip('\n')  
            path1 = os.path.join(self.data_dir, vol_name+'_image.nii.gz')
            path2 = os.path.join(self.data_dir, vol_name+'_label.nii.gz')
            image = nib.load(path1).get_fdata()
            label = nib.load(path2).get_fdata()

        #if label.ndim ==2: label=label[ :,:, np.newaxis,]
        #if image.ndim ==2: image=image[:,:,np.newaxis]
        sample = {'image': image, 'label': label}

        if self.filter:
            if len(image.shape)<3:
                sample = self.filter(sample)
            else:
                image = (image * 255).astype('uint8')
                for i in range(image.shape[-1]):
                    img = image[:,:,i]
                    img = cv2.equalizeHist(img)
                    image[:,:,i]=img
                sample['image'] = image

        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        if self.split == 'test':
            sample['case_name']=self.vol_list[idx].strip('\n')

        return sample


if __name__=='__main__':
    from torchvision import transforms
    mode = 'train'
    dataset=CT_dataset(base_dir='/content/drive/MyDrive/lab_internship/code/TransUNet/TransUNet-main/data/ImageCHD_dataset2', list_dir='/content/drive/MyDrive/lab_internship/code/TransUNet/TransUNet-main/lists',filter=hist_filter,split=mode, nclass=2,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[512,512])]))
    #db_test = Synapse_dataset(base_dir='C:/Users/AIMEDIC/Desktop/LALV/data_LV', split="test", list_dir='../lists/lists_Synapse',nclass=2)
    print(dataset[0]['label'].shape)
    print(dataset[0]['image'].shape)
