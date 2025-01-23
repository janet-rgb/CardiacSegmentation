from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import nibabel as nib 
from skimage.transform import resize
import matplotlib_inline
import matplotlib
import matplotlib.pyplot as plt
from train_config import *
import re
from concurrent.futures import ThreadPoolExecutor

def filter_classes(label_slice, keep_classes=[0, 1, 3]):
    filtered_label_slice = np.zeros_like(label_slice)
    for cls in keep_classes:
        filtered_label_slice[label_slice == cls] = cls
    return filtered_label_slice

def process_slice(slice_2d, target_size):
   # print('in process slice')
    slice_2d = slice_2d / np.max(slice_2d) if np.max(slice_2d) > 0 else slice_2d
    return resize(slice_2d, target_size, preserve_range=True)

def load_nii_file(filepath, target_size):
    nifti = nib.load(filepath)
    volume = nifti.get_fdata()

    with ThreadPoolExecutor() as executor:
        slices = list(executor.map(lambda i: process_slice(volume[:, :, i], target_size), range(volume.shape[2])))

    return np.array(slices, dtype=np.float32)


def load_mask_file(filepath, target_size):

    nifti = nib.load(filepath)
    volume = nifti.get_fdata()  # Load data without normalization

    slices = []
 
    for i in range(volume.shape[2]):
        slice_2d = volume[:, :, i]
        resized_slice = resize(slice_2d, target_size, preserve_range=True, order=0)  # Use order=0 to preserve discrete values
        resized_slice=np.squeeze(resized_slice)
        slices.append(resized_slice)
    return np.array(slices, dtype=np.uint8)  # Ensure integer type


from sklearn.model_selection import train_test_split

def split_train_val(train_path, val_size=0.2, seed=42):
    # List all image and mask files
    image_files = sorted([os.path.join(train_path, f) for f in os.listdir(train_path) if "image.nii.gz" in f])
    mask_files = sorted([os.path.join(train_path, f) for f in os.listdir(train_path) if "label.nii.gz" in f])

    train_images, val_images, train_masks, val_masks = train_test_split(
    image_files, mask_files, test_size=val_size, random_state=seed
    )
    return train_images, val_images, train_masks, val_masks


def dataGenerator(image_files, mask_files, batch_size=None, aug_dict=None, 
                target_size=(512, 512), target_class=None, seed=1, shuffle=True):
    print("Generator initialized")

    if aug_dict is None:
        aug_dict = aug_dict  # 여기서 train_config.py의 aug_dict를 자동 적용

    use_augmentation = aug_dict and (aug_dict.get("image", {}) or aug_dict.get("mask", {}))

    if use_augmentation:
        image_datagen = ImageDataGenerator(**aug_dict["image"])
        mask_datagen = ImageDataGenerator(**aug_dict["mask"])
    else:
        image_datagen = None
        mask_datagen = None
    # Set random seed for reproducibility
    np.random.seed(seed)

    while True:
        # Shuffle files
        if shuffle:
            indices = np.random.permutation(len(image_files))
            image_files = [image_files[i] for i in indices]
            mask_files = [mask_files[i] for i in indices]

        for img_file, mask_file in zip(image_files, mask_files):
            # Load slices dynamically
            img_slices = load_nii_file(img_file, target_size)
            mask_slices = load_mask_file(mask_file, target_size)

            # Preprocess slices dynamically
            img_slices = img_slices / 255.0 if np.max(img_slices) > 1 else img_slices
            mask_slices = filter_classes(mask_slices, keep_classes=[0, 1, 3])

            for idx in range(0, img_slices.shape[0], batch_size):
                img_batch = img_slices[idx:idx + batch_size]
                mask_batch = mask_slices[idx:idx + batch_size]

                # Apply target class filter
                if target_class is not None:
                    mask_batch = (mask_batch == target_class).astype(np.uint8)

                # Add channel dimensions
                img_batch = np.expand_dims(img_batch, axis=-1)
                mask_batch = np.expand_dims(mask_batch, axis=-1)

                # Apply augmentations
                if use_augmentation:
                    augmented_imgs, augmented_masks = [], []
                    for img, mask in zip(img_batch, mask_batch):
                        img_aug = image_datagen.flow(np.expand_dims(img, 0), batch_size=1, seed=seed).__next__()[0]
                        mask_aug = mask_datagen.flow(np.expand_dims(mask, 0), batch_size=1, seed=seed).__next__()[0]
                        augmented_imgs.append(img_aug)
                        augmented_masks.append(mask_aug)
                    img_batch = np.array(augmented_imgs, dtype=np.float32)
                    mask_batch = np.array(augmented_masks, dtype=np.uint8)

                # Yield batch dynamically
                yield (
                    tf.convert_to_tensor(img_batch, dtype=tf.float32),
                    tf.convert_to_tensor(mask_batch, dtype=tf.float32)
                )
   
def testGeneratorWithLabels(data_path, target_class=None, target_size=(512, 512)):

    # 모든 파일 리스트 로드
    files = sorted(os.listdir(data_path))
    print(f"Files in {data_path}: {files}")  # 디버깅용 출력

    # 파일 이름에서 CT ID 추출 및 그룹화
    ct_groups = {}
    for f in files:
        match = re.match(r"(ct_\d+)_(image|label)\.nii\.gz", f)
        if match:
            ct_id = match.group(1)  # 예: ct_1140
            file_type = match.group(2)  # image 또는 label
            if ct_id not in ct_groups:
                ct_groups[ct_id] = {"image": None, "label": None}
            ct_groups[ct_id][file_type] = os.path.join(data_path, f)
    print(f"CT groups: {ct_groups}")  # 디버깅용 출력

    # CT 그룹별로 데이터 처리
    for ct_id, files in ct_groups.items():
        image_file = files["image"]
        label_file = files["label"]

        if not image_file or not label_file:
            print(f"Skipping CT {ct_id} due to missing files. Image: {image_file}, Label: {label_file}")
            continue

        # 이미지 및 라벨 데이터 로드
        image_slices = load_nii_file(image_file, target_size)  # Shape: (depth, height, width)
        label_slices = load_mask_file(label_file, target_size)  # Shape: (depth, height, width)

        for i in range(image_slices.shape[0]):
            # 슬라이스 단위 처리
            image_slice = image_slices[i].astype(np.float32)
            image_slice = image_slice / 255.0 if np.max(image_slice) > 1 else image_slice
            image_slice = np.expand_dims(image_slice, axis=-1)  # 채널 차원 추가

            # 라벨 슬라이스 처리
            label_slice = filter_classes(label_slices[i], keep_classes=[0, 1, 3])
            if target_class is not None:
                label_slice = (label_slice == target_class).astype(np.uint8)
                if np.sum(label_slice) == 0:  # 타겟 클래스가 없는 경우 스킵
                    continue

            yield np.expand_dims(image_slice, axis=0), np.expand_dims(label_slice, axis=0), ct_id



