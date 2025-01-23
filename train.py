from data import *
from tensorflow.keras.callbacks import ModelCheckpoint
import os
from keras.losses import BinaryCrossentropy
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
# 모든 데이터를 Training 데이터로 로드
train_images_LA = sorted([os.path.join('data/train', f) for f in os.listdir('data/train') if "image.nii.gz" in f])
train_masks_LA = sorted([os.path.join('data/train', f) for f in os.listdir('data/train') if "label.nii.gz" in f])

train_gen_LA = dataGenerator(
    train_images_LA, train_masks_LA, batch_size=2, aug_dict=aug_dict, target_size=(512, 512), shuffle=True, target_class=3
)

class_ratios = {}
for mask_file in train_masks_LA:
    mask_data = load_mask_file(mask_file, (512, 512))
    unique, counts = np.unique(mask_data, return_counts=True)

    for k, v in zip(unique, counts):
        class_ratios[k] = class_ratios.get(k, 0) + v
print(f"Class Ratios: {class_ratios}")

total = sum(class_ratios.values())
class_weights = {k: total / (2 * v) for k, v in class_ratios.items()}
print(f"Class Weights: {class_weights}")

def weighted_focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        pt = tf.exp(-bce)  # pt = y_pred if y_true == 1, 1-y_pred if y_true == 0
        focal = alpha * tf.pow(1 - pt, gamma) * bce
        return tf.reduce_mean(focal)
    return loss  

model_LA = unet_binary(input_size=(512, 512, 1), pretrained_weights=None)
model_LA.compile(optimizer=Adam(learning_rate=1e-5), loss=weighted_focal_loss(alpha=0.25, gamma=2.0), metrics=['binary_accuracy'])

checkpoint = ModelCheckpoint(
    filepath="best_LA_model.weights.h5",  # 저장할 가중치 파일 경로
    monitor="loss",                   # Training 손실을 기준으로 저장
    save_best_only=True,              # 최적의 가중치만 저장
    save_weights_only=True,           # 모델 구조는 저장하지 않고 가중치만 저장
    mode="min",                       # 손실이 최소일 때 저장
    verbose=0                         # 저장 여부를 출력
)
if __name__=="__main__":
  history= model_LA.fit(
    train_gen_LA,
    steps_per_epoch=20000 //2,
    epochs=5,
    callbacks=[checkpoint]            # ModelCheckpoint 추가
)