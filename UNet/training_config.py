from model import *
from data import *
from loss_metrics import *
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import os

aug_dict = {
    "image": dict(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.1,
        fill_mode="nearest"
    ),
    "mask": dict(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.1,
        fill_mode="constant",  # Mask는 빈 영역을 0으로
        cval=0
    )
}

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss',       # Metric to monitor (e.g., 'loss', 'val_loss', or 'accuracy')
    factor=0.5,           # Factor by which to reduce the learning rate
    patience=5,           # Number of epochs with no improvement before reducing LR
    min_lr=1e-6           # Minimum learning rate
)

# LA 모델의 체크포인트 콜백
checkpoint_LA = ModelCheckpoint(
    "weights/best_LA_model.weights.h5",  # 저장 경로
    monitor="val_loss",             # 검증 손실을 기준으로 가장 좋은 가중치 저장
    save_best_only=True,   
    save_weights_only=True,         # 가중치만 저장
    mode="min",                     # 손실이 최소일 때 저장
    verbose=1
)


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=10,          # Stop training if no improvement after 5 epochs
    restore_best_weights=True  # Restore the weights of the best epoch
)