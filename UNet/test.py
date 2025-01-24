from model import *
from data import *
from loss_metrics import *
import matplotlib.pyplot as plt
import os
import pandas as pd
import h5py

save_path = "data/test/results"
if not os.path.exists(save_path):
    os.makedirs(save_path)
data_path = "data/test"
evalGene_LA = list(testGeneratorWithLabels(data_path, target_class=3))

weights_path_LA ="best_LA_model.weights.h5"
model_LA = unet_binary(input_size=(512, 512, 1), pretrained_weights=None)
if os.path.exists(weights_path_LA):
    model_LA.load_weights(weights_path_LA)
    print("LA Model weights loaded successfully!")
else:
    raise FileNotFoundError(f"LA weights not found at {weights_path_LA}")

# 평가: LA total_slices=5632
mean_dice_LA, mean_hd95_LA = evaluate_model(model_LA, evalGene_LA, len(evalGene_LA),save_path= "data/test1/results",output_file="test_results.xlsx") 
print(f"LA - Mean Dice: {mean_dice_LA:.4f}, Mean HD95: {mean_hd95_LA:.4f}")
results = {
    "LA": {"Mean Dice": mean_dice_LA, "Mean HD95": mean_hd95_LA}
}



