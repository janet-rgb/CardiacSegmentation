from IPython.display import Image, display
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import os
import pandas as pd

def dice_score(pred, gt):
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)
    if union == 0:  # GT와 Pred 모두 없는 경우
        return 1.0 if np.sum(pred) == 0 else 0.0
    return 2.0 * intersection / union


def hausdorff95(pred, gt):

    pred_points = np.argwhere(pred > 0)
    gt_points = np.argwhere(gt > 0)

    if len(pred_points) == 0 or len(gt_points) == 0:  # 둘 중 하나라도 비어 있으면 inf 반환
        return None

    forward_dist = directed_hausdorff(pred_points, gt_points)[0]
    backward_dist = directed_hausdorff(gt_points, pred_points)[0]
    return max(forward_dist, backward_dist)

# 결과 저장 경로
#save_path = "data/test/results"
#os.makedirs(save_path, exist_ok=True)  # 결과 저장 디렉토리 생성

def save_prediction(img, true_mask, predicted_mask, idx, save_path):

# 원본 이미지 90도 회전 (시계 방향으로)
    img_rotated = np.rot90(img.squeeze(), k=3)
    true_mask_rotated = np.rot90(true_mask.squeeze(), k=3)
    predicted_mask_rotated = np.rot90(predicted_mask.squeeze(), k=3)

    # 이미지 좌우 반전
    img_flipped = np.fliplr(img_rotated)
    true_mask_flipped = np.fliplr(true_mask_rotated)
    predicted_mask_flipped = np.fliplr(predicted_mask_rotated)

    # 원본 이미지 값 범위와 데이터 유형 보정
    if img_flipped.max() <= 1:  # 값 범위가 0~1인 경우 0~255로 스케일 업
        img_flipped = (img_flipped * 255).astype(np.uint8)
    img_flipped = (img_flipped * 0.4).clip(0, 255).astype(np.uint8)

    # Predicted Mask (초록색) Overlay
    overlay_pred = np.stack([img_flipped, img_flipped, img_flipped], axis=-1)  # Grayscale to RGB
    overlay_pred[:, :, 1] = np.where(predicted_mask_flipped > 0, 255, img_flipped)  # Green channel, 100% opaque

    # Ground Truth (파란색) Overlay
    overlay_gt = np.stack([img_flipped, img_flipped, img_flipped], axis=-1) # Grayscale to RGB
    overlay_gt[:, :, 2] = np.where(true_mask_flipped > 0, 255, img_flipped)  # Blue channel, 100% opaque

       # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'wspace': 0})

    # 첫 번째 이미지를 왼쪽에
    axes[0].imshow(overlay_pred.astype(np.uint8))
    axes[0].axis('off')

    # 두 번째 이미지를 오른쪽에
    axes[1].imshow(overlay_gt.astype(np.uint8))
    axes[1].axis('off')

    # 간격 완전 제거
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # 파일 저장
    result_filename = os.path.join(save_path, f"result_{idx}.png")
    plt.savefig(result_filename, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    print(f"Result for slice {idx} saved as {result_filename}")

def evaluate_model(model, eval_data, total_slices, save_path, output_file="test_results.xlsx"):
    print('evaluate start')
    print(f"Total slices in eval_data: {len(eval_data)}")

    dice_scores = {}
    hd95_scores = {}

    for idx, (test_image, ground_truth, ct_id) in enumerate(eval_data[:total_slices]):
        prediction= model.predict(test_image)
    
        # Sigmoid 결과를 바이너리 형태로 변환
        pred = (prediction.squeeze(axis=0) > 0.3).astype(np.uint8)
        pred= np.squeeze(pred)

        # Ground Truth (이진화된 값)
        gt = ground_truth.squeeze(axis=0).astype(np.uint8)
        gt=np.squeeze(gt)
        print(f"Slice {idx}, CT {ct_id}: Prediction sum = {np.sum(pred)}, Ground Truth sum = {np.sum(gt)}")

        if np.sum(gt) == 0 or np.sum(pred) == 0:
            print(f"Skipping slice {idx} in CT {ct_id} due to empty prediction or ground truth.")
            continue
        # Dice Score 계산
        dice = dice_score(pred, gt)
        # HD95 계산
        hd95 = hausdorff95(pred, gt)
        #if hd95 is not None:  # 유효한 HD95만 추가

  # CT 단위로 결과 저장
        if ct_id not in dice_scores:
            dice_scores[ct_id] = []
            hd95_scores[ct_id] = []

        dice_scores[ct_id].append(dice)
        if hd95 is not None:  # 유효한 HD95만 추가
            hd95_scores[ct_id].append(hd95)
        else:
            print(f"Warning: HD95 is None for slice {idx} in CT {ct_id}")
        save_prediction(test_image, ground_truth, pred, idx, save_path)

        print(f"Slice {idx}: Dice = {dice:.4f}, HD95 = {hd95 if hd95 is not None else 'Invalid'}")
        
        if idx >= total_slices - 1:
          break
   # CT 단위로 평균 계산
    ct_results = []
    for ct_id in dice_scores:
        avg_dice = np.mean(dice_scores[ct_id])
        avg_hd95 = np.mean([hd for hd in hd95_scores[ct_id] if hd is not None])
        ct_results.append({
            "CT ID": ct_id,
            "Mean Dice Score": avg_dice,
            "Mean HD95 Score": avg_hd95
        })
    print("\nEvaluation Metrics:")
    for result in ct_results:
      print(f"CT {result['CT ID']}: Mean Dice = {result['Mean Dice Score']:.4f}, Mean HD95 = {result['Mean HD95 Score']:.4f}")          
 
 # 엑셀 파일 저장
    df = pd.DataFrame(ct_results)
    df.to_excel(output_file, index=False)
    print(f"CT-wise results saved to {output_file}")

    # 전체 평균 반환
    overall_dice = np.mean([res["Mean Dice Score"] for res in ct_results])
    overall_hd95 = np.mean([res["Mean HD95 Score"] for res in ct_results])
    return overall_dice, overall_hd95
