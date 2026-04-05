import os

import cv2
import numpy as np
import torch

from settings import OUTPUT_ROOT_DIR
from utils import read_dicom_image, save_overlay, build_gt_mask, save_gt_overlay


def predict_single(
    model,
    dicom_path,
    device,
    image_size=128,
    threshold=0.2,
    prefix="pred",
    image_id=None,
    dicom_map=None,
    rle_map=None
):
    original = read_dicom_image(dicom_path)

    resized = cv2.resize(original, (image_size, image_size))
    tensor = resized.astype(np.float32) / 255.0
    tensor = np.expand_dims(tensor, axis=0)
    tensor = np.expand_dims(tensor, axis=0)
    tensor = torch.tensor(tensor, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output)
        pred = (prob > threshold).float()

    print(f"\nprediction for: {os.path.basename(dicom_path)}")
    print(f"min prob:  {prob.min().item():.6f}")
    print(f"max prob:  {prob.max().item():.6f}")
    print(f"mean prob: {prob.mean().item():.6f}")

    output_dir = os.path.join(OUTPUT_ROOT_DIR, prefix)
    os.makedirs(output_dir, exist_ok=True)
    # original_path = os.path.join(OUTPUT_DIR, f"{prefix}_original.png")
    # mask_path = os.path.join(OUTPUT_DIR, f"{prefix}_mask.png")
    # prob_path = os.path.join(OUTPUT_DIR, f"{prefix}_probability.png")
    overlay_path = os.path.join(output_dir, f"{prefix}_prediction.png")

    # save_original_image(original, original_path)
    # save_mask(pred.cpu(), mask_path)
    # save_probability_map(prob.cpu(), prob_path)
    save_overlay(original, pred.cpu(), overlay_path)

    # сохраняем эталонную маску и эталонный overlay, если есть данные
    if image_id is not None and dicom_map is not None and rle_map is not None:
        gt_mask = build_gt_mask(image_id, dicom_map, rle_map)

        # gt_mask_path = os.path.join(OUTPUT_DIR, f"{prefix}_gt_mask.png")
        gt_overlay_path = os.path.join(output_dir, f"{prefix}_original.png")

        # cv2.imwrite(gt_mask_path, gt_mask)
        save_gt_overlay(original, gt_mask, gt_overlay_path)

        # print(gt_mask_path)
        print(gt_overlay_path)

    print("saved:")
    # print(original_path)
    # print(mask_path)
    # print(prob_path)
    print(overlay_path)