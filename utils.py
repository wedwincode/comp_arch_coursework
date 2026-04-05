import os
import random
import cv2
import torch
import pydicom
import numpy as np
import pandas as pd

from settings import CHECKPOINT_DIR, OUTPUT_ROOT_DIR


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)


def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [str(col).strip() for col in df.columns]
    return df


def get_image_id_column(df):
    candidates = ["ImageId", "image_id", "id", "ID"]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"image id column not found. columns: {df.columns.tolist()}")


def get_mask_column(df):
    candidates = [
        "EncodedPixels",
        "encodedpixels",
        "encoded_pixels",
        "mask_rle",
        "rle",
        "RLE",
        "Mask"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"mask/rle column not found. columns: {df.columns.tolist()}")


def build_dicom_map(dicom_dir):
    dicom_map = {}
    for root, _, files in os.walk(dicom_dir):
        for file_name in files:
            if file_name.lower().endswith(".dcm"):
                image_id = os.path.splitext(file_name)[0]
                full_path = os.path.join(root, file_name)
                dicom_map[image_id] = full_path
    return dicom_map

def prepare_pos_ids(positive_ids):
    return [v for i, v in enumerate(positive_ids[:20]) if i in (0, 4, 7, 11, 12, 16, 17, 18)]

def prepare_neg_ids(negative_ids):
    return [v for i, v in enumerate(negative_ids[:20]) if i in (4, 5, 6, 7, 13, 18, 19)]

def rle_decode(mask_rle, shape):
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    if pd.isna(mask_rle) or str(mask_rle).strip() == "-1":
        return mask.reshape(shape)

    s = list(map(int, str(mask_rle).split()))
    starts = s[0::2]
    lengths = s[1::2]

    current_position = 0
    for start, length in zip(starts, lengths):
        current_position += start
        mask[current_position:current_position + length] = 1
        current_position += length

    return mask.reshape(shape).T

def read_dicom_image(dicom_path):
    ds = pydicom.dcmread(dicom_path)
    image = ds.pixel_array.astype(np.float32)

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = image.max() - image

    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image = image.astype(np.uint8)
    return image


def save_mask(mask_pred, save_path):
    mask = mask_pred.squeeze().cpu().numpy()
    # mask = (mask > 0.5).astype(np.uint8) * 255 # todo
    mask = (mask > 0.2).astype(np.uint8) * 255
    cv2.imwrite(save_path, mask)

def build_gt_mask(image_id, dicom_map, rle_map):
    dicom_path = dicom_map[image_id]
    image = read_dicom_image(dicom_path)

    h, w = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    rles = rle_map.get(image_id, [])
    for rle in rles:
        decoded = rle_decode(rle, (h, w))
        mask = np.maximum(mask, decoded)

    return (mask * 255).astype(np.uint8)

def save_gt_mask(image_id, dicom_map, rle_map, save_path):
    mask = build_gt_mask(image_id, dicom_map, rle_map)
    cv2.imwrite(save_path, mask)

def save_gt_overlay(original, gt_mask, save_path):
    mask = (gt_mask > 0).astype(np.uint8) * 255

    original_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    green_mask = np.zeros_like(original_bgr)
    green_mask[:, :, 1] = mask  # зелёный

    overlay = cv2.addWeighted(original_bgr, 1.0, green_mask, 0.4, 0)
    cv2.imwrite(save_path, overlay)


def save_probability_map(prob_pred, save_path):
    prob = prob_pred.squeeze().cpu().numpy()
    prob = np.clip(prob, 0.0, 1.0)
    prob = (prob * 255).astype(np.uint8)
    cv2.imwrite(save_path, prob)


def save_overlay(original, pred_mask, save_path):
    mask = pred_mask.squeeze().cpu().numpy().astype(np.uint8) * 255
    mask = cv2.resize(mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)

    original_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    red_mask = np.zeros_like(original_bgr)
    red_mask[:, :, 2] = mask

    overlay = cv2.addWeighted(original_bgr, 1.0, red_mask, 0.4, 0)
    cv2.imwrite(save_path, overlay)


def save_original_image(original, save_path):
    cv2.imwrite(save_path, original)


def dice_coeff(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    # pred = (pred > 0.5).float() # todo
    pred = (pred > 0.2).float()

    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()


def iou_score(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    # pred = (pred > 0.5).float() # todo
    pred = (pred > 0.2).float()

    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()