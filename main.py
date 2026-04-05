import random

import torch
from torch.utils.data import DataLoader

from coursework.coursework.dataset_dicom import PneumothoraxDicomDataset
from coursework.coursework.model import UNet
from coursework.coursework.predict_dicom import predict_single
from coursework.coursework.settings import *
from coursework.coursework.train_dicom import train_model, load_checkpoint
from coursework.coursework.utils import load_csv, get_image_id_column, get_mask_column, build_dicom_map, set_seed, \
    ensure_dirs


def prepare_data():
    df = load_csv(CSV_PATH)
    image_id_col = get_image_id_column(df)
    mask_col = get_mask_column(df)

    dicom_map = build_dicom_map(DICOM_DIR)
    print(f"found dicom files: {len(dicom_map)}")

    rle_map = df.groupby(image_id_col)[mask_col].apply(list).to_dict()

    positive_ids = sorted(df[df[mask_col].astype(str).str.strip() != "-1"][image_id_col].unique().tolist())
    negative_ids = sorted(df[df[mask_col].astype(str).str.strip() == "-1"][image_id_col].unique().tolist())

    # оставляем только те id, для которых реально есть dcm
    positive_ids = [x for x in positive_ids if x in dicom_map]
    negative_ids = [x for x in negative_ids if x in dicom_map]

    print(f"positive ids with files: {len(positive_ids)}")
    print(f"negative ids with files: {len(negative_ids)}")

    return dicom_map, rle_map, positive_ids, negative_ids


def make_overfit_split(positive_ids, count=10):
    ids = positive_ids[:count]
    return ids, ids


def make_balanced_split(positive_ids, negative_ids, pos_count=1000, neg_count=1000, val_ratio=0.2):
    positive_ids = positive_ids.copy()
    negative_ids = negative_ids.copy()

    random.shuffle(positive_ids)
    random.shuffle(negative_ids)

    pos_ids = positive_ids[:pos_count]
    neg_ids = negative_ids[:neg_count]

    all_ids = pos_ids + neg_ids
    random.shuffle(all_ids)

    split_idx = int(len(all_ids) * (1.0 - val_ratio))
    train_ids = all_ids[:split_idx]
    val_ids = all_ids[split_idx:]

    return train_ids, val_ids


def choose_predict_id(predict_source, positive_ids, negative_ids):
    if predict_source == "positive":
        if not positive_ids:
            raise RuntimeError("no positive ids available")
        return random.choice(positive_ids)

    if predict_source == "negative":
        if not negative_ids:
            raise RuntimeError("no negative ids available")
        return random.choice(negative_ids)

    if predict_source == "random":
        all_ids = positive_ids + negative_ids
        if not all_ids:
            raise RuntimeError("no images available")
        return random.choice(all_ids)

    raise ValueError("PREDICT_SOURCE should be 'positive', 'negative' or 'random'")

def main():
    if RUN_MODE == "train":
        set_seed(SEED)
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    dicom_map, rle_map, positive_ids, negative_ids = prepare_data()

    if len(positive_ids) == 0:
        raise RuntimeError("positive images not found")

    if RUN_MODE == "train":
        if MODE == "overfit":
            image_size = OVERFIT_IMAGE_SIZE
            batch_size = OVERFIT_BATCH_SIZE
            epochs = OVERFIT_EPOCHS

            train_ids, val_ids = make_overfit_split(
                positive_ids=positive_ids,
                count=OVERFIT_POSITIVE_COUNT
            )

            print("\nmode: overfit")
            print("train ids:", len(train_ids))
            print("val ids:", len(val_ids))

        elif MODE == "balanced":
            image_size = BALANCED_IMAGE_SIZE
            batch_size = BALANCED_BATCH_SIZE
            epochs = BALANCED_EPOCHS

            train_ids, val_ids = make_balanced_split(
                positive_ids=positive_ids,
                negative_ids=negative_ids,
                pos_count=BALANCED_POSITIVE_COUNT,
                neg_count=BALANCED_NEGATIVE_COUNT,
                val_ratio=BALANCED_VAL_RATIO
            )

            print("\nmode: balanced")
            print("train ids:", len(train_ids))
            print("val ids:", len(val_ids))

        else:
            raise ValueError("MODE should be 'overfit' or 'balanced'")

        train_dataset = PneumothoraxDicomDataset(
            dicom_map=dicom_map,
            rle_map=rle_map,
            id_list=train_ids,
            image_size=image_size
        )

        val_dataset = PneumothoraxDicomDataset(
            dicom_map=dicom_map,
            rle_map=rle_map,
            id_list=val_ids,
            image_size=image_size
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0, # 0
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # 0
            pin_memory=True
        )

        model = UNet(in_channels=1, out_channels=1, base_channels=16).to(device)

        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            device=device
        )

        if RUN_PREDICT_AFTER_TRAIN:
            model = UNet(in_channels=1, out_channels=1, base_channels=16)
            model = load_checkpoint(model, MODEL_PATH, device)

            test_id = train_ids[0]
            test_path = dicom_map[test_id]

            predict_single(
                model=model,
                dicom_path=test_path,
                device=device,
                image_size=image_size,
                threshold=PRED_THRESHOLD,
                prefix="train_check",
                image_id=test_id,
                dicom_map=dicom_map,
                rle_map=rle_map
            )

    elif RUN_MODE == "predict":
        image_size = BALANCED_IMAGE_SIZE

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"trained model not found: {MODEL_PATH}")

        test_id = choose_predict_id(
            predict_source=PREDICT_SOURCE,
            positive_ids=positive_ids,
            negative_ids=negative_ids
        )

        test_path = dicom_map[test_id]

        model = UNet(in_channels=1, out_channels=1, base_channels=16)
        model = load_checkpoint(model, MODEL_PATH, device)

        predict_single(
            model=model,
            dicom_path=test_path,
            device=device,
            image_size=image_size,
            threshold=PRED_THRESHOLD,
            prefix=f"{PREDICT_SOURCE}_{test_id}",
            image_id=test_id,
            dicom_map=dicom_map,
            rle_map=rle_map
        )

    else:
        raise ValueError("RUN_MODE should be 'train' or 'predict'")


if __name__ == "__main__":
    main()
