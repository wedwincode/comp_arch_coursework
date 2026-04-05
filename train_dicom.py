import torch
from tqdm import tqdm

from dataset_dicom import BCEDiceLoss
from settings import LEARNING_RATE, MODEL_PATH
from utils import dice_coeff, iou_score


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    loop = tqdm(loader, desc="train", leave=False)

    for images, masks, _ in loop:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / max(len(loader), 1)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    with torch.no_grad():
        loop = tqdm(loader, desc="validate", leave=False)

        for images, masks, _ in loop:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()
            total_dice += dice_coeff(outputs, masks)
            total_iou += iou_score(outputs, masks)

    n = max(len(loader), 1)
    return total_loss / n, total_dice / n, total_iou / n


def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def train_model(model, train_loader, val_loader, epochs, device):
    # pos_weight = torch.tensor([3.0], device=device) # todo
    # criterion = BCEDiceLoss(pos_weight=pos_weight)
    criterion = BCEDiceLoss(pos_weight=None) # todo
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_dice = -1.0

    for epoch in range(epochs):
        print(f"\nepoch {epoch + 1}/{epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, device)

        print(f"\ntrain loss: {train_loss:.4f}")
        print(f"val loss:   {val_loss:.4f}")
        print(f"val dice:   {val_dice:.4f}")
        print(f"val iou:    {val_iou:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(model, MODEL_PATH)
            print("\nthe best model is saved")

    print("\ntrain completed")
    print(f"best dice: {best_dice:.4f}")
