import time
import numpy as np
import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from PIL import Image
import cv2  # type: ignore


def save_prediction_images(img_tensor, pred_tensor, index, save_dir="pred"):

    os.makedirs(save_dir, exist_ok=True)

    img_norm = normalize(img_tensor)
    pred_norm = normalize(pred_tensor.squeeze().cpu())

    input_np = (img_norm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    pred_np = (pred_norm.cpu().numpy() * 255).astype(np.uint8)

    h, w = input_np.shape[:2]
    pred_np_resized = cv2.resize(pred_np, (w, h), cv2.INTER_LINEAR)

    if input_np.shape[2] == 1:
        input_np = np.repeat(input_np, 3, axis=2)

    pred_img_3ch = np.stack([pred_np_resized]*3, axis=2)

    combined_img = np.concatenate((input_np, pred_img_3ch), axis=1)

    combined_img_pil = Image.fromarray(combined_img)
    combined_img_pil.save(os.path.join(save_dir, f"combined_{index}.png"))


def normalize(img):
    """Normalize array for visualization."""
    if isinstance(img, torch.Tensor):
        img_min = img.min()
        img_max = img.max()
        return (img - img_min) / (img_max - img_min + 1e-8)
    else:
        img_min = np.min(img)
        img_max = np.max(img)
        return (img - img_min) / (img_max - img_min + 1e-8)


def evaluate_zoedepth_model(model, dataloader, device):
    model.eval()
    model.to(device)

    maes = []
    rmses = []
    times = []

    with torch.no_grad():
        for i, (images, depths_gt) in enumerate(dataloader):
            print(f"\n--- Batch {i} ---")

            images = images.to(device)
            depths_gt = depths_gt.to(device)

            start = time.time()
            preds = model(images)
            times.append(time.time() - start)

            if isinstance(preds, dict):
                preds = preds["metric_depth"]
            elif isinstance(preds, (list, tuple)):
                preds = preds[0]

            if preds.shape[-2:] != depths_gt.shape[-2:]:
                target_size = depths_gt.shape[-2:]
                preds = F.interpolate(
                    preds,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False
                )

            preds_np = preds.cpu().numpy().reshape(-1)
            gt_np = depths_gt.cpu().numpy().reshape(-1)

            mae = mean_absolute_error(gt_np, preds_np)
            rmse = np.sqrt(mean_squared_error(gt_np, preds_np))
            maes.append(mae)
            rmses.append(rmse)

            img, pred = images[0].cpu(), preds[0].cpu()
            save_prediction_images(img, pred, index=i, save_dir='save_dir')

            if i < 0:
                fig, axs = plt.subplots(1, 2, figsize=(12, 6))

                axs[0].imshow(images.squeeze().cpu().permute(1, 2, 0))
                axs[0].set_title("Input Image")
                axs[0].axis('off')

                pred_np_img = preds.squeeze().cpu().numpy()
                axs[1].imshow(normalize(pred_np_img), cmap='inferno')
                axs[1].set_title("Predicted Depth (Normalized)")
                axs[1].axis('off')

                plt.tight_layout()
                plt.show()

    return np.mean(rmses), np.mean(maes), np.mean(times)
