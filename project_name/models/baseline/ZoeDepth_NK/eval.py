import time
import numpy as np
import torch # type: ignore
import torch.nn.functional as F # type: ignore
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def evaluate_zoedepth_model(model, dataloader, device):
    model.eval()
    model.to(device)

    maes = []
    rmses = []
    times = []

    with torch.no_grad():
        for images, depths_gt in dataloader:
            images = images.to(device)
            depths_gt = depths_gt.to(device)

            start = time.time()
            preds = model(images)
            times.append(time.time() - start)

            if isinstance(preds, (list, tuple)):
                preds = preds[0]

            if preds.shape[-2:] != depths_gt.shape[-2:]:
                preds = F.interpolate(preds, size=depths_gt.shape[-2:], mode="bilinear", align_corners=False)

            preds_np = preds.squeeze().cpu().numpy().flatten()
            gt_np = depths_gt.squeeze().cpu().numpy().flatten()

            mae = mean_absolute_error(gt_np, preds_np)
            rmse = root_mean_squared_error(gt_np, preds_np, squared=False)

            maes.append(mae)
            rmses.append(rmse)

    print(f"ZoeDepth Evaluation:")
    print(f"  Avg RMSE: {np.mean(rmses):.4f}")
    print(f"  Avg MAE:  {np.mean(maes):.4f}")
    print(f"  Avg Inference time: {np.mean(times):.4f} seconds")

    return np.mean(rmses), np.mean(maes), np.mean(times)
