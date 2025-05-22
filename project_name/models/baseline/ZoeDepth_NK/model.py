import torch
from torch.utils.data import DataLoader
from data_loader import ZoeDepthDataset
from eval import evaluate_zoedepth_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    repo = "isl-org/ZoeDepth"
    model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True)

    dataset = ZoeDepthDataset("val_subset")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # Evaluate
    evaluate_zoedepth_model(model_zoe_nk, dataloader, device)


if __name__ == "__main__":
    main()
