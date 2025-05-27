import torch  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
from .data_loader import ZoeDepthDataset
from .eval import evaluate_zoedepth_model


class ZoeDepthEvaluator:
    def __init__(
        self,
        split: str = "val",
        batch_size: int = 1,
        num_workers: int = 4
    ):
        d = torch.device("cuda"if torch.cuda.is_available() else "cpu")
        self.device = d

        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers

        print(f"Using device: {self.device}")
        self.model = self._load_model()
        self.dataloader = self._load_data()

    def _load_model(self):
        print("Loading ZoeDepth model from torch.hub...")
        model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)
        return model

    def _load_data(self):
        print(f"Loading dataset split: {self.split}")
        dataset = ZoeDepthDataset(self.split)
        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                shuffle=False, num_workers=self.num_workers)
        return dataloader

    def evaluate(self):
        print("Evaluating model...")
        rmse, mae, inference_time = evaluate_zoedepth_model(
            self.model, self.dataloader, self.device
        )
        return {
            "rmse": rmse,
            "mae": mae,
            "avg_inference_time": inference_time
        }


def main():
    evaluator = ZoeDepthEvaluator(split="val", batch_size=1, num_workers=4)
    results = evaluator.evaluate()

    print("\n--- Final Results ---")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"MAE:  {results['mae']:.4f}")
    print(f"Inference Time: {results['avg_inference_time']:.4f} sec")


if __name__ == "__main__":
    main()
