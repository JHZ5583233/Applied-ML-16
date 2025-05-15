import os
import numpy as np
from skimage import io, color, filters
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def extract_features_and_targets(rgb_image, depth_map, tile_size=64):
    features = []
    targets = []

    gray = color.rgb2gray(rgb_image)
    edges = filters.sobel(gray)

    h, w = rgb_image.shape[:2]
    for y in range(0, h - tile_size + 1, tile_size):
        for x in range(0, w - tile_size + 1, tile_size):
            rgb_patch = rgb_image[y:y+tile_size, x:x+tile_size]
            edge_patch = edges[y:y+tile_size, x:x+tile_size]
            depth_patch = depth_map[y:y+tile_size, x:x+tile_size]

            mean_brightness = np.mean(rgb_patch)
            mean_edge_strength = np.mean(edge_patch)
            mean_depth = np.mean(depth_patch)

            features.append([mean_brightness, mean_edge_strength])
            targets.append(mean_depth)

    return features, targets


def load_dataset(image_dir, depth_dir, tile_size=64):
    X, y = [], []
    for fname in sorted(os.listdir(image_dir)):
        if fname.endswith(".png") or fname.endswith(".jpg"):
            rgb_path = os.path.join(image_dir, fname)
            depth_path = os.path.join(depth_dir, fname.replace("rgb", "depth"))

            if not os.path.exists(depth_path):
                print(f"Warning: Missing depth file for {fname}")
                continue

            rgb = io.imread(rgb_path).astype(np.float32) / 255.0
            depth = io.imread(depth_path).astype(np.float32)

            features, targets = extract_features_and_targets(rgb, depth, tile_size)
            X.extend(features)
            y.extend(targets)

    return np.array(X), np.array(y)


def train_linear_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\n📏 Test RMSE: {rmse:.3f}")

    return model, X_test, y_test, y_pred


def plot_results(y_test, y_pred):
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.5, s=10)
    plt.xlabel("True Mean Depth")
    plt.ylabel("Predicted Mean Depth")
    plt.title("Linear Regression Depth Prediction")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    IMAGE_DIR = ""
    DEPTH_DIR = ""
    TILE_SIZE = 64

    print("Loading and processing data...")
    X, y = load_dataset(IMAGE_DIR, DEPTH_DIR, TILE_SIZE)
    print(f"Dataset loaded: {len(X)} samples")

    print("\n Training linear regression model...")
    model, X_test, y_test, y_pred = train_linear_regression(X, y)

    print("\nPlotting prediction vs ground truth...")
    plot_results(y_test, y_pred)
