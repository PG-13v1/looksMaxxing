import os
import yaml
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import load_model
from .dataset import load_splits, FaceDatasetGenerator


def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    _, _, test = load_splits(config)
    model_path = os.path.join(config['MODEL']['save_dir'], 'best_model.h5')
    model = load_model(model_path, compile=False)
    gen = FaceDatasetGenerator(test, config)

    y_true, y_pred = [], []
    for X, y in gen:
        p = model.predict(X)
        y_true.extend(y)
        y_pred.extend(p.flatten())

    print(f"MSE: {mean_squared_error(y_true, y_pred):.4f}, R²: {r2_score(y_true, y_pred):.4f}")


if __name__ == '__main__':
    main()