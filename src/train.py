import yaml
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from .dataset import load_splits, FaceDatasetGenerator
from .model import build_model


def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    os.makedirs(config['MODEL']['save_dir'], exist_ok=True)

    train, val, _ = load_splits(config)
    train_gen = FaceDatasetGenerator(train, config)
    val_gen = FaceDatasetGenerator(val, config)

    model = build_model(config, tab_dim=68 * 2 + 10)
    model.compile(optimizer=Adam(config['MODEL']['lr']), loss='mse', metrics=['mse'])

    ckpt_path = os.path.join(config['MODEL']['save_dir'], 'best_model.h5')
    cb = [
        ModelCheckpoint(ckpt_path, save_best_only=True),
        ReduceLROnPlateau(factor=0.5, patience=3),
        EarlyStopping(patience=5, restore_best_weights=True),
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=int(config['MODEL']['epochs']),
        callbacks=cb,
    )

    print('✅ Training complete. Best model saved.')


if __name__ == '__main__':
    main()