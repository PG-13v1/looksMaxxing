import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
from tqdm import tqdm
from .landmark_extractor import LandmarkExtractor
from .features import landmarks_to_feature_vector


class FaceDatasetGenerator(Sequence):
    def __init__(self, df, config):
        self.df = df.reset_index(drop=True)
        self.bs = int(config['MODEL']['batch_size'])
        self.img_size = tuple(config['DATA']['image_size'])
        self.img_base = config['DATA']['image_base_path']
        self.extractor = LandmarkExtractor(config['DLIB']['predictor_path'])
        self.use_landmarks = bool(config['FEATURES']['use_landmark_features'])
        self.use_images = bool(config['FEATURES']['use_image_features'])

        # precompute a zero feature vector (assumes 68 landmarks if missing)
        try:
            self._zero_feat = landmarks_to_feature_vector(np.zeros((68, 2)))
        except Exception:
            # fallback to a reasonable default length
            self._zero_feat = np.zeros(68 * 2 + 16)

    def __len__(self):
        return int(np.ceil(len(self.df) / self.bs))

    def __getitem__(self, idx):
        batch = self.df.iloc[idx * self.bs:(idx + 1) * self.bs]
        imgs, feats, ys = [], [], []
        for _, row in batch.iterrows():
            path = os.path.join(self.img_base, str(row['image_path']))
            img = cv2.imread(path)
            if img is None:
                # skip missing images (or you may want to raise)
                continue
            img = cv2.resize(img, self.img_size)
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0)
            lm = self.extractor.extract(img)
            feats.append(landmarks_to_feature_vector(lm) if lm is not None else self._zero_feat)
            ys.append(float(row['score']))

        X_img = np.array(imgs)
        X_tab = np.array(feats)
        y = np.array(ys)

        if self.use_images and self.use_landmarks:
            return {'image_input': X_img, 'tab_input': X_tab}, y
        return (X_img if self.use_images else X_tab), y


def load_splits(config):
    df = pd.read_csv(config['DATA']['labels_csv'])
    train, temp = train_test_split(df, test_size=0.25, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    return train, val, test