import numpy as np


def euclid(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def compute_symmetry(landmarks):
    xs = landmarks[:, 0]
    midx = (xs[27] + xs[30]) / 2.0 if landmarks.shape[0] > 30 else np.mean(xs)
    face_w = max(xs) - min(xs) + 1e-6
    diffs = []
    for l, r in zip(range(0, 14), range(16, 2, -1)):
        dl = abs(landmarks[l, 0] - midx) / face_w
        dr = abs(landmarks[r, 0] - midx) / face_w
        diffs.append(abs(dl - dr))
    return np.array(diffs)


def compute_golden_features(landmarks):
    ys, xs = landmarks[:, 1], landmarks[:, 0]
    face_h = max(ys) - min(ys) + 1e-6
    face_w = max(xs) - min(xs) + 1e-6
    feats = [face_h / face_w]
    try:
        nose_len = euclid(landmarks[27], landmarks[33])
        mouth_w = euclid(landmarks[48], landmarks[54])
        feats.append(nose_len / mouth_w)
        eye_w = (euclid(landmarks[36], landmarks[39]) + euclid(landmarks[42], landmarks[45])) / (2 * face_w)
        feats.append(eye_w)
    except Exception:
        feats.append(0)
    return np.array(feats)


def landmarks_to_feature_vector(landmarks):
    return np.concatenate([compute_golden_features(landmarks),
                           compute_symmetry(landmarks),
                           landmarks.flatten() / 1000.0])