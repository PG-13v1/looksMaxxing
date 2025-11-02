import os, cv2, dlib, mediapipe as mp, numpy as np


class LandmarkExtractor:
  def __init__(self, predictor_path=None):
   self.detector = dlib.get_frontal_face_detector()
   self.predictor = dlib.shape_predictor(predictor_path) if predictor_path and os.path.exists(predictor_path) else None
   self.mp_face = mp.solutions.face_mesh
   self.mp_mesh = self.mp_face.FaceMesh(static_image_mode=True, refine_landmarks=False, max_num_faces=1)


  def _dlib_landmarks(self, img_gray):
   rects = self.detector(img_gray, 1)
   if len(rects) == 0: return None
   shape = self.predictor(img_gray, rects[0])
   return np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype=np.int32)


  def _mediapipe_landmarks(self, img_rgb):
   res = self.mp_mesh.process(img_rgb)
   if not res.multi_face_landmarks: return None
   lm = res.multi_face_landmarks[0].landmark
   h, w = img_rgb.shape[:2] 
   pts = np.array([(int(p.x*w), int(p.y*h)) for p in lm])
   if pts.shape[0] < 68:
    padded = np.zeros((68, 2), dtype=np.int32)
    padded[:pts.shape[0]] = pts
    padded[pts.shape[0]:] = pts[-1]
    return padded
   return pts[:68]


  def extract(self, img_bgr):
   img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
   if self.predictor is not None:
    try:
        lm = self._dlib_landmarks(img_gray)
        if lm is not None:
            return lm
    except:
        pass
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return self._mediapipe_landmarks(img_rgb)