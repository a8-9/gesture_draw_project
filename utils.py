import cv2
import numpy as np
import sys
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from dataclasses import dataclass
from typing import List, Optional, Tuple


def get_resource_path(relative_path):
    """获取资源文件绝对路径（兼容开发环境和打包环境）"""
    if getattr(sys, 'frozen', False):
        if hasattr(sys, '_MEIPASS'):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    full_path = os.path.join(base_path, relative_path)
    
    # 调试：打印路径信息
    print(f"[Resource] Looking for: {relative_path}")
    print(f"[Resource] Full path: {full_path}")
    print(f"[Resource] Exists: {os.path.exists(full_path)}")
    
    return full_path


# ============ 配置类 ============
@dataclass
class HandConfig:
    model_name: str = "hand_landmarker.task"
    num_hands: int = 1
    running_mode: vision.RunningMode = vision.RunningMode.VIDEO
    finger_thresholds: dict = None
    
    def __post_init__(self):
        if self.finger_thresholds is None:
            self.finger_thresholds = {
                'index': 1.2, 'middle': 1.2, 'ring': 1.2, 
                'pinky': 1.2, 'thumb': 1.1
            }


# ============ 多模式 Detector 管理器 ============
class HandDetectorManager:
    _detectors = {}
    
    @classmethod
    def get_detector(cls, config: HandConfig = None):
        config = config or HandConfig()
        key = config.running_mode.name
        
        if key not in cls._detectors or cls._detectors[key] is None:
            model_path = get_resource_path(config.model_name)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model not found: {model_path}\n"
                    f"Current dir: {os.getcwd()}\n"
                    f"Contents: {os.listdir(os.path.dirname(model_path)) if os.path.exists(os.path.dirname(model_path)) else 'N/A'}"
                )
            
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=config.num_hands,
                running_mode=config.running_mode
            )
            cls._detectors[key] = vision.HandLandmarker.create_from_options(options)
        
        return cls._detectors[key]
    
    @classmethod
    def reset(cls):
        cls._detectors.clear()


# ============ 常量 ============
GESTURE_NAMES = ["Palm", "Fist", "Point", "OK", "Scissors"]
GESTURE_LABELS = {name: i for i, name in enumerate(GESTURE_NAMES)}

TIP_INDICES = {'index': 8, 'middle': 12, 'ring': 16, 'pinky': 20, 'thumb': 4}
PIP_INDICES = {'index': 6, 'middle': 10, 'ring': 14, 'pinky': 18, 'thumb': 3}

def is_finger_extended(landmarks, finger_name: str, wrist_idx: int = 0, threshold: Optional[float] = None) -> bool:
    config = HandConfig()
    threshold = threshold or config.finger_thresholds.get(finger_name, 1.2)
    
    tip_idx = TIP_INDICES[finger_name]
    pip_idx = PIP_INDICES[finger_name]
    
    wrist = np.array([landmarks[wrist_idx].x, landmarks[wrist_idx].y])
    tip = np.array([landmarks[tip_idx].x, landmarks[tip_idx].y])
    pip = np.array([landmarks[pip_idx].x, landmarks[pip_idx].y])
    
    dist_tip = np.linalg.norm(tip - wrist)
    dist_pip = np.linalg.norm(pip - wrist)
    
    return dist_pip > 1e-6 and (dist_tip / dist_pip) > threshold

def get_finger_states(landmarks, custom_thresholds: Optional[dict] = None) -> List[int]:
    thresholds = custom_thresholds or {}
    return [
        1 if is_finger_extended(landmarks, name, threshold=thresholds.get(name)) else 0
        for name in ['index', 'middle', 'ring', 'pinky', 'thumb']
    ]

def count_extended_fingers(landmarks) -> int:
    return sum(get_finger_states(landmarks))

def _get_landmark_array(landmarks, idx: int) -> np.ndarray:
    return np.array([landmarks[idx].x, landmarks[idx].y])

def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.arccos(np.clip(cos, -1.0, 1.0))

def extract_features(landmarks, include_z: bool = False, enhanced: bool = True) -> np.ndarray:
    coords = []
    for lm in landmarks:
        coords.extend([lm.x, lm.y])
        if include_z and hasattr(lm, 'z'):
            coords.append(lm.z)
    
    finger_states = get_finger_states(landmarks)
    features = coords + finger_states + [sum(finger_states)]
    
    if enhanced:
        index_tip = _get_landmark_array(landmarks, 8)
        middle_tip = _get_landmark_array(landmarks, 12)
        index_middle_dist = np.linalg.norm(index_tip - middle_tip)
        features.append(index_middle_dist)
        
        index_pip = _get_landmark_array(landmarks, 6)
        index_middle_pip_dist = np.linalg.norm(index_pip - middle_tip)
        features.append(index_middle_pip_dist)
        
        wrist = _get_landmark_array(landmarks, 0)
        v_index = index_tip - wrist
        v_middle = middle_tip - wrist
        spread_angle = _angle_between(v_index, v_middle)
        features.append(spread_angle)
        
        middle_pip = _get_landmark_array(landmarks, 10)
        middle_len = np.linalg.norm(middle_tip - middle_pip)
        index_len = np.linalg.norm(index_tip - index_pip)
        middle_relative_len = middle_len / (index_len + 1e-6)
        features.append(middle_relative_len)
        
        features.append(abs(index_tip[1] - middle_tip[1]))
    
    return np.array(features, dtype=np.float32)

def get_hand_landmarks(frame, timestamp_ms: int = 0, config: HandConfig = None):
    if frame is None or frame.size == 0:
        return None, False
    
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detector = HandDetectorManager.get_detector(config)
        result = detector.detect_for_video(mp_image, timestamp_ms)
        
        if result.hand_landmarks:
            return result.hand_landmarks[0], True
    except Exception as e:
        print(f"[Error] Hand detection failed: {e}")
    
    return None, False