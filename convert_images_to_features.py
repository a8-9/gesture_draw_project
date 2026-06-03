import cv2
import numpy as np
import os
import argparse
import json
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Dict

import mediapipe as mp
from mediapipe.tasks.python import vision

from utils import HandDetectorManager, extract_features, HandConfig

LABEL_MAP = {
    'open_hand': 0,
    'fist': 1,
    'point': 2,
    'ok_sign': 3,
    'scissors': 4,
}

_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def detect_hand_static(img, detector):
    if img is None or img.size == 0:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    try:
        result = detector.detect(mp_image)
    except Exception:
        return None
    return result.hand_landmarks[0] if result.hand_landmarks else None

def process_folder(root_path: str, label_map: Dict[str, int] = None, output_file: str = "gesture_data.npz") -> Tuple[np.ndarray, np.ndarray]:
    label_map = label_map or LABEL_MAP
    root = Path(root_path)
    
    config = HandConfig(running_mode=vision.RunningMode.IMAGE)
    detector = HandDetectorManager.get_detector(config)
    
    all_X, all_y = [], []
    stats = {'total': 0, 'success': 0, 'failed': 0, 'per_gesture': {}}
    
    for split in ['train', 'val', 'test']:
        split_path = root / split
        if not split_path.is_dir():
            print(f"⚠️ 跳过不存在的目录: {split_path}")
            continue
            
        for gesture_name, label in label_map.items():
            gesture_path = split_path / gesture_name
            if not gesture_path.is_dir():
                continue
                
            img_files = [f for f in gesture_path.glob("*") if f.suffix.lower() in _IMAGE_EXTS]
            if not img_files:
                continue
            
            key = f"{split}/{gesture_name}"
            stats['per_gesture'][key] = {'total': len(img_files), 'success': 0}
                
            print(f"📁 处理 {key}: {len(img_files)} 张图片")
            
            for img_file in tqdm(img_files, desc=gesture_name):
                img = cv2.imread(str(img_file))
                stats['total'] += 1
                
                if img is None:
                    stats['failed'] += 1
                    continue
                    
                landmarks = detect_hand_static(img, detector)
                if landmarks is not None:
                    feat = extract_features(landmarks, enhanced=True)
                    all_X.append(feat)
                    all_y.append(label)
                    stats['success'] += 1
                    stats['per_gesture'][key]['success'] += 1
                else:
                    stats['failed'] += 1
    
    with open('processing_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    if len(all_X) == 0:
        raise ValueError("未提取到任何有效样本，请检查图片质量")
    
    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int32)
    
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    
    np.savez(output_file, X=X, y=y)
    print(f"\n✅ 保存 {len(X)} 个样本到 {output_file}，特征维度: {X.shape[1]}")
    print(f"📊 标签分布: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"📈 处理统计: 成功{stats['success']}/总计{stats['total']}")
    
    return X, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将手势图片数据集转换为特征向量（增强版）")
    parser.add_argument("dataset_root", help="数据集根目录路径")
    parser.add_argument("-o", "--output", default="gesture_data.npz", help="输出文件名")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_root):
        print(f"❌ 错误：路径不存在 - {args.dataset_root}")
        exit(1)
    
    process_folder(args.dataset_root, output_file=args.output)