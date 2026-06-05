import cv2
import os
import shutil
import random
import numpy as np
import time

# 配置区域
SRC_PATH = os.path.abspath('hand_dataset')  # 自动获取绝对路径
DST_PATH = 'dataset_final'                  # 清洗与划分后的最终数据集路径
IMG_SIZE = (224, 224)                       # 图像尺寸归一化目标

# 按人划分比例
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

# 客观指标阈值
THRES_DIM_LIGHT = 65.0             
THRES_COMPLEX_BG = 0.06            

def analyze_and_correct_env(img_path, subjective_env):
    """ 利用计算机视觉客观指标审计并修正用户的主观环境标签 """
    try:
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None, None
        
    if img is None:
        return None, None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    objective_env = subjective_env
    if brightness < THRES_DIM_LIGHT:
        objective_env = "Dim_Light"
    elif edge_density > THRES_COMPLEX_BG and subjective_env != "Dim_Light":
        objective_env = "Complex_BG"
    else:
        if subjective_env == "Dim_Light" and brightness > (THRES_DIM_LIGHT + 20):
            objective_env = "Normal_Light"
            
    return img, objective_env

def save_image_to_dir(img, mode, gesture, name):
    save_path = os.path.join(DST_PATH, mode, gesture, name)
    _, img_encode = cv2.imencode('.jpg', img)
    img_encode.tofile(save_path)

def clean_and_split_dataset():
    if not os.path.exists(SRC_PATH):
        print(f"[错误] 找不到源文件夹: {SRC_PATH}")
        return
        
    gestures = [d for d in os.listdir(SRC_PATH) if os.path.isdir(os.path.join(SRC_PATH, d)) and d != 'failure_cases']
    if not gestures:
        print("[警告] 没有找到任何手势子文件夹！")
        return

    print("======== 1. 正在全局检索人员 ID 列表 ========")
    all_persons = set()
    for gesture in gestures:
        gesture_dir = os.path.join(SRC_PATH, gesture)
        p_dirs = [d for d in os.listdir(gesture_dir) if os.path.isdir(os.path.join(gesture_dir, d))]
        for p_dir in p_dirs:
            all_persons.add(p_dir)
            
    person_list = list(all_persons)
    random.shuffle(person_list)
    
    num_people = len(person_list)
    if num_people < 3:
        print(f"[数据过少警告] 总共只检测到 {num_people} 个人，无法完美划分三部曲，将退回强制分配。")
        train_people = person_list[:1]
        val_people = person_list[1:2]
        test_people = person_list[2:] if num_people == 3 else person_list[1:]
    else:
        train_end = max(1, int(num_people * TRAIN_RATIO))
        val_end = max(train_end + 1, train_end + int(num_people * VAL_RATIO))
        
        train_people = set(person_list[:train_end])
        val_people = set(person_list[train_end:val_end])
        test_people = set(person_list[val_end:])

    print(f"[人员划分报告] 总参与人数: {num_people}")
    print(f" -> 训练集(train) 包含人数: {len(train_people)} 人 ({train_people})")
    print(f" -> 验证集(val)   包含人数: {len(val_people)} 人 ({val_people})")
    print(f" -> 测试集(test)  包含人数: {len(test_people)} 人 ({test_people})")

    # 创建目标目录结构
    for mode in ['train', 'val', 'test']:
        for gesture in gestures:
            os.makedirs(os.path.join(DST_PATH, mode, gesture), exist_ok=True)
            
    print("\n======== 2. 开始执行数据审计与规范化流转 ========")
    correction_count = 0
    total_count = 0
    saved_count = 0 

    mode_counters = {'train': 0, 'val': 0, 'test': 0}

    for gesture in gestures:
        gesture_dir = os.path.join(SRC_PATH, gesture)
        p_dirs = [d for d in os.listdir(gesture_dir) if os.path.isdir(os.path.join(gesture_dir, d))]
        
        for pid in p_dirs:
            if pid in train_people:
                mode = 'train'
            elif pid in val_people:
                mode = 'val'
            else:
                mode = 'test'
                
            person_folder_path = os.path.join(gesture_dir, pid)
            all_files = [f for f in os.listdir(person_folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # 引入受试者内部的文件自增序号，避免重名
            for idx, file_name in enumerate(all_files):
                total_count += 1
                file_path = os.path.join(person_folder_path, file_name)
                
                parts = file_name.split('_')
                subjective_env = "_".join(parts[1:-1]) if len(parts) >= 3 else "Normal_Light"
                    
                img, objective_env = analyze_and_correct_env(file_path, subjective_env)
                if img is None:
                    print(f"[提示] 图片损坏或无法解析，已被清洗过滤: {file_path}")
                    continue
                    
                if subjective_env != objective_env:
                    correction_count += 1
                    
                # 统一尺寸
                resized_img = cv2.resize(img, IMG_SIZE)
                
                # 文件名格式：手势_环境标签_人头ID_序号.jpg
                # 去除pid中的空格，防止文件名中出现非必要空格
                clean_pid = pid.replace(" ", "")
                new_name = f"{gesture}_{objective_env}_{clean_pid}_{idx}.jpg"
                
                # 保存到对应归属人的数据集
                save_image_to_dir(resized_img, mode, gesture, new_name)
                mode_counters[mode] += 1
                saved_count += 1
            
        print(f"[-] 手势 [{gesture}] 归流审计完成。")

    print("\n======== 流水线运行结束 ========")
    print(f"[报告数据] 共检索到原始样本: {total_count} 张")
    print(f"[报告数据] 自动因图片损坏清洗过滤: {total_count - saved_count} 张")
    print(f"[报告数据] 自动纠正标签偏误: {correction_count} 张")
    print(f"[流转统计] 最终保存图片分布 -> 训练集: {mode_counters['train']}张 | 验证集: {mode_counters['val']}张 | 测试集: {mode_counters['test']}张")
    print(f"[流转统计] 最终实际写入总数: {saved_count} 张")
    print(f"[系统提示] 规整后的 Cross-Subject 数据集已保存在: ./{DST_PATH}")

if __name__ == "__main__":
    random.seed(42)
    clean_and_split_dataset()