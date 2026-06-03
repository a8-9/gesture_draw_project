import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import os
import json
import argparse
from datetime import datetime

DATA_FILE = "gesture_data.npz"

def load_data(test_size: float = 0.2):
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found")
    
    data = np.load(DATA_FILE)
    X, y = data['X'], data['y']
    data.close()
    
    print(f"📊 总样本: {len(X)}, 特征维度: {X.shape[1]}")
    print(f"标签分布: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name: str):
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    # 预测并分析混淆
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"🤖 {model_name}")
    print(f"   测试准确率: {acc:.4f}")
    print(f"   交叉验证: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"\n混淆矩阵:")
    print(cm)
    print(f"\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=["Palm", "Fist", "Point", "OK", "Scissors"]))
    
    return {
        'name': model_name,
        'test_acc': acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'model': model
    }

def train_and_compare(X_train, X_test, y_train, y_test):
    results = []
    
    # 随机森林（重点关注Point和Scissors的区分）
    rf = RandomForestClassifier(
        n_estimators=500, 
        max_depth=25,
        min_samples_split=3,
        random_state=42, 
        n_jobs=-1, 
        class_weight='balanced'  # 自动平衡类别
    )
    results.append(evaluate_model(rf, X_train, X_test, y_train, y_test, "RandomForest"))
    
    # SVM
    svm_pipe = Pipeline([
        ('scaler', StandardScaler()), 
        ('svm', SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42))
    ])
    results.append(evaluate_model(svm_pipe, X_train, X_test, y_train, y_test, "SVM"))
    
    # KNN
    knn_pipe = Pipeline([
        ('scaler', StandardScaler()), 
        ('knn', KNeighborsClassifier(n_neighbors=7, weights='distance'))
    ])
    results.append(evaluate_model(knn_pipe, X_train, X_test, y_train, y_test, "KNN"))
    
    best = max(results, key=lambda x: x['cv_mean'])
    print(f"\n🏆 最佳模型: {best['name']} (CV: {best['cv_mean']:.4f})")
    
    return best['model'], results

def save_model(model, filename: str = "gesture_model.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    
    meta = {
        'saved_at': datetime.now().isoformat(),
        'model_type': type(model).__name__,
    }
    with open(filename.replace('.pkl', '_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n💾 模型已保存: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="gesture_model.pkl", help="输出文件名")
    args = parser.parse_args()
    
    X_train, X_test, y_train, y_test = load_data()
    best_model, all_results = train_and_compare(X_train, X_test, y_train, y_test)
    save_model(best_model, args.output)