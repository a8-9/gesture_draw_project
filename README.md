

# 基于手势识别的交互式绘图系统


**项目开源地址：**

https://github.com/A8-9/gesture_draw_project.git
 

**组员分工**

 - **（A）**王紫艓****：手势样本采集、数据集预处理，`gesture_data.npz`数据集制作
  - **（B）殷世薇**：手部关键点算法、特征工程、分类模型训练，负责`utils.py`、`train_model.py`、`gesture_model.pkl`开发
  -  **（C）张颖**：项目Demo调试、程序录屏截图、README编写、答辩PPT制作 
  ## 一、运行环境配置
   ### 1.环境要求 Python版本：`3.8 ~ 3.10`
   ### 2.两种安装依赖方式
 
1）手动逐条安装 ：pip install opencv-python mediapipe numpy scikit-learn 
2）依托requirements.txt一键批量安装（项目标准化部署） 在项目

`根目录`打开CMD执行： pip install -r requirements.txt  

  ### 3.GitHub一键拉取完整项目（异地部署）:
git clone https://github.com/A8-9/gesture_draw_project.git 
cd gesture_draw_project  随后执行上方依赖安装命令

    
 ## 二、项目启动方式 

方式1：源码运行（开发调试） 进入项目`src,models,data`目录，目录栏唤起CMD，执行： `python dr1.py ` 
方式2：打包exe运行（免Python环境） 进入`dist`文件夹，双击`dr1.exe`直接启动程序  `注意：Windows用户名含中文时，exe大概率运行异常`

## 三、数据处理

### 1. 数据集采集

7 名受试者采集 5 类手势原始图片，原始数据集存放于`data/hand_dataset`（数据库较大，未上传GitHub）
### 2. 数据清洗与数据集划分

执行数据清洗脚本，自动校验、修正标签、划分训练 / 验证 / 测试集：`cd src python data_cleaning_pipeline.py`
-   原始总样本：7400 张
-   分组规则：7 人划分为：训练集 4 人、验证集 1 人、测试集 2 人
-   输出规整数据集：`data/dataset_final/[train/val/test]`（图片集不上传仓库）
-   特征提取后生成压缩数据集：`data/gesture_data.npz`（仓库内置）

### 3. 模型训练:`cd src python train_model.py`

   ## 四、数据集与模型说明 
 1. `gesture_data.npz`：项目数据集，存储标准化手部关键点特征、5分类手势标签，由实拍多组手势数据预处理生成；
   2. `gesture_model.pkl`：训练完成的机器学习分类模型，由`train_model.py`训练输出；
     3. `convert_images_to_features.py`：数据预处理脚本，实现原始手势图片 → 标准化特征数据集的批量转换。
   ## 五、源码文件说明
 dr1.py:
项目入口主程序：摄像头画面读取、mediapipe 手部关键点检测、手势推理、画布绘图交互全逻辑
utils.py:
通用工具类：21 点手部坐标提取、坐标归一化、特征预处理、辅助判定函数封装
train_model.py:
分类模型训练脚本，载入数据集训练并保存分类模型
convert_images_to_features.py:
原始图片批量提取特征、生成 npz 数据集
    
   
   ## 六、五种手势功能定义 
   1. **单指伸出**：画笔模式，指尖轨迹实时绘制线条
    2. **食指+中指（剪刀手）**：橡皮擦模式，擦除画布笔迹
     3. **OK手势**：循环切换画笔颜色 
     4. **握拳保持2~3秒**：一键清空画布全部内容 
     5. **全手掌张开保持2~3秒**：暂停/恢复绘图 
     
   ## 七、现存缺陷与优化方向 
 1. 手部距离摄像头＞40cm时，关键点易丢失，手势识别失效;
 2. 手指被遮挡会造成特征缺失，识别准确率明显下降；
 3. 暗光环境画面噪点大，关键点偏移，识别效果变差；
 4. 程序当前平均FPS≈13.3，帧率偏低，可通过降低摄像头分辨率提升运行速度。
  
   补充说明 `build/、dist/、__pycache__/`为打包、程序运行自动生成的临时文件夹，**不属于项目源码**，项目归档、上传GitHub时可直接删除。
  
