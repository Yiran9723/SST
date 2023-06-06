# 大论文第一个方法（第二章）
基本框架来源于论文：STGAT: Modeling Spatial-Temporal Interactions for Human Trajectory Prediction；主要工作就是把原文的空间交互GAT改成了增强边图卷积。

## 评价指标
ADE（平均轨迹误差）和FDE（末点轨迹误差）

## 配置
* Python 3
* PyTorch (1.2)
* Matplotlib

## 数据集
NGSIM数据集（US101&I80）；

这个代码文件里的dataset是我处理过的数据集，具体在dataset文件夹里有说明。

## 运行
train.py训练模型；evaluation_model.py测试模型；draw_trajectory.py可以画轨迹图。