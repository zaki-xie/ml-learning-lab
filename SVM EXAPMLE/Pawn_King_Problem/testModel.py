from libsvm.svmutil import *
import os
import numpy as np


#############################
# 加载模型和测试数据进行测试

# 获取当前 py 文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 拼接数据文件的绝对路径
model_path = os.path.join(current_dir, "final_model.model")
xTesting_path = os.path.join(current_dir, "xTesting.npy")
yTesting_path = os.path.join(current_dir, "yTesting.npy")

# 加载模型
model = svm_load_model(model_path)
# 加载测试数据
xTesting = np.load(xTesting_path)
yTesting = np.load(yTesting_path)
print("加载测试数据，特征维度:" + str(xTesting.shape))
print("加载测试数据，标签维度:" + str(yTesting.shape))
# 在测试集上进行预测，启用概率输出
p_label, p_acc, p_val = svm_predict(yTesting, xTesting, model, '-b 1')
print("测试集预测完成。")
print("预测标签:", np.array(p_label))   # 模型对于测试集的预测标签，前面定义了draw和棋为+1，其余为-1，参考数据:[-1. -1. -1. ... -1.  1. -1.]

# 模型在测试集上的准确率[准确率, 均方误差, 相关系数的平方（衡量预测值与真实值的相关性）]  (99.37977099236642, 0.02480916030534351, 0.9341247848103147)
# 准确率Accuracy：测试集中预测正确的样本数 / 总样本数 × 100%。
# 均方差MSE：预测值与真实值之间差异的平方的平均值，反映了预测误差的大小。MES=1/n * Σ(预测值 - 真实值)²
#        衡量预测值与真实值的偏差大小。越接近 0 越好
# 相关系数的平方SCC： 衡量预测值与真实值之间线性关系的强度，取值范围为0到1，越接近1表示线性关系越强。
#                   R²= [Σ(预测值 - 预测值均值)(真实值 - 真实值均值)]² / [Σ(预测值 - 预测值均值)² * Σ(真实值 - 真实值均值)²]

print("预测准确率:", p_acc)

labels = model.get_labels()  # 获取模型的类别标签顺序
print("模型类别标签顺序:", labels)
# 每个样本属于各类别的概率值，shape=(样本数, 类别数)
# 在二分类时，每行通常是 [prob_negative, prob_positive]
# 打印出labels后可知，结构为[[样本1正类概率,样本1负类概率], [样本2正类概率,样本2负类概率], ...]
# [[1.00000000e-07 9.99999900e-01]
#  [1.02607786e-02 9.89739221e-01]
#  [1.00000000e-07 9.99999900e-01]
#  ...
#  [1.00000000e-07 9.99999900e-01]
#  [9.37231779e-01 6.27682214e-02]
#  [1.00000000e-07 9.99999900e-01]]
print("预测概率值:", np.array(p_val))   



# 检查 p_val 是否为空且每行至少有两个元素
# 非空检测、检测p_val中的每一行是否为列表或数组且长度大于1
if p_val and all(isinstance(row, (list, np.ndarray)) and len(row) > 1 for row in p_val):
    pos_index = labels.index(1)   # 找到 +1 的索引
    print("正类(+1)在标签中的索引:", pos_index)
    prob_pos = np.array([row[pos_index] for row in p_val])# 找到正类的概率对于标签顺序[1, -1]，正类索引为0
else:
    raise ValueError("p_val 为空或格式不正确，无法提取正类概率。")

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import mplcursors
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc

# 计算ROC曲线数据
# 输入yTesting真实标签，prob_pos为预测为正类的概率，pos_label=1表示正类标签为1
# 输出: fpr假正率, tpr真正率, thresholds阈值
# fpr=假正率(False Positive Rate)：将负类错误分类为正类的比例。计算公式为：FPR = FP / (FP + TN)
# tpr=真正率(True Positive Rate)：将正类正确分类为正类的比例，也称为召回率。计算公式为：TPR = TP / (TP + FN)
# thresholds=阈值：用于分类的不同阈值,当预测概率 ≥ 某个阈值时判为正类,随着阈值从高到低移动，你会得到一条从左下到右上的 ROC 曲线
fpr, tpr, thresholds = roc_curve(yTesting, prob_pos, pos_label=1)
# 计算AUC值
# AUC=Area Under the Curve，ROC曲线下的面积，衡量分类器性能的指标
# ROC曲线：横坐标为假正率FPR，纵坐标为真正率TPR，随着阈值变化绘制(FPR, TPR) 点的连线
# AUC=0.5:模型和随机猜测一样，没有区分能力
# AUC<0.5:模型表现比随机猜测还差(通常是正负类概率取反了)
# AUC=1.0:完美分类器，能完全区分正负类
roc_auc = auc(fpr, tpr)
print("AUC值:", roc_auc)

###################
# 本地交互式 ROC 曲线
plt.figure()    # 创建绘图窗口
# 以横坐标fpr，纵坐标tpr绘制ROC曲线，label中显示AUC值
line_Roc, = plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.5f})')
# 绘制一条对角线，表示随机分类器的性能
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # 随机分类器参考线
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")


# 添加交互功能
cursor = mplcursors.cursor(line_Roc, hover=True)

@cursor.connect("add")
def on_add(sel):
    idx = int(sel.index)   # 转成整数索引
    sel.annotation.set_text(
        f"FPR={fpr[idx]:.3f}\nTPR={tpr[idx]:.3f}\nThreshold={thresholds[idx]:.3f}"
    )
plt.show()

###################
# WEB 交互式 ROC 曲线
# fig = go.Figure()

# fig.add_trace(go.Scatter(
#     x=fpr, y=tpr,
#     mode='lines',
#     name=f'ROC curve (AUC = {roc_auc:.2f})',
#     line=dict(color='blue'),
#     hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<br>Threshold: %{text}',
#     text=[f"{thr:.3f}" for thr in thresholds]  # 鼠标悬停时显示阈值
# ))

# # 添加随机分类器参考线
# fig.add_trace(go.Scatter(
#     x=[0, 1], y=[0, 1],
#     mode='lines',
#     name='Random classifier',
#     line=dict(color='gray', dash='dash')
# ))

# fig.update_layout(
#     title="ROC Curve",
#     xaxis_title="False Positive Rate",
#     yaxis_title="True Positive Rate",
#     legend=dict(x=0.6, y=0.05)
# )

# fig.show()

