from libsvm.svmutil import *
import os
import numpy as np

########################################################################
# 1、提取数据集：

# 获取当前 py 文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 拼接数据文件的绝对路径
file_path = os.path.join(current_dir, "krkopt.data")
# 打开数据文件,默认系统GB2312有乱码
with open(file_path, "r",encoding="utf-8-sig") as f:
    lines = f.readlines()

xapp = []  # 特征矩阵，每个样本是一个长度为6的向量
yapp = []  # 标签向量

for line in lines:
    line = line.strip()  # 去掉换行符
    if len(line) > 10:   # 过滤掉空行或不完整行
        parts = line.split(",")  # 按逗号分割
        # print(parts)

        # 提取特征：字母转数字 (a=1,b=2,...)，数字字符转数值
        vec = np.zeros(6, dtype=int)
        vec[0] = ord(parts[0]) - 96   # 第一个字母 a→1, b→2...
        vec[1] = int(parts[1])        # 第一个数字
        vec[2] = ord(parts[2]) - 96   # 第二个字母
        vec[3] = int(parts[3])        # 第二个数字
        vec[4] = ord(parts[4]) - 96   # 第三个字母
        vec[5] = int(parts[5])        # 第三个数字

        xapp.append(vec)
        # print(vec)
        # print(xapp)

    # 标签：如果第7个字符是 'draw' → 正类(+1)，否则负类(-1)
    if parts[6] == 'draw':
        yapp.append(1)
    else:
        yapp.append(-1)
#print(xapp)

# 确保是二维数组 (M,6)，每行一个样本
xapp = np.array(xapp)
yapp = np.array(yapp)
# 强制标签为 {1, -1}
yapp = np.where(yapp == 1, 1, -1)
print("数据集大小:")   
print("样本个数" +str(len(xapp)))
print("标签个数" +str(len(yapp)))

########################################################################
# 2、随机提取训练样本和测试样本
# 总样本数和训练样本数

numberOfSamples = len(xapp)   # 样本数
# 生成随机排列索引,用于提取随机样本作为训练集
indexList = np.random.permutation(numberOfSamples)
print("随机排列索引:" + str(indexList))
print("随机排列索引长:" + str(len(indexList)))

numberOfSamplesForTraining = 5000

# 训练集索引（前5000个随机索引）
train_idx = indexList[:numberOfSamplesForTraining]
# 测试集索引（剩余的随机索引）
test_idx = indexList[numberOfSamplesForTraining:]

# 提取训练集和测试集
xTraining = xapp[train_idx, :]   # (5000, 6)
yTraining = yapp[train_idx]      # (5000,)
print("训练集特征维度:" + str(xTraining.shape))
print("训练集标签维度:" + str(yTraining.shape))

xTesting = xapp[test_idx, :]     # (M-5000, 6)
yTesting = yapp[test_idx]        # (M-5000,)
print("测试集特征维度:" + str(xTesting.shape))
print("测试集标签维度:" + str(yTesting.shape))

########################################################################
# 3、数据归一化（对训练集计算均值与标准差，然后用相同参数对测试集进行归一化）
avgX = np.mean(xTraining, axis=0)
stdX = np.std(xTraining, axis=0)
print("特征均值:" + str(avgX))
print("特征标准差:" + str(stdX))
# 避免除以零的情况
stdX[stdX == 0] = 1
xTraining = (xTraining - avgX) / stdX
xTesting = (xTesting - avgX) / stdX

########################################################################
# 4、大范围搜索
# SVM 高斯核（RBF）参数搜索
# 核函数 K(x1,x2) = exp(-||x1-x2||^2 / gamma)
# 目标：通过交叉验证（5 折）找到使识别率最高的 C 和 gamma
# 首先在粗尺度上搜索 C 和 gamma（参考 "A practical Guide to Support Vector Classification"）

CScale = [-5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15] # C = 2.^CScale
gammaScale = [-15, -13, -11, -9, -7, -5, -3, -1, 1, 3] #gamma = 2.^gammaScale
C_List = [2**c for c in CScale]
gamma_List = [2**g for g in gammaScale]
print("C列表:" + str(C_List))
print("gamma列表:" + str(gamma_List))

# 训练查找最佳参数
maxRecognitionRate = 0
bestC = None
bestGamma = None
bestCIndex = -1
bestGammaIndex = -1

for i, C in enumerate(C_List):
    for j, gamma in enumerate(gamma_List):
        # 设置 SVM 参数
        param = f'-t 2 -c {C} -g {gamma} -v 5 -q'#-p quiet 模式，屏蔽输出
        # 交叉验证
        recognitionRate = svm_train(yTraining.tolist(), xTraining.tolist(), param)
        print(f"C={C} (index={i}), gamma={gamma} (index={j}), 识别率={recognitionRate}%")
        # 更新最佳参数及其索引
        if recognitionRate > maxRecognitionRate:
            maxRecognitionRate = recognitionRate
            bestC = C
            bestGamma = gamma
            bestCIndex = i
            bestGammaIndex = j

print(f"最佳参数: C={bestC} (index={bestCIndex}), gamma={bestGamma} (index={bestGammaIndex}), 识别率={maxRecognitionRate}%")
# for C in C_List:
#     for gamma in gamma_List:
#         # 设置 SVM 参数
#         param = f'-t 2 -c {C} -g {gamma} -v 5'
#         # 交叉验证
#         recognitionRate = svm_train(yTraining.tolist(), xTraining.tolist(), param)
#         print("C={}, gamma={}, 识别率={}%".format(C, gamma, recognitionRate))
#         # 更新最佳参数
#         if recognitionRate > maxRecognitionRate:
#             maxRecognitionRate = recognitionRate
#             bestC = C
#             bestGamma = gamma
# print("最佳参数: C={}, gamma={}, 识别率={}%" .format(bestC, bestGamma, maxRecognitionRate))


########################################################################
# 5、根据搜索结果进行精细尺度搜索
n = 10  # 插值点数

# 在最佳刻度的前后插值，生成更细的刻度范围
minCScale = 0.5 * (CScale[max(0, bestCIndex-1)] + CScale[bestCIndex])
maxCScale = 0.5 * (CScale[min(len(CScale)-1, bestCIndex+1)] + CScale[bestCIndex])
newCScale = np.linspace(minCScale, maxCScale, n)    # 在该范围内生成 n 个均匀分布的刻度值
newC = [2**c for c in newCScale]

minGammaScale = 0.5 * (gammaScale[max(0, bestGammaIndex-1)] + gammaScale[bestGammaIndex])
maxGammaScale = 0.5 * (gammaScale[min(len(gammaScale)-1, bestGammaIndex+1)] + gammaScale[bestGammaIndex])
newGammaScale = np.linspace(minGammaScale, maxGammaScale, n)
newGamma = [2**g for g in newGammaScale]

print("精细搜索 C 列表:", newC)
print("精细搜索 gamma 列表:", newGamma)

# 精细搜索
maxRecognitionRate_fine = 0
bestC_fine = None
bestGamma_fine = None
bestCIndex_fine = -1
bestGammaIndex_fine = -1

for i, C in enumerate(newC):
    for j, gamma in enumerate(newGamma):
        # 设置 SVM 参数
        param = f'-t 2 -c {C} -g {gamma} -v 5 -q'#-p quiet 模式，屏蔽输出
        # 交叉验证
        recognitionRate = svm_train(yTraining.tolist(), xTraining.tolist(), param)
        print(f"[精细搜索] C={C} (index={i}), gamma={gamma} (index={j}), 识别率={recognitionRate}%")
        # 更新最佳参数及其索引
        if recognitionRate > maxRecognitionRate_fine:
            maxRecognitionRate_fine = recognitionRate
            bestC_fine = C
            bestGamma_fine = gamma
            bestCIndex_fine = i
            bestGammaIndex_fine = j
print(f"[精细搜索] 最佳参数: C={bestC_fine} (index={bestCIndex_fine}), gamma={bestGamma_fine} (index={bestGammaIndex_fine}), 识别率={maxRecognitionRate_fine}%")

# 使用精细搜索得到的最佳参数
finalC = bestC_fine
finalGamma = bestGamma_fine

########################################################################
# 6、使用最佳参数在整个训练集上训练最终模型，并保存模型和测试数据
# 设置最终训练参数（不使用 -v）
param = f'-t 2 -c {finalC} -g {finalGamma} -b 1'#-b 1 启用概率估计,不启用-v交叉验证

# 在整个训练集上训练模型
model = svm_train(yTraining.tolist(), xTraining.tolist(), param)



# 拼接数据文件的绝对路径
model_save_path = os.path.join(current_dir, "final_model.model")
xTesting_save_path = os.path.join(current_dir, "xTesting.npy")
yTesting_save_path = os.path.join(current_dir, "yTesting.npy")

# 保存模型到文件
svm_save_model(model_save_path, model)
# 保存测试数据到文件（用 numpy 保存）
np.save(xTesting_save_path, xTesting)
np.save(yTesting_save_path, yTesting)
print("最终模型和测试数据已保存到文件。")

