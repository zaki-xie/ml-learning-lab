import os
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split

def load_krk_dataset(
    filename="krkopt.data",
    ratioTraining=0.4,
    ratioValidation=0.1,
    ratioTesting=0.5
):
    """
    加载 KRK 国际象棋数据集，并按比例划分训练集、验证集、测试集。
    返回：
        xTrain, yTrain, xVal, yVal, xTest, yTest
    """

    # -----------------------------
    # 1. 读取数据文件
    # -----------------------------
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, filename)

    with open(file_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    # 数据集大小固定为 28056
    xapp = np.zeros((28056, 6), dtype=float)
    yapp = np.zeros((28056, 2), dtype=float)

    index = 0

    # -----------------------------
    # 2. 解析每一行数据
    # -----------------------------
    for line in lines:
        line = line.strip()
        if len(line) > 10:  # 有效行
            parts = line.split(",")

            # 特征向量
            vec = np.zeros(6, dtype=int)
            vec[0] = ord(parts[0]) - 96
            vec[1] = int(parts[1])
            vec[2] = ord(parts[2]) - 96
            vec[3] = int(parts[3])
            vec[4] = ord(parts[4]) - 96
            vec[5] = int(parts[5])

            xapp[index] = vec

            # 标签（独热编码）
            if parts[6] == "draw":
                yapp[index] = [1, 0]
            else:
                yapp[index] = [0, 1]

            index += 1

    # -----------------------------
    # 3. 数据集划分
    # -----------------------------
    # 举例：第一次划分：训练集 40%，剩余 60%
    xTrain, xRemain, yTrain, yRemain = train_test_split(
        xapp, yapp,
        test_size=1 - ratioTraining,
        random_state=0
    )

    # 举例：第二次划分：从剩余 60% 中划分验证集 10%，测试集 50%
    xVal, xTest, yVal, yTest = train_test_split(
        xRemain, yRemain,
        test_size=ratioTesting / (ratioTesting + ratioValidation),
        random_state=0
    )
    # -----------------------------
    # 4. 数据集归一化
    # -----------------------------
    scaler = StandardScaler(copy=False)
    scaler.fit(xTrain)          # 计算训练集的均值和标准差
    scaler.transform(xTrain)    # 标准化训练集
    scaler.transform(xVal)      # 标准化验证集
    scaler.transform(xTest)     # 标准化测试集
    


    return xTrain, yTrain, xVal, yVal, xTest, yTest

if __name__ == "__main__":
    xTrain, yTrain, xVal, yVal, xTest, yTest = load_krk_dataset()

    print("训练集:", len(xTrain))
    print("验证集:", len(xVal))
    print("测试集:", len(xTest))
    print("训练集标签示例:", yTrain[:5])