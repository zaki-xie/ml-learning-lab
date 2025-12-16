## 基础环境
Anaconda环境

## 配置指令
### 安装libsvm库
pip install -U libsvm-official
该库通过调用底层C函数实现，需要好像是C14.0.0+的环境，若上述libsvm安装失败，根据提示安装MSVC最新的环境即可


## 项目目录
-final_model.model 训练完成得到的最终模型 

-kekopt.data 兵王问题数据集

-testModel.py 训练结果测试

-trainModel.py 模型训练

-xTesting.npy 测试样本x

-yTesting.npy 测试样本y