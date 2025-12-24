import numpy as np

class Option:
    def __init__(self):
        """
        神经网络训练选项类。
        包含训练过程中的批量大小和迭代次数等参数。

        变量说明
        ----------
        batch_size : int
            每次迭代的批量大小。
        iteration : int
            迭代次数。
        """
        self.batch_size = 0# 每次迭代的批量大小
        self.iteration = 0# 迭代次数

class NeuralNetworksClass:
    def __init__(self, **arg):
        """
        初始化一个可配置的全连接神经网络（Fully‑Connected Neural Network）。
        支持多层结构、不同激活函数、多种优化算法、稀疏自编码器、
        批量归一化（Batch Normalization）以及多种损失函数。

        参数说明
        ----------
        layer : list[int]
            网络结构列表，每个元素表示对应层的神经元个数。
            例如 [6, 10, 10, 2] 表示：
            输入层 6 → 隐藏层 10 → 隐藏层 10 → 输出层 2。

        active_function : str, 默认 'sigmoid'
            隐藏层激活函数。
            可选：'relu'、'sigmoid'、'tanh'、'leaky_relu'。

        output_function : str, 默认 'sigmoid'
            输出层激活函数。
            可选：'sigmoid'、'softmax'、'linear'。
            若 objective_function='Cross Entropy'，将自动切换为 'softmax'。

        learning_rate : float, 默认 1.5
            学习率，用于控制梯度下降的步长。

        weight_decay : float, 默认 0
            L2 正则化系数，用于抑制过拟合。

        cost : dict, 默认 {}
            成本敏感学习参数。
            例如 {'0':1, '1':5} 表示类别 1 的误分类代价是类别 0 的 5 倍。

        encoder : int (0 或 1), 默认 0
            是否启用自编码器结构（输入 = 输出）。

        sparsity : float, 默认 0.03
            稀疏自编码器的目标稀疏度 ρ。

        beta : float, 默认 3
            稀疏性惩罚项的权重，用于 KL 散度稀疏约束。

        batch_normalization : int (0 或 1), 默认 0
            是否启用 Batch Normalization。

        grad_squared : int (0 或 1), 默认 0
            是否启用梯度平方累积（用于 AdaGrad / RMSProp / Adam）。

        r : float, 默认 0
            梯度平方累积的衰减率（如 RMSProp/Adam 的 β2）。

        optimization_method : str, 默认 'normal'
            优化算法选择。
            可选：'normal'、'Momentum'、'AdaGrad'、'RMSProp'、'Adam'。

        objective_function : str, 默认 'MSE'
            损失函数类型。
            可选：'MSE'（均方误差）、'Cross Entropy'（交叉熵）。

        说明
        ----
        - 初始化流程先加载默认参数，再用用户传入的参数覆盖默认值。
        - 本方法仅完成网络结构与超参数的配置；
        权重矩阵、偏置、动量项、BN 参数等会在后续步骤中按层初始化。
        - 所有网络参数（W、b、梯度、动量、BN 参数等）均使用字典按层编号存储。

        """
        init = {'layer':[],# 网络层结构列表，如[2,3,1],每个元素表示每层神经元个数
                'active_function':'sigmoid', #激活函数选择'relu','sigmoid','tanh','leaky_relu'
                'output_function':'sigmoid', #输出层激活函数选择'sigmoid','softmax','linear'
                'learning_rate':1.5, #学习率
                'weight_decay':0,#L2正则化参数
                'cost':{}, #成本敏感学习参数字典结构，如{'0':1,'1':5}表示类别0的样本误分类代价为1，类别1的样本误分类代价为5
                'encoder':0,#是否使用编码器结构，1表示是，0表示否
                'sparsity':0.03,#稀疏性参数
                'beta':3,#稀疏性惩罚项权重
                'batch_normalization':0,#是否使用批量归一化，1表示是，0表示否
                'grad_squared':0,#是否使用梯度平方累积，1表示是，0表示否
                'r':0,#梯度平方累积的衰减率
                'optimization_method':'normal',#优化方法选择'normal','Momentum','AdaGrad','RMSProp','Adam'
                'objective_function':'MSE'#目标函数选择'MSE','Cross Entropy'
               }
        


        # 初始化参数
        param = dict() #字典结构实现参数列表
        param.update(init)#用默认值初始化
        param.update(arg)#用用户传入的值覆盖默认值

        # 初始化网络参数
        self.size = param['layer'] #取出字典的值初始化网络参数
        self.depth = len(self.size)# 网络深度
        self.active_function = param['active_function']# 激活函数
        self.output_function = param['output_function']# 输出层激活函数
        self.learning_rate = param['learning_rate']# 学习率
        self.weight_decay = param['weight_decay']# L2 正则化参数
        self.encoder = param['encoder']# 是否使用编码器结构
        self.sparsity = param['sparsity']# 稀疏性参数
        self.beta = param['beta']# 稀疏性惩罚项权重
        self.cost = param['cost']# 成本敏感学习参数
        self.batch_normalization = param['batch_normalization']# 是否使用批量归一化
        self.grad_squared = param['grad_squared']# 是否使用梯度平方累积
        self.r = param['r']# 梯度平方累积的衰减率
        self.optimization_method = param['optimization_method']# 优化方法
        self.objective_function = param['objective_function']# 目标函数
        self.a = dict()# 激活值字典

        # 若使用交叉熵损失函数，强制输出层激活函数为 softmax
        if self.objective_function == 'Cross Entropy':
            self.output_function = 'softmax'

        # 初始化权重矩阵、偏置、动量项、BN 参数等
        self.W = dict(); self.b = dict(); self.vW = dict(); self.vb = dict() #python必须要先初始化字典才能用
        self.rW = dict(); self.rb = dict(); self.sW = dict(); self.sb = dict() #注意要单独初始化，否则它们以后也一直是一样的
        self.E = dict(); self.S = dict(); self.Gamma = dict(); self.Beta = dict()
        self.vGamma = dict(); self.rGamma = dict(); self.vBeta = dict(); self.rBeta = dict(); 
        self.sGamma = dict(); self.sBeta = dict(); self.W_grad = dict(); self.b_grad = dict(); self.theta = dict()
        self.Gamma_grad = dict(); self.Beta_grad = dict()
        
        # 初始化每层的参数
        for k in range(self.depth - 1):
            width = self.size[k]    #神经元数量，权重矩阵的列数
            height = self.size[k + 1]   #下一层神经元数量，权重矩阵的行数
            #self.W{ k } = (np.random.rand(height, width) - 0.5) * 2 * np.sqrt(6 / (height + width + 1)) - np.sqrt(6 / (height + width + 1));
            
            #初始化第k层到第k+1层的权重矩阵，W[k]=(height,width)
            #np.random.rand(height, width)产生[0,1)均匀分布随机数矩阵
            #权重范围[-sqrt(1/width), sqrt(1/width)]，矩阵尺寸(height,width)
            self.W[k] = 2 * np.random.rand(height, width) / np.sqrt(width) - 1 / np.sqrt(width)
            
            #self.W{ k } = 2 * np.random.rand(height, width) - 1;
            #Xavier initialization
            #初始化偏置项
            #当激活函数为ReLU时，偏置初始化范围为[0.01,1.01]，尺寸为(height,1)
            #当激活函数为其他类型时，偏置初始化为[-1/sqrt(width), sqrt(width)]，尺寸为(height,1)
            if self.active_function == 'relu':
                self.b[k] = np.random.rand(height, 1) + 0.01
            else:
                self.b[k] = 2 * np.random.rand(height, 1) / np.sqrt(width) - 1 / np.sqrt(width)

            #parameters for moments
            method = self.optimization_method

            #当使用动量法时，初始化动量项为零矩阵
            if method == 'Momentum':
                self.vW[k] = np.zeros((height, width), dtype=float)
                self.vb[k] = np.zeros((height, 1), dtype=float)

            #当使用 AdaGrad / RMSProp / Adam 时，初始化梯度平方累积为零矩阵
            if method == 'AdaGrad' or method == 'RMSProp' or method == 'Adam':
                self.rW[k] = np.zeros((height, width), dtype=float)
                self.rb[k] = np.zeros((height, 1), dtype=float)

            #当使用 Adam 时，初始化二阶动量项为零矩阵
            if method == 'Adam':
                self.sW[k] = np.zeros((height, width), dtype=float)
                self.sb[k] = np.zeros((height, 1), dtype=float)

            #parameters for batch normalization.
            #初始化批量归一化参数
            if self.batch_normalization:
                #初始化E和S为零矩阵，Gamma初始化为1，Beta初始化为0
                self.E[k] = np.zeros((height, 1), dtype=float)
                self.S[k] = np.zeros((height, 1), dtype=float)
                self.Gamma[k] = 1
                self.Beta[k] = 0

                #当使用动量法时，设置动量项初始值
                if  method == 'Momentum':
                    self.vGamma[k] = 1
                    self.vBeta[k] = 0
                #当使用 AdaGrad / RMSProp / Adam 时，设置梯度平方累积初始值
                if method == 'AdaGrad' or method == 'RMSProp' or method == 'Adam':
                    self.rW[k] = np.zeros((height, width), dtype=float)
                    self.rb[k] = np.zeros((height, 1), dtype=float)
                    self.rGamma[k] = 0
                    self.rBeta[k] = 0
                #当使用 Adam 时，设置二阶动量项初始值
                if  method == 'Adam':
                    self.sGamma[k] = 1
                    self.sBeta[k] = 0
                # 初始化梯度矩阵
                self.vecNum = 0

            # 初始化权重梯度矩阵
            self.W_grad[k] = np.zeros((height, width), dtype=float)

    def neuralNetworksTrain(self, option : Option, train_x : np.ndarray, train_y : np.ndarray):
        """
        使用指定的训练选项训练神经网络。

        参数说明
        ----------
        option : Option
            神经网络训练选项对象，包含批量大小和迭代次数等参数。

        train_x : numpy.ndarray
            训练数据特征矩阵，形状为 (样本数量, 特征数量)。
            例:train_x.shape = (1000, 20) 表示有 1000 个样本，每个样本有 20 个特征。
            
        train_y : numpy.ndarray
            训练数据标签矩阵，形状为 (样本数量, 标签数量)。
            例:train_y.shape = (1000, 2) 表示有 1000 个样本，每个样本有 2 个标签（独热编码）。

        返回值
        -------
        NeuralNetworksClass
            经过训练后的神经网络对象。
        """
        iteration = option.iteration    # 迭代次数
        batch_size = option.batch_size  # 每次迭代的批量大小
        m = train_x.shape[0]            # 训练样本总数
        num_batches = m / batch_size    # 每次迭代的批次数量
        for k in range(iteration):
            kk = np.random.permutation(m)   # 得到一个随机打乱样本顺序的索引数组
            for j in range(int(num_batches)):
                #(j+1)*batch_size也可以改成max((j+1)*batch_size, len(kk))
                batch_x = train_x[kk[j * batch_size : (j + 1) * batch_size], :] # 取出当前批次的特征数据
                batch_y = train_y[kk[j * batch_size : (j + 1) * batch_size], :] # 取出当前批次的标签数据
                self = nn_forward(self,batch_x,batch_y) # 前向传播计算
                self = nn_backpropagation(self,batch_y) # 反向传播计算梯度
                self = nn_applygradient(self) # 应用梯度更新参数
                
        return self