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
                self = self.forward(batch_x, batch_y) # 前向传播计算
                self = nn_backpropagation(batch_y) # 反向传播计算梯度
                self = nn_applygradient() # 应用梯度更新参数
                
        return self
    
    def forward(self, batch_x : np.ndarray, batch_y : np.ndarray):
        """
        神经网络的前向传播计算。

        参数说明
        ----------
        batch_x : numpy.ndarray
            输入数据批次，形状为 (批次大小, 特征数量)。
            例:batch_x.shape = (32, 20) 表示有 32 个样本，每个样本有 20 个特征。
        batch_y : numpy.ndarray
            目标标签批次，形状为 (批次大小, 标签数量)。
            例:batch_y.shape = (32, 2) 表示有 32 个样本，每个样本有 2 个标签（独热编码）。

        返回值
        -------
        NeuralNetworksClass
            包含前向传播结果的神经网络对象。

        """
        s = len(self.cost) + 1 # 当前成本索引
        batch_x = batch_x.T # 转置输入数据，形状变为 (特征数量, 批次大小)
        batch_y = batch_y.T # 转置标签数据，形状变为 (标签数量, 批次大小)
        m = batch_x.shape[1]# 批次大小
        # 输入层激活值:
        #   激活值指的是每一层神经元的输出值
        #   第0层即输入层的激活值就是输入数据本身
        self.a[0] = batch_x

        cost2 = 0 # 正则化成本初始化
        for k in range(1, self.depth):# 遍历每一层（从第一隐藏层到输出层）
            # 计算当前层的线性组合输入
            #np.dot()函数用于矩阵乘法运算,np.tile()函数用于重复数组以匹配指定形状
            #self.W[k-1]：第k-1层到第k层的权重矩阵，形状为 (当前层神经元数量, 上一层神经元数量)
            #self.a[k-1]：上一层的激活值，形状为 (上一层神经元数量, 批次大小)
            #self.b[k-1]：当前层的偏置项，形状为 (当前层神经元数量, 1)
            #np.tile(self.b[k-1], (1, m))：将偏置项沿列方向重复 m 次，形状变为 (当前层神经元数量, 批次大小)
            y = np.dot(self.W[k-1], self.a[k-1]) + np.tile(self.b[k-1], (1, m)) #np.tile就是matlab中的repmat(replicate matrix)
            
            #若启用批量归一化，则对线性组合输入进行归一化处理
            if self.batch_normalization:
                # 累积均值
                # E[k-1]：第k-1层的均值参数，形状为 (当前层神经元数量, 1)
                # np.sum(y, axis=1)：计算当前层线性组合输入 y 在每一行的和，形状为 (当前层神经元数量,)
                # vecNum：已处理样本总数
                # 均值的公式：u = 1/N * Σx_i
                # 累计均值公式：E_new = (E_old * N_old + Σx_i) / (N_old + m)
                self.E[k-1] = self.E[k-1]*self.vecNum + np.array([np.sum(y, axis=1)]).T
                # 累积方差
                # S[k-1]：第k-1层的方差参数，形状为 (当前层神经元数量, 1)
                # np.std(y,ddof=1,axis=1)：计算当前层线性组合输入 y 在每一行的标准差，形状为 (当前层神经元数量,)ddof=1 表示计算无偏估计
                # self.vecNum - 1：已处理样本总数减 1
                # self.S[k-1]**2将标准差转换为方差S_old
                # np.std(y,ddof=1,axis=1)** 2将标准差转换为方差std(y)²
                # 方差的公式：σ² = 1/(N-1) * Σ(x_i - u)²
                # 累计方差公式：S_new = (S_old * (N_old - 1) + (m - 1) * std(y)²) / (N_old + m - 1)
                self.S[k-1] = self.S[k-1]**2 * (self.vecNum - 1) + np.array([(m - 1)*np.std(y,ddof=1,axis=1)** 2]).T #ddof=1计算无偏估计
                self.vecNum = self.vecNum + m # 更新已处理样本总数
                self.E[k-1] = self.E[k-1] / self.vecNum # 计算新的均值，即/ (N_old + m)
                self.S[k-1] = np.sqrt(self.S[k-1] / (self.vecNum - 1)) # 计算新的标准差，即 / (N_old + m - 1)并开方将方差转换为标准差
                # 标准化处理
                # 对当前层的线性组合输入 y 进行标准化处理
                # 标准化公式：y_norm = (y - u) / (σ + ε)
                # 其中 u 是均值，σ 是标准差，ε 是一个小常数防止除零
                #0.0001*np.ones(self.S[k-1].shape)得到一个与self.S[k-1]形状相同的全0.0001矩阵，防止除零
                y = (y - np.tile(self.E[k-1], (1, m))) / np.tile(self.S[k-1]+0.0001*np.ones(self.S[k-1].shape), (1, m)) 
                # 缩放和平移
                # 使用可学习的参数 Gamma 和 Beta 对标准化后的输入进行缩放和平移
                # Gamma：缩放参数，形状为 (当前层神经元数量, 1)
                # Beta：平移参数，形状为 (当前层神经元数量, 1)
                y = self.Gamma[k-1]*y + self.Beta[k-1] 

            # 应用激活函数
            # 隐藏层不使用softmax激活函数，否则会导致梯度消失问题
            if k == self.depth - 1:# 输出层
                f = self.output_function# 输出层激活函数
                if f == 'sigmoid' :
                    self.a[k] = self.sigmoid(y)
                elif f == 'tanh' :
                    self.a[k] = np.tanh(y)# 输出层激活函数为双曲正切函数，公式为 tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
                elif f == 'relu' :
                    self.a[k] = np.maximum(y, 0)# 输出层激活函数为 ReLU 函数，公式为 ReLU(x) = max(0, x)，逐元素取最大值
                elif f == 'softmax' :
                    self.a[k] = self.softmax(y)

            else:
                f = self.active_function
                if f == 'sigmoid' :
                    self.a[k] = self.sigmoid(y)
                elif f == 'tanh' :
                    self.a[k] = np.tanh(y)
                elif f == 'relu' :
                    self.a[k] = np.maximum(y, 0)

            cost2 = cost2 + np.sum(self.W[k-1]**2)# 累积权重的平方和，用于正则化成本计算

        if self.encoder == 1:# 自编码器结构
            # 计算自编码器的成本函数，包括重构误差、权重正则化和稀疏性惩罚
            roj = np.sum(self.a[2], axis=1) / m # 计算隐藏层激活值的平均值，axis=1表示按行求和
            # 成本函数公式：
            # J = 0.5 * ||a[k] - batch_y||² / m + 0.5 * weight_decay * Σ||W||² + β * Σ(ρ log(ρ/roj) + (1-ρ) log((1-ρ)/(1-roj)))
            # 其中 ||·|| 表示 Frobenius 范数，ρ 是目标稀疏度，roj 是实际稀疏度
            # β 是稀疏性惩罚项的权重
            self.cost[s] = 0.5 * np.sum((self.a[k] -batch_y)**2) / m + 0.5 * self.weight_decay * cost2 + 3 * sum(self.sparsity * np.log(self.sparsity / roj) + (1-self.sparsity) * np.log((1-self.sparsity) / (1-roj)))
        else:
            if self.objective_function == 'MSE':# 均方误差损失函数
                # 成本函数公式：
                # J = 0.5 * ||a[k] - batch_y||² / m + 0.5 * weight_decay * Σ||W||²
                # 其中 ||·|| 表示 Frobenius 范数
                self.cost[s] = 0.5 / m * sum(sum((self.a[k] -batch_y)** 2)) + 0.5 * self.weight_decay * cost2 
            elif self.objective_function == 'Cross Entropy':# 交叉熵损失函数
                # 成本函数公式：
                # J = -0.5 * Σ(batch_y * log(a[k])) / m + 0.5 * weight_decay * Σ||W||²
                # 其中 ||·|| 表示 Frobenius 范数
                self.cost[s] = -0.5 * sum(sum(batch_y*np.log(self.a[k]))) / m + 0.5 * self.weight_decay * cost2 
        # self.cost[s]
    
        return self
    
    def sigmoid(x):
        """
        Sigmoid 激活函数。
        参数说明
        ----------
        x : numpy.ndarray
            输入数据，可以是标量、向量或矩阵。

        返回值
        -------
        numpy.ndarray
            经过 Sigmoid 函数变换后的输出，形状与输入相同。
        
        公式
        -------
        Sigmoid 函数定义为：
            S(x) = 1 / (1 + exp(-x))
        """
        np.seterr(divide='ignore', invalid='ignore')# 忽略除零和无效值警告
        return 1 / (1 + np.exp(-x))

    def softmax(x):
        """
        Softmax 激活函数。
        参数说明
        ----------
        x : numpy.ndarray
            输入数据，可以是标量、向量或矩阵。

        返回值
        -------
        numpy.ndarray
            经过 Softmax 函数变换后的输出，形状与输入相同。

        公式
        -------
        Softmax 函数定义为：
            S(x_i) = exp(x_i) / Σ exp(x_j)
        """
        """返回对应的概率值"""
        np.seterr(divide='ignore', invalid='ignore')# 忽略除零和无效值警告
        exp_x = np.exp(x)
        softmax_x = np.zeros(x.shape,dtype=float)
        for i in range(len(x[0])):
            softmax_x[:,i] = exp_x[:,i] / (exp_x[0,i] + exp_x[1,i])
            
        return softmax_x 