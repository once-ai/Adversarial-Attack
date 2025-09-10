# Adversarial-Attack

对抗攻击： 从线性假说到FGSM实践，基于MNIST数据集


***

## 1. 对抗攻击的威胁

+ 对抗攻击：

  2013 年，Szegedy 等人[^1]提出，在图像中添加人类难以察觉的微小扰动后，可以欺骗深度神经网络模型，使其以高置信度输出错误结果。这类攻击被称为对抗攻击。

+ 对抗样本：

  这种添加了扰动并导致网络输出错误分类的图像称为对抗样本。

+ 对抗扰动：

  添加的扰动称为对抗扰动。

在自动驾驶、金融系统和医疗诊断等安全关键领域，对抗攻击使系统产生错误决策，可能对生命、财产以及社会稳定构成严重威胁。比如在自动驾驶场景中，攻击者可以篡改交通标志或道路标记，诱导车辆做出错误反应，进而引发交通事故。人脸识别等身份验证系统也可能因对抗攻击出现误识别，导致身份盗用和非法访问。

模型为什么会对这些微小的扰动如此敏感？这引起了对抗样本存在性的思考。

## 2. 线性假说与FGSM方法

学术界存在多种理论，Goodfellow等人[^2]提出的线性假说为对抗样本的存在提供了关键解释。他们认为**高维空间的线性行为足以产生对抗样本**。

+ **线性模型的高维扰动累积效应**:

  对于线性模型，对抗样本的存在可以通过权重的线性叠加解释：

  <img width="733" height="149" alt="图片" src="https://github.com/user-attachments/assets/3e76d89d-86b2-43c2-9234-0692366941b4" />

  x为输入，x’=x+η为对抗输入，η足够小，那么分类器就应该把x与x’分为同一类。

  <img width="410" height="83" alt="图片" src="https://github.com/user-attachments/assets/8a6daead-32d1-4104-8c19-f7c1da52d4c8" />

  考虑权重向量和对抗样本的点积：

  ω是一个抽象的权重向量，代表了模型对输入空间中不同方向的敏感程度。

​  此时的对抗扰动导致模型的激活值增长ωTη，如果ω权重向量，具有n维，元素平均大小为m，那么激活度就增长到εmn。在高维空间中，扰动η与ω的累积效应会线性放大输出误差，微小扰动也会导致输出剧烈变化。

+ **对神经网络的解释**

  虽然神经网络包含非线性激活函数（如ReLU），但实际训练中，模型参数会被优化为让大部分激活值处于“线性区域”（例如ReLU在正区间是线性的），因为它们更容易优化。这种设计初衷是为了避免梯度消失、加速训练，但也导致模型对线性扰动的敏感性远超预期。

​  以ReLU为例，ReLU的定义为f(x)=max(0,x)

<img width="515" height="276" alt="图片" src="https://github.com/user-attachments/assets/0d087a42-8b59-4e79-b44c-bab8519d0fbf" />

​  当输入x>0时，输出为x，此时ReLU表现为线性函数，当x≤0时，输出恒为0，梯度也为0，神经元处于“关闭”状态。在训练神经网络时，模型通过反向传播调整权重和偏置，以最小化损失函数。ReLU在正区间的梯度为1，能稳定传递梯度，促进参数更新，若神经元长期处于小于等于0的区域，其反向传播梯度为0，导致参数无法更新，所以为了避免梯度消失、加速训练，模型会通过优化参数，使大部分神经元的输入x落入正区间，激活ReLU的线性部分。

​  论文接着提出了基于线性假说的经典对抗攻击方法——FGSM快速梯度符号法。

<img width="620" height="80" alt="图片" src="https://github.com/user-attachments/assets/9bef3465-25a0-4609-9151-a01ac19a9ab7" />

​  x为原始输入，θ为模型参数，y为模型对应的label值，ε是一个微小的扰动值，J(θ,x,y)为模型的损失函数，∇xJ表示损失函数关于输入x的梯度，sign()函数用于取梯度的符号。

​  该方法计算损失函数关于输入图像的梯度，然后沿着梯度的符号方向添加扰动，使模型的损失函数值增大，从而使模型分类错误。

​  实验证明FGSM这种简单的算法确实可以产生误分类的对抗样本，从而证明了Goodfellow等人假设的对抗样本的产生原因是由于模型的线性特性。

下面以MNIST手写数字分类任务为例实践FGSM。

***

## 3. 攻击与防御实现

### 3.1 攻击

#### 3.1.1 训练模型

定义了一个简化的LeNet网络，包含2个卷积层和2个全连接层，激活函数为ReLU，输出层使用log_softmax。

```python
# LeNet 模型定义
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```

经5轮训练后模型分类准确率达到了98.4%。

然后对测试集进行FGSM攻击，观察不同扰动强度下模型的准确率。

#### 3.1.2 FGSM对抗攻击

```python
# FGSM攻击函数
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()                 # 取梯度符号（±1）
    perturbed_image = image + epsilon * sign_data_grad # 沿梯度方向添加扰动
    perturbed_image = torch.clamp(perturbed_image, 0, 1)  # 限制像素值范围
    return perturbed_image
```

这是FGSM生成对抗样本的函数，基于FGSM公式，通过反向传播获取输入梯度，沿梯度方向添加扰动，ε控制扰动强度。

#### 3.1.3 攻击的核心流程

（1）前向传播计算损失

首先前向传播计算损失并获得模型对原始数据的预测结果。

前向传播：将原始数据输入模型，得到输出概率分布，取概率最大的类别索引，获得模型对原始数据的预测结果

```python
data.requires_grad = True #启用输入数据的梯度计算
output = model(data) #前向传播，将原始数据输入模型，得到输出概率分布 
init_pred = output.max(1, keepdim=True)[1] #获取模型对原始数据的预测结果，取概率最大的类别索引
loss = F.nll_loss(output, target) #计算损失，使用负对数似然损失
```

（2）反向传播获取输入梯度

反向传播，计算损失关于输入数据的梯度。

```python
model.zero_grad() #清空模型参数的梯度缓存
loss.backward() #反向传播，计算损失关于输入数据的梯度
data_grad = data.grad.data #提取输入数据的梯度张量
```

（3）生成对抗样本并进行预测

生成对抗样本并进行预测，通过对比原始数据和对抗样本的预测结果，可以评估攻击效果。

```python
perturbed_data = fgsm_attack(data, epsilon, data_grad) #生成对抗样本
output = model(perturbed_data) #前向传播，将对抗样本输入模型
final_pred = output.max(1, keepdim=True)[1] #获取模型对对抗样本的预测结果
```

#### 3.1.4 攻击结果

<img width="797" height="399" alt="图片" src="https://github.com/user-attachments/assets/23f64e01-9821-4bba-97c1-953a17ee61fe" />

<img width="456" height="174" alt="图片" src="https://github.com/user-attachments/assets/261ee454-f939-433b-940b-d08c8d8ed644" />

攻击结果显示，ε越大，攻击越强，模型准确率显著下降，ε=0.15时，准确率已经降到70%以下，ε=0.3时，下降到10%以下

<img width="896" height="527" alt="图片" src="https://github.com/user-attachments/assets/b32bad64-6858-4034-95e7-cb7cd106059e" />

但也可以看出扰动越强，图像被改变越明显，如ε=0.3时，数字轮廓扭曲

### 3.2 对抗训练防御

接着对模型进行对抗训练防御，对抗训练是通过在训练数据中加入对抗样本，使模型在训练时“见过”对抗样本，同时学习正常样本和对抗样本的特征，从而提高鲁棒性。

#### 3.2.1 策略

在每个训练批次中，混合50%原始样本与50%对抗样本（这里设置ε=0.1生成对抗样本）

```python
# ----------------- 对抗训练防御函数 -----------------
def adversarial_train(epochs=5, epsilon=0.1):
    """在训练中混合原始数据与对抗样本，提升模型鲁棒性"""
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            # 生成对抗样本
            data.requires_grad = True
            output = model(data)
            loss = F.nll_loss(output, target)
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            perturbed_data = fgsm_attack(data, epsilon, data_grad)
            
            # 混合原始数据与对抗样本
            combined_data = torch.cat([data, perturbed_data], dim=0)
            combined_target = torch.cat([target, target], dim=0)
            
            # 使用混合数据训练
            optimizer.zero_grad()
            output = model(combined_data)
            loss = F.nll_loss(output, combined_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

```

然后对测试集进行相同的FGSM攻击

<img width="693" height="347" alt="图片" src="https://github.com/user-attachments/assets/8d8a169e-2267-484e-9d8d-6fcf94f5a19e" />

可以看到ε=0.15时，准确率从70%以下恢复到了85%以上，说明对抗训练有效。

***

## 4. 项目总结

### 4.1 对抗攻击方法

+ 基于梯度的攻击方法，利用干净图像的梯度信息生成对抗样本；

+ 基于优化的攻击方法，利用优化目标函数生成对抗样本；

+ 基于迁移的攻击方法，利用对抗攻击之间的迁移性生成对抗样本；

+ 基于GAN的攻击方法，利用GAN网络生成深度神经网络难以区分的对抗样本；

+ 基于决策边界的攻击方法，利用差分进化算法，以迭代的方式生成最佳的对抗样本。

### 4.2 对抗攻击防御方法

+ 数据预处理；

+ 增强神经网络的鲁棒性；

+ 检测对抗样本。

### 4.3 总结

对抗攻击暴露了模型的 “决策脆弱性”—— 即使在干净数据上表现优异，也可能在对抗扰动下产生高置信度错误，这对依赖 AI 的安全关键领域（如自动驾驶、医疗诊断）构成威胁。

自对抗攻击提出以来，就引起了世界广泛的关注，研究人员提出了众多对抗攻击方法和防御方法。

在以后的研究过程中，或许可以将存在的原因作为切入点，解释对抗样本的形成，在深入了解存在的原因之后，研究设计鲁棒性较好的网络模型。


[^1]: SZEGEDY C，ZAREMBA W，SUTSKEVER I，et al.Intriguing properties of neural networks. arXiv preprint arXiv: 1312.6199, 2013.
[^2]: GOODFELLOW I J,SHLENS J,SZEGEDY C.Explaining and harnessing adversarial examples[OL].(2015-02-25).https：//arxiv.org/pdf/1412.6572.pdf. 

