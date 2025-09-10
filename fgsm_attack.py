# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 禁用CUDA，强制使用CPU
use_cuda = False

# LeNet 模型定义
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # 第1层卷积：1通道→10通道，5x5卷积核
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # 第2层卷积：10通道→20通道
        self.conv2_drop = nn.Dropout2d()  # 随机失活层，防止过拟合
        self.fc1 = nn.Linear(320, 50)     # 全连接层1：320维→50维
        self.fc2 = nn.Linear(50, 10)       # 全连接层2：50维→10维（MNIST类别数）
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))    # 卷积→ReLU→最大池化（降维）
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))  # 带Dropout的卷积操作
        x = x.view(-1, 320)  # 展平特征图为320维向量
        x = F.relu(self.fc1(x))  # 全连接层+ReLU激活
        x = F.dropout(x, training=self.training)  # 训练时应用Dropout
        x = self.fc2(x)          # 输出层
        return F.log_softmax(x, dim=1)  # 对数Softmax，配合NLLLoss使用


# 数据加载配置（需加载训练集）
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)  # 新增训练集
test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # 训练数据加载器
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# 设备配置
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
print(f"Using device: {device}")

# 模型初始化（直接训练，不加载预训练参数）
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ----------------- 新增训练函数 -----------------
def train_model(epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

# FGSM攻击函数
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()                 # 取梯度符号（±1）
    perturbed_image = image + epsilon * sign_data_grad # 沿梯度方向添加扰动
    perturbed_image = torch.clamp(perturbed_image, 0, 1)  # 限制像素值范围
    return perturbed_image

# 攻击测试函数
def test_attack(model, device, test_loader, epsilon):
    correct = 0
    adv_examples = []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        data.requires_grad = True #启用输入数据的梯度计算
        output = model(data) #前向传播，将原始数据输入模型，得到输出概率分布 
        init_pred = output.max(1, keepdim=True)[1] #获取模型对原始数据的预测结果，取概率最大的类别索引
        loss = F.nll_loss(output, target) #计算损失，使用负对数似然损失

        
        model.zero_grad() #清空模型参数的梯度缓存
        loss.backward() #反向传播，计算损失关于输入数据的梯度
        data_grad = data.grad.data #提取输入数据的梯度张量
        
        perturbed_data = fgsm_attack(data, epsilon, data_grad) #生成对抗样本
        output = model(perturbed_data) #前向传播，将对抗样本输入模型
        final_pred = output.max(1, keepdim=True)[1] #获取模型对对抗样本的预测结果
        
        if final_pred.item() == target.item():
            correct += 1
        if len(adv_examples) < 5:
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
    
    final_acc = correct / len(test_loader)
    print(f"Epsilon: {epsilon:.2f}\tAccuracy: {correct}/{len(test_loader)} ({final_acc*100:.1f}%)")
    return final_acc, adv_examples

# 主执行流程
if __name__ == "__main__":
    # 步骤1：训练模型
    train_model(epochs=5)  # 训练3个epoch（约3分钟）
    model.eval()  # 切换到评估模式
    
    # 步骤2：执行对抗攻击
    epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    accuracies = []
    examples = []

    for eps in epsilons:
        acc, ex = test_attack(model, device, test_loader, eps)
        accuracies.append(acc)
        examples.append(ex)

    # 可视化
    plt.figure(figsize=(8,4))
    plt.plot(epsilons, accuracies, "*-")
    plt.xticks(epsilons)
    plt.yticks(torch.arange(0, 1.1, step=0.1))
    plt.title("Model Accuracy vs Attack Strength")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12,8))
    for i, eps in enumerate(epsilons):
        for j in range(len(examples[i])):
            orig, adv, img = examples[i][j]
            plt_idx = i * len(examples[0]) + j + 1
            plt.subplot(len(epsilons), len(examples[0]), plt_idx)
            plt.axis('off')
            color = "green" if orig == adv else "red"
            plt.title(f"ε={eps}\n{orig}→{adv}", color=color)
            plt.imshow(img, cmap="gray")
    plt.tight_layout()
    plt.show() 
