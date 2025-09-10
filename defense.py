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

# 数据加载配置
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# 设备配置
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
print(f"Using device: {device}")

# 模型初始化
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ----------------- 对抗训练防御函数 -----------------
def adversarial_train(epochs=5, epsilon=0.3):
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

# FGSM攻击函数
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

# 攻击测试函数
def test_attack(model, device, test_loader, epsilon):
    correct = 0
    adv_examples = []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        
        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]
        
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
    # 步骤1：执行对抗训练（防御）
    print("=== 对抗训练（防御）开始 ===")
    adversarial_train(epochs=5, epsilon=0.1)  # 训练5个epoch
    model.eval()
    
    # 步骤2：测试防御效果
    print("\n=== 对抗攻击测试 ===")
    epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    accuracies = []
    examples = []

    for eps in epsilons:
        acc, ex = test_attack(model, device, test_loader, eps)
        accuracies.append(acc)
        examples.append(ex)

    # 可视化对比
    plt.figure(figsize=(8,4))
    plt.plot(epsilons, accuracies, "*-")
    plt.xticks(epsilons)
    plt.yticks(torch.arange(0, 1.1, step=0.1))
    plt.title("Adversarial Training Defense Effect")
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