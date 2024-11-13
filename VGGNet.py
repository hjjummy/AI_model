# VGG 모델 코드
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import trange

# 하이퍼파라미터 설정
batch_size = 50
learning_rate = 0.0002
num_epoch = 100

# 데이터 전처리 및 로드
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cifar10_train = datasets.CIFAR10(root="../Data/", train=True, transform=transform, download=True)
cifar10_test = datasets.CIFAR10(root="../Data/", train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=cifar10_train, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
test_loader = DataLoader(dataset=cifar10_test, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

# 클래스 목록 정의
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 공통 장치 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# VGG 모델 정의
class VGG(nn.Module):
    def __init__(self, base_dim, num_classes=10):
        super(VGG, self).__init__()
        self.feature = nn.Sequential(
            self.conv_2_block(3, base_dim),
            self.conv_2_block(base_dim, 2 * base_dim),
            self.conv_3_block(2 * base_dim, 4 * base_dim),
            self.conv_3_block(4 * base_dim, 8 * base_dim),
            self.conv_3_block(8 * base_dim, 8 * base_dim),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(8 * base_dim * 1 * 1, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes),
        )

    def conv_2_block(self, in_dim, out_dim):
        model = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        return model

    def conv_3_block(self, in_dim, out_dim):
        model = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        return model

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

# VGG 모델 학습 및 평가 함수 정의
def train_and_evaluate_vgg():
    model = VGG(base_dim=64).to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 모델 학습
    loss_arr = []
    for i in trange(num_epoch, desc="Training VGG Net"):
        model.train()
        for j, (image, label) in enumerate(train_loader):
            x = image.to(device)
            y_ = label.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = loss_func(output, y_)
            loss.backward()
            optimizer.step()

        if i % 10 == 0:
            print(f"Epoch {i}, Loss: {loss.item()}")
            loss_arr.append(loss.cpu().detach().numpy())

    # 손실 그래프 그리기
    plt.plot(loss_arr, label='VGG Net Loss')
    plt.legend()
    plt.xlabel('Epoch (x10)')
    plt.ylabel('Loss')
    plt.title('VGG Net Training Loss')
    plt.show()

    # 모델 평가
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for image, label in test_loader:
            x = image.to(device)
            y = label.to(device)
            output = model(x)
            _, output_index = torch.max(output, 1)

            total += label.size(0)
            correct += (output_index == y).sum().float()

    accuracy = 100 * correct / total
    print(f"Accuracy of VGG Net on Test Data: {accuracy}%")
    return accuracy

# VGG 모델 학습 및 평가 호출
train_and_evaluate_vgg()