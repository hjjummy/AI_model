# GoogLeNet 모델 정의 및 CIFAR-10 성능 비교
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import trange

# 하이퍼파라미터 설정
batch_size = 100
learning_rate = 0.0002
num_epoch = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 데이터 전처리 및 로드
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cifar10_train = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
cifar10_test = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=cifar10_train, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
test_loader = DataLoader(dataset=cifar10_test, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

# 인셉션 모듈 구성 요소 정의
def conv_1(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 1, 1),  # 1x1 컨볼루션
        nn.ReLU(),
    )
    return model

def conv_1_3(in_dim, mid_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, mid_dim, 1, 1),  # 1x1 컨볼루션
        nn.ReLU(),
        nn.Conv2d(mid_dim, out_dim, 3, 1, 1),  # 3x3 컨볼루션
        nn.ReLU()
    )
    return model

def conv_1_5(in_dim, mid_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, mid_dim, 1, 1),  # 1x1 컨볼루션
        nn.ReLU(),
        nn.Conv2d(mid_dim, out_dim, 5, 1, 2),  # 5x5 컨볼루션
        nn.ReLU()
    )
    return model

def max_3_1(in_dim, out_dim):
    model = nn.Sequential(
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),  # 3x3 맥스풀링
        nn.Conv2d(in_dim, out_dim, 1, 1),  # 1x1 컨볼루션
        nn.ReLU(),
    )
    return model

# 인셉션 모듈 정의
class inception_module(nn.Module):
    def __init__(self, in_dim, out_dim_1, mid_dim_3, out_dim_3, mid_dim_5, out_dim_5, pool_dim):
        super(inception_module, self).__init__()
        self.conv_1 = conv_1(in_dim, out_dim_1)
        self.conv_1_3 = conv_1_3(in_dim, mid_dim_3, out_dim_3)
        self.conv_1_5 = conv_1_5(in_dim, mid_dim_5, out_dim_5)
        self.max_3_1 = max_3_1(in_dim, pool_dim)

    def forward(self, x):
        # 인셉션 모듈 내 각 구성 요소로부터 출력 받기
        out_1 = self.conv_1(x)
        out_2 = self.conv_1_3(x)
        out_3 = self.conv_1_5(x)
        out_4 = self.max_3_1(x)
        # 모든 출력을 연결
        output = torch.cat([out_1, out_2, out_3, out_4], 1)
        return output

# GoogLeNet 모델 정의
class GoogLeNet(nn.Module):
    def __init__(self, base_dim, num_classes=10):
        super(GoogLeNet, self).__init__()
        # 모델의 초기 층들
        self.layer_1 = nn.Sequential(
            nn.Conv2d(3, base_dim, 7, 2, 3),  # 첫 번째 컨볼루션 층
            nn.MaxPool2d(3, 2, 1),  # 첫 번째 맥스풀링
            nn.Conv2d(base_dim, base_dim * 3, 3, 1, 1),  # 두 번째 컨볼루션 층
            nn.MaxPool2d(3, 2, 1),  # 두 번째 맥스풀링
        )
        # 인셉션 모듈 적용
        self.layer_2 = nn.Sequential(
            inception_module(base_dim * 3, 64, 96, 128, 16, 32, 32),
            inception_module(base_dim * 4, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3, 2, 1),
        )
        self.layer_3 = nn.Sequential(
            inception_module(480, 192, 96, 208, 16, 48, 64),
            inception_module(512, 160, 112, 224, 24, 64, 64),
            inception_module(512, 128, 128, 256, 24, 64, 64),
            inception_module(512, 112, 144, 288, 32, 64, 64),
            inception_module(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3, 2, 1),
        )
        self.layer_4 = nn.Sequential(
            inception_module(832, 256, 160, 320, 32, 128, 128),
            inception_module(832, 384, 192, 384, 48, 128, 128),
            nn.AdaptiveAvgPool2d((1, 1)),  # 수정: AdaptiveAvgPool2d로 변경하여 출력 크기를 1x1로 고정
        )
        self.layer_5 = nn.Dropout2d(0.4)  # 드롭아웃 적용
        self.fc_layer = nn.Linear(1024, num_classes)  # 최종 분류를 위한 선형 계층

    def forward(self, x):
        # 각 층을 순차적으로 통과
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = out.view(out.size(0), -1)  # 플래튼
        out = self.fc_layer(out)  # 최종 출력
        return out

# 학습 및 평가 함수 정의
def train_and_evaluate(model, model_name):
    model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 학습 과정
    loss_arr = []
    for i in trange(num_epoch, desc=f"Training {model_name}"):
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
    plt.plot(loss_arr, label=f'{model_name} Loss')
    plt.legend()
    plt.xlabel('Epoch (x10)')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Training Loss')
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
    print(f"Accuracy of {model_name} on Test Data: {accuracy}%")
    return accuracy

# GoogLeNet 모델 정의 및 학습, 평가 호출
googlenet_model = GoogLeNet(base_dim=64)
googlenet_accuracy = train_and_evaluate(googlenet_model, "GoogLeNet")

# 결과 출력
print(f"GoogLeNet Accuracy: {googlenet_accuracy}%")
