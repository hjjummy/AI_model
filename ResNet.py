# ResNet 모델 코드
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

# ResNet 모델 정의
class BottleNeck(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, activation, down=False):
        super(BottleNeck, self).__init__()
        self.down = down

        # 특성지도의 크기가 감소하는 경우
        if self.down:
            self.layer = nn.Sequential(
                conv_block_1(in_dim, mid_dim, activation, stride=2),
                conv_block_3(mid_dim, mid_dim, activation, stride=1),
                conv_block_1(mid_dim, out_dim, activation, stride=1),
            )

            # 특성지도 크기 + 채널을 맞춰주는 부분
            self.downsample = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=2)

        # 특성지도의 크기가 그대로인 경우
        else:
            self.layer = nn.Sequential(
                conv_block_1(in_dim, mid_dim, activation, stride=1),
                conv_block_3(mid_dim, mid_dim, activation, stride=1),
                conv_block_1(mid_dim, out_dim, activation, stride=1),
            )

        # 채널을 맞춰주는 부분
        self.dim_equalizer = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, x):
        if self.down:
            downsample = self.downsample(x)
            out = self.layer(x)
            out = out + downsample
        else:
            out = self.layer(x)
            if x.size() != out.size():
                x = self.dim_equalizer(x)
            out = out + x
        return out


def conv_block_1(in_dim, out_dim, activation, stride=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride),
        nn.BatchNorm2d(out_dim),
        activation,
    )
    return model


def conv_block_3(in_dim, out_dim, activation, stride=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(out_dim),
        activation,
    )
    return model


class ResNet(nn.Module):
    def __init__(self, base_dim, num_classes=10):
        super(ResNet, self).__init__()
        self.activation = nn.ReLU()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(3, base_dim, 7, 2, 3),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
        )
        self.layer_2 = nn.Sequential(
            BottleNeck(base_dim, base_dim, base_dim * 4, self.activation),
            BottleNeck(base_dim * 4, base_dim, base_dim * 4, self.activation),
            BottleNeck(base_dim * 4, base_dim, base_dim * 4, self.activation, down=True),
        )
        self.layer_3 = nn.Sequential(
            BottleNeck(base_dim * 4, base_dim * 2, base_dim * 8, self.activation),
            BottleNeck(base_dim * 8, base_dim * 2, base_dim * 8, self.activation),
            BottleNeck(base_dim * 8, base_dim * 2, base_dim * 8, self.activation),
            BottleNeck(base_dim * 8, base_dim * 2, base_dim * 8, self.activation, down=True),
        )
        self.layer_4 = nn.Sequential(
            BottleNeck(base_dim * 8, base_dim * 4, base_dim * 16, self.activation),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.activation),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.activation),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.activation),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.activation),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.activation, down=True),
        )
        self.layer_5 = nn.Sequential(
            BottleNeck(base_dim * 16, base_dim * 8, base_dim * 32, self.activation),
            BottleNeck(base_dim * 32, base_dim * 8, base_dim * 32, self.activation),
            BottleNeck(base_dim * 32, base_dim * 8, base_dim * 32, self.activation),
        )
        self.avgpool = nn.AvgPool2d(1, 1)
        self.fc_layer = nn.Linear(base_dim * 32, num_classes)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc_layer(out)
        return out

# ResNet 모델 학습 및 평가 함수 정의
def train_and_evaluate_resnet():
    model = ResNet(base_dim=64).to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 모델 학습
    loss_arr = []
    for i in trange(num_epoch, desc="Training ResNet"):
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
    plt.plot(loss_arr, label='ResNet Loss')
    plt.legend()
    plt.xlabel('Epoch (x10)')
    plt.ylabel('Loss')
    plt.title('ResNet Training Loss')
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
    print(f"Accuracy of ResNet on Test Data: {accuracy}%")
    return accuracy

# ResNet 모델 학습 및 평가 호출
train_and_evaluate_resnet()
