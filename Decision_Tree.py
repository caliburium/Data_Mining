import wandb
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Decision Tree 모델 정의
class DecisionTree:
    def __init__(self):
        self.model = DecisionTreeClassifier()

    def train(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)


def train_decision_tree():
    # W&B 초기화
    wandb.init(project="Datamining_Midterm",
               name='DecisionTree')

    # 데이터 전처리 및 로딩
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 32
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 모델 초기화
    model = DecisionTree(device, 3 * 32 * 32, 10)  # CIFAR-10 이미지 사이즈, Class 갯수

    # Loss 및 Optimizer 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 정확도 저장을 위한 리스트
    accs = []
    loss_list = []

    for epoch in range(30):
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.view(inputs.size(0), -1)
            optimizer.zero_grad()
            outputs = model.predict(inputs.numpy())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

            # 정확도 계산
            predicted = model.predict(inputs.numpy())
            total += labels.size(0)
            correct += np.sum(predicted == labels.numpy())

        accuracy = 100 * correct / total
        accs.append(accuracy)
        # W&B에 Train 정확도 기록
        wandb.log({'train/epoch/loss': np.mean(loss_list), 'train/epoch/acc': np.mean(accs)}, step=epoch)

        correct = 0
        total = 0
        predicted_labels = []
        true_labels = []
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.view(images.size(0), -1)
                predicted = model.predict(images.numpy())
                total += labels.size(0)
                correct += np.sum(predicted == labels.numpy())
                predicted_labels.extend(predicted)
                true_labels.extend(labels.numpy())

        acc = 100 * correct / total
        precision = precision_score(true_labels, predicted_labels, average='macro')
        recall = recall_score(true_labels, predicted_labels, average='macro')
        f1 = f1_score(true_labels, predicted_labels, average='macro')

        print('epoch: %d, loss: %.7f, test acc: %.3f, test precision: %.3f, test recall: %.3f, test f1: %.3f' %
              (epoch + 1, np.mean(loss_list), acc, precision, recall, f1))

        # W&B에 Test 메트릭 기록
        wandb.log({'test/epoch/loss': loss.item(), 'test/epoch/acc': acc, 'test/epoch/precision': precision,
                   'test/epoch/recall': recall, 'test/epoch/f1': f1}, step=epoch)

    wandb.finish()
    print('Finished')


if __name__ == '__main__':
    train_decision_tree()
