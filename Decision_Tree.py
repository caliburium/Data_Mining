import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():
    # 데이터 전처리 및 로딩
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 32
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Decision Tree 모델 초기화
    model = DecisionTreeClassifier()

    # 데이터 및 레이블 초기화
    train_data = []
    train_labels = []

    # 학습 데이터 로딩
    for data in trainloader:
        inputs, labels = data
        inputs = inputs.view(inputs.size(0), -1).numpy()
        train_data.extend(inputs)
        train_labels.extend(labels.numpy())

    # 모델 학습
    model.fit(train_data, train_labels)

    # 정확도, 정밀도, 재현율, F1 점수 계산
    correct = 0
    total = 0
    predicted_labels = []
    true_labels = []

    for data in testloader:
        images, labels = data
        images = images.view(images.size(0), -1).numpy()
        predicted = model.predict(images)
        total += labels.size(0)
        correct += np.sum(predicted == labels.numpy())
        predicted_labels.extend(predicted)
        true_labels.extend(labels.numpy())

    accuracy = 100 * correct / total
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')

    # 결과 기록
    print('Test Accuracy: %.3f' % accuracy)
    print('Test Precision: %.3f' % precision)
    print('Test Recall: %.3f' % recall)
    print('Test F1-Score: %.3f' % f1)


if __name__ == '__main__':
    main()
