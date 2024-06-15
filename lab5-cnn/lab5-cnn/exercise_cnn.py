import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
from cnn_hard_version import ImageCNN
from cnn_easy_version import ImageCNN as ImageCNN_easy
import pickle
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'

class CIFAR10Dataset():
    def __init__(self, data_path, train=True, transform=None):
        X, y = self.load_data(data_path, train)
        self.X = X
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        img = self.X[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.y is None:
            return img
        else:
            return img, int(self.y[index])

    def __len__(self):
        return len(self.X)

    def load_data(self, data_path, train):
        y_train = None

        if train:
            with open(data_path + '_labels', 'rb') as f:
                y_train = np.asarray(pickle.load(f)).reshape(-1)

        with open(data_path + '_images', 'rb') as f:
            x_train = np.asarray(pickle.load(f)).reshape(-1, 3, 32*32)
        return x_train, y_train

def display_cifar():
    data_train = CIFAR10Dataset('./input/train', train=True)
    fig = plt.figure()
    index = np.arange(len(data_train))
    np.random.shuffle(index)
    index2label = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(np.transpose(data_train[index[i]][0].reshape(3, 32, 32), (1, 2, 0)), interpolation='none')
        plt.title("Ground Truth: {}".format(index2label[data_train[index[i]][1]]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def load_cifar10():
    train_set = CIFAR10Dataset('./input/train', train=True,
                             transform = transforms.Compose([
                                         transforms.ToTensor()
                                         ]))
    test_set = CIFAR10Dataset('./input/test', train=False,
                            transform = transforms.Compose([
                                        transforms.ToTensor()
                                        ]))
    return train_set, test_set


def train_with_CNN_hard(train_set, test_set):

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    model = ImageCNN((32, 32) , 10, 3, 16, (8, 8), (5, 5))

    # gpu环境下
    # model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_func = nn.CrossEntropyLoss()

    num_epochs = 50
    for epoch in range(1, num_epochs + 1):
        train_l_sum, train_acc_sum, n = 0., 0., 0
        for i, Xy in enumerate(train_dataloader):
            #if i> 10:continue
            X, y = Xy

            # gpu环境下
            # X = X.cuda()
            # y = y.cuda()

            X = X.reshape(-1, 3, 32, 32)
            y_hat = model(X).squeeze(1)
            loss = loss_func(y_hat, y.long()).sum()
            # print('epoch:',epoch,'  iteros:',i,'  loss:',loss)

            # 梯度清零
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_l_sum += loss.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]

        print('epoch %d, loss %.4f, train acc %.3f'
              % (epoch, train_l_sum / n, train_acc_sum / n))

    X = torch.stack(list(test_set), 0)
    X = X.reshape(-1, 3, 32, 32)

    # gpu环境下
    # X = X.cuda()

    y_hat = model(X)
    y_hat = y_hat[:, :].argmax(dim=1).numpy()

    # gpu环境下
    # y_hat = y_hat[:, :].argmax(dim=1).cpu().numpy()  # 将结果从 GPU 移动到 CPU，然后转换为 NumPy 数组

    pred_txt = [str(w) for w in y_hat]
    g = open('./output/predict.txt', 'w')
    g.write('\n'.join(pred_txt))
    g.close()
    print('./output/predict.txt successfully saved')

def train_with_CNN_easy(train_set, test_set):

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    model = ImageCNN_easy(10, 3, 16, (5, 5), (3, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_func = nn.CrossEntropyLoss()

    num_epochs = 100
    for epoch in range(1, num_epochs + 1):
        train_l_sum, train_acc_sum, n = 0., 0., 0
        for i, Xy in enumerate(train_dataloader):
            #if i> 10:continue
            X, y = Xy
            X = X.reshape(-1, 3, 32, 32)
            y_hat = model(X).squeeze(1)
            loss = loss_func(y_hat, y.long()).sum()

            # 梯度清零
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_l_sum += loss.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]

        print('epoch %d, loss %.4f, train acc %.3f'
              % (epoch, train_l_sum / n, train_acc_sum / n))

if __name__ == '__main__':
    train_set, test_set = load_cifar10()

    display_cifar()

    train_with_CNN_hard(train_set, test_set)

    train_with_CNN_easy(train_set, test_set)

