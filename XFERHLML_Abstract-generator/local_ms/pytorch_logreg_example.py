import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from settings import DIR_MODELS

"""
High-level huggingface.co documentation:
- https://huggingface.co/transformers/quicktour.html
- https://huggingface.co/transformers/philosophy.html

Raw PyTorch approach
1) Define feedforward input/output
2) Prepare "Forward" method
3) Define loss function
4) Code training loop
5) Define testing pipeline (e.g. generation)
"""


class LogRegNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        output = self.linear(x)
        probs = self.softmax(output)
        return probs


def build_gaussian_data(class_0_amount, class_1_amount):
    # makes gaussian blob with mean [1, 1, 1]
    mean_type_0 = 1
    std_type_0 = 1.5
    blob_type_0 = torch.normal(mean=mean_type_0,
                               std=std_type_0,
                               size=(class_0_amount, input_dim))
    labels_type_0 = torch.zeros(class_0_amount, dtype=torch.int64)

    # makes gaussian blob with mean [-2,-2,-2]
    mean_type_1 = -2
    std_type_1 = 2
    blob_type_1 = torch.normal(mean=mean_type_1,
                               std=std_type_1,
                               size=(class_1_amount, input_dim))
    labels_type_1 = torch.ones(class_1_amount, dtype=torch.int64)

    # store in big dataset array, X, and label array, Y
    data_X = torch.cat((blob_type_0, blob_type_1))
    data_Y = torch.cat((labels_type_0, labels_type_1))

    return data_X, data_Y


class DataLogReg(Dataset):

    def __init__(self, X, Y):
        self.x = X
        self.y = Y
        self.input_dim = self.x.size()[1]

    # Mandatory: Get input pair for training
    def __getitem__(self, idx):
        return self.x[idx, :], self.y[idx]

    # Mandatory: Number of elements in dataset
    def __len__(self):
        X_len = self.x.size()[0]
        return X_len

    # This function is not needed
    def plot(self):
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xs = self.x.numpy()[:, 0]
        ys = self.x.numpy()[:, 1]
        zs = self.x.numpy()[:, 2]
        ax.scatter(xs, ys, zs, c=self.y)
        ax.set_xlabel('x');
        ax.set_ylabel('y');
        ax.set_zlabel('z')
        plt.show()


def report_test_error():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            samples, labels = data
            outputs = net(samples)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


if __name__ == '__main__':

    ################################################################################
    # Build or load data
    ################################################################################
    input_dim = 3
    output_dim = 2

    X_train, Y_train = build_gaussian_data(1000, 200)
    X_test, Y_test = build_gaussian_data(2000, 2000)

    # specify training and testing datasets
    train_dataset = DataLogReg(X_train, Y_train)
    test_dataset = DataLogReg(X_test, Y_test)
    train_dataset.plot()

    ################################################################################
    # Initialize network class
    ################################################################################
    net = LogRegNet(input_dim, output_dim)
    print(net)

    # inspect network class
    params = list(net.parameters())
    print('Num of params matrices to flag_train:', len(params))
    print(params[0].size())

    sample_input = torch.randn(1, input_dim)     # first dimension is the batch dimension when feeding to nn layers
    features_out = net(sample_input)             # expect size batch_size x 2
    print(features_out.size())
    print(features_out)

    ################################################################################
    # Choose loss
    ################################################################################

    criterion = nn.NLLLoss()

    output = net(sample_input)
    target = torch.tensor([1])      # a dummy target class in [0,1], for example
    loss = criterion(output, target)
    print(loss)

    print(loss.grad_fn)  # MSELoss
    print(loss.grad_fn.next_functions[0][0])  # Linear
    print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU


    ################################################################################
    # Optimization
    ################################################################################

    # Training loop hyperparameters
    batch_size = 20
    epochs = 10
    learning_rate = 0.001

    # Choose an optimizer or create one
    import torch.optim as optim
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    # Setup data batching
    nwork = 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nwork)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=nwork)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    ########inputs, labels = data[0].to(device), data[1].to(device)  # what to send to deive? the Dataset object?
    print(device)

    for epoch in range(epochs):       # loop over the dataset multiple times
        print(epoch)
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data  # get the inputs; data is a list of [inputs, labels]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 9
                print('Epoch: %d, batch: %5d, loss: %.3f' %
                      (epoch, i + 1, running_loss))
                report_test_error()
                running_loss = 0.0

    print('Finished Training')

    ################################################################################
    # Save model
    ################################################################################
    import os
    model_path = DIR_MODELS + os.sep + 'net_logreg.pth'
    torch.save(net.state_dict(), model_path)

    ################################################################################
    # Load model
    ################################################################################
    dataiter = iter(test_loader)
    samples, labels = dataiter.next()

    net = LogRegNet(input_dim, output_dim)
    net.load_state_dict(torch.load(model_path))

    outputs = net(samples)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % predicted[j] for j in range(4)))
    print('True: ', ' '.join('%5s' % labels[j] for j in range(4)))

    report_test_error()
