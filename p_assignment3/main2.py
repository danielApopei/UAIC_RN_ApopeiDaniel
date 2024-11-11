import numpy as np
from torchvision.datasets import MNIST
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder


def download_mnist(is_train: bool, max_samples=None):
    dataset = MNIST(root='./data',
                    transform=lambda x: np.array(x).flatten(),
                    download=True,
                    train=is_train)
    mnist_data = []
    mnist_labels = []
    for i, (image, label) in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break
        mnist_data.append(image)
        mnist_labels.append(label)
    return mnist_data, mnist_labels


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def forward_propagation(x, w1, b1, w2, b2, dropout_rate=0.1, training=True):
    z1 = np.dot(x, w1) + b1
    a1 = relu(z1)
    if training:
        dropout_mask = (np.random.rand(*a1.shape) > dropout_rate).astype(float)
        a1 *= dropout_mask
        a1 /= (1 - dropout_rate)
    z2 = np.dot(a1, w2) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2


def compute_loss(y_hat, y_true):
    m = y_true.shape[0]
    cross_entropy_loss = -np.sum(y_true * np.log(y_hat + 1e-8)) / m
    return cross_entropy_loss


def back_propagation(x, y_true, z1, a1, z2, a2, w1, w2, b1, b2, learning_rate):
    m = x.shape[0]

    dz2 = a2 - y_true
    dw2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m
    dz1 = np.dot(dz2, w2.T) * relu_derivative(z1)
    dw1 = np.dot(x.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2
    return w1, b1, w2, b2


def train(x, y_true, w1, b1, w2, b2, learning_rate, epochs, batch_size, dropout_rate=0.1):
    m = x.shape[0]
    for epoch in range(epochs):
        perm = np.random.permutation(m)
        x_shuffled = x[perm]
        y_shuffled = y_true[perm]
        for i in range(0, m, batch_size):
            x_batch = x_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            z1, a1, z2, a2 = forward_propagation(x_batch, w1, b1, w2, b2, dropout_rate, training=True)
            w1, b1, w2, b2 = back_propagation(x_batch, y_batch, z1, a1, z2, a2, w1, w2, b1, b2, learning_rate)
        if epoch % 5 == 0:
            _, _, _, y_hat = forward_propagation(x, w1, b1, w2, b2, dropout_rate, training=False)
            loss = compute_loss(y_hat, y_true)
            y_train_pred = np.argmax(y_hat, axis=1)
            acc = accuracy_score(np.argmax(y_true, axis=1), y_train_pred)
            print(f'Epoch {epoch:<2} --- Train Accuracy: {100*acc:.5f}%, Loss: {loss:.5f}')
        if epoch >= 25:
            learning_rate /= 1.12
        if epoch >= 30:
            learning_rate /= 1.12
        if epoch >= 40:
            learning_rate /= 1.12
    return w1, b1, w2, b2


def test(x, y_true, w1, b1, w2, b2):
    _, _, _, a2 = forward_propagation(x, w1, b1, w2, b2, dropout_rate=0.0, training=False)
    y_pred = np.argmax(a2, axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


def main():
    x_train, y_train = download_mnist(is_train=True, max_samples=60000)
    x_test, y_test = download_mnist(is_train=False, max_samples=10000)
    enc = OneHotEncoder()
    y_train_encoded = enc.fit_transform(np.array(y_train).reshape(-1, 1)).toarray()
    x_train = np.array(x_train) / 255.0
    x_test = np.array(x_test) / 255.0
    np.random.seed(50)
    w1 = np.random.randn(784, 100).astype(np.float32) * 0.1
    b1 = np.zeros((1, 100), dtype=np.float32)
    w2 = np.random.randn(100, 10).astype(np.float32) * 0.1
    b2 = np.zeros((1, 10), dtype=np.float32)
    learning_rate = 0.27
    epochs = 50
    batch_size = 100
    dropout_rate = 0.1
    w1, b1, w2, b2 = train(x_train, y_train_encoded, w1, b1, w2, b2, learning_rate, epochs, batch_size, dropout_rate)
    accuracy = test(x_test, y_test, w1, b1, w2, b2)
    print(f'Test Accuracy: {accuracy}')


main()