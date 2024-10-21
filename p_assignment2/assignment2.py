import numpy as np
from torchvision.datasets import MNIST
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
# import matplotlib.pyplot as plt


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


def softmax(x, w, b):
    # initial score
    z = np.dot(x, w) + b
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))

    # div by sum to get sum of prob 1
    y_hat = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    return y_hat


def compute_loss(y_hat, y_true):
    m = y_true.shape[0]  # Number of examples
    loss = -np.sum(y_true * np.log(y_hat + 1e-8)) / m  # Cross-entropy loss
    return loss


def back_prop(x, y_true, y_hat, w, b, learning_rate):
    m = x.shape[0]  # Number of examples
    error = y_true - y_hat

    # correct weights + bias
    w += learning_rate * np.dot(x.T, error) / m
    b += learning_rate * np.sum(error, axis=0) / m
    return w, b


def train(x, y_true, w, b, learning_rate, epochs, batch_size):
    m = x.shape[0]  # number of training examples
    last_loss = 10000
    for epoch in range(epochs):
        # Shuffle data at the beginning of each epoch
        perm = np.random.permutation(m)
        x_shuffled = x[perm]
        y_shuffled = y_true[perm]
        loss = None

        # split data into batches
        for i in range(0, m, batch_size):
            x_batch = x_shuffled[i : i+batch_size]
            y_batch = y_shuffled[i : i+batch_size]

            # forward propagation: scores -> probabilities
            y_hat = softmax(x_batch, w, b)

            # compute the loss
            loss = compute_loss(y_hat, y_batch)

            # backward propagation for weight and bias recalc
            w, b = back_prop(x_batch, y_batch, y_hat, w, b, learning_rate)

        if epoch % 5 == 0:
            y_train_pred = predict(x, w, b)
            acc = accuracy_score(np.argmax(y_true, axis=1), y_train_pred)
            if loss < last_loss:
                print(f'\033[0mEpoch {epoch:<2}: Train Accuracy: {100*acc:.5f}%, Loss: \033[32m{loss:.5f}')
            else:
                print(f'\033[0mEpoch {epoch:<2}: Train Accuracy: {100 * acc:.5f}%, Loss: \033[31m{loss:.5f}')
            last_loss = loss

    return w, b


def predict(x, w, b):
    y_hat = softmax(x, w, b)
    return np.argmax(y_hat, axis=1)


def main():
    # Load dataset into numpy arrays
    train_x, train_y = download_mnist(True, max_samples=60000)  # Load only 10,000 samples
    test_x, test_y = download_mnist(False, max_samples=10000)    # Load only 2,000 samples
    train_x = np.array(train_x, dtype=float)
    test_x = np.array(test_x, dtype=float)
    train_y = np.array(train_y, dtype=float)
    test_y = np.array(test_y, dtype=float)

    # Normalize pixels to [0, 1]
    train_x /= 255.0
    test_x /= 255.0

    # encode labels with OneHot
    encoder = OneHotEncoder(sparse_output=False)  # Updated parameter
    train_y_encoded = encoder.fit_transform(train_y.reshape(-1, 1))
    test_y_encoded = encoder.transform(test_y.reshape(-1, 1))

    # initial weights and bias
    np.random.seed(42069)  # reproducible output with seed
    w = np.random.randn(784, 10).astype(np.float32) * 0.1  # not as important as learning rate!
    b = np.zeros((1, 10), dtype=np.float32)

    # initial accuracy before training
    y_test_pred_initial = predict(test_x, w, b)
    initial_accuracy = accuracy_score(test_y, y_test_pred_initial)
    print(f'Initial Accuracy: {100*initial_accuracy:.2f}%')

    # training
    learning_rate = 0.04  # more means, more radical correcting -> will fluctuate more!
    epochs = 100
    batch_size = 100
    w, b = train(train_x, train_y_encoded, w, b, learning_rate, epochs, batch_size)

    # final accuracy
    y_test_pred_final = predict(test_x, w, b)
    final_accuracy = accuracy_score(test_y, y_test_pred_final)
    print(f'\n\033[0mFinal accuracy after {epochs} epochs on training dataset: {100*final_accuracy:.2f}%')


if __name__ == "__main__":
    main()
