import numpy as np
import sys


def knn(train_set, labels_arr, test_set, k):
    new_knn_labels = []
    # for each sample in test:
    for i in range(len(test_set)):
        line = 0
        dictionary = {}
        dist_list = []
        test = np.array(test_set[i])
        # find the euclidean distance to all training points - store in list
        for j in range(len(train_set)):
            train = np.array(train_set[j])
            dist = np.linalg.norm(test - train)
            dist_list.append(dist)
            # insert the label to the dict
            dictionary[dist] = line
            line += 1

        # sort the list
        sorted_distances = sorted(dist_list)
        # get the k smallest points
        k_smallest = sorted_distances[:k]
        labels = []
        closes_labels = []
        for ind in k_smallest:
            labels.append(dictionary[ind])
        for ind2 in labels:
            ind2 = int(ind2)
            result = int(labels_arr[ind2])
            closes_labels.append(result)

        # assign a class to the test point according to the majority of the classes
        count = np.bincount(closes_labels)
        class_assign = np.argmax(count)
        new_knn_labels.append(class_assign)
    return new_knn_labels


def mc_perceptron(train_x, train_y, test_x):
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    # add bias
    train_bias = [1] * len(train_x)
    train_x = np.concatenate((train_x, np.array(train_bias)[:, None]), axis=1)
    test_bias = [1] * len(test_x)
    test_x = np.concatenate((test_x, np.array(test_bias)[:, None]), axis=1)

    # initialize array of w
    w = np.zeros((3, 13))
    # initialize eta and epochs
    eta = 0.1
    epochs = 100

    # train
    for epoch in range(epochs):
        # shuffle the training seta
        zip_info = list(zip(train_x, train_y))
        np.random.shuffle(zip_info)
        # run on each train sample
        for x, y in zip(train_x, train_y):
            y = int(y)
            y_hat = np.argmax(np.dot(w, x))
            y_hat = int(y_hat)
            # update
            if y != y_hat:
                w[y, :] += eta * x
                w[y_hat, :] -= eta * x

    perc = []
    # get the test classification
    for test in test_x:
        test = np.array(test)
        classify = np.argmax(np.dot(w, test))
        perc.append(classify)
    return perc


def passive_aggressive(train_x, train_y, test_x):
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    # add bias
    train_bias = [1] * len(train_x)
    train_x = np.concatenate((train_x, np.array(train_bias)[:, None]), axis=1)
    test_bias = [1] * len(test_x)
    test_x = np.concatenate((test_x, np.array(test_bias)[:, None]), axis=1)

    # initialize array of w and epochs
    w = np.zeros((3, 13))
    epochs = 100

    # train
    for epoch in range(epochs):
        # shuffle the training seta
        zip_info = list(zip(train_x, train_y))
        np.random.shuffle(zip_info)
        # run on each train sample
        for x, y in zip(train_x, train_y):
            y = int(y)
            y_hat = np.argmax(np.dot(w, x))
            y_hat = int(y_hat)
            # update - if y = y_hat then we don't need to update. can see that here:
            # https://www.geeksforgeeks.org/passive-aggressive-classifiers/
            if y != y_hat:
                loss = max(0, 1 - (np.dot(w[y, :], x)) + (np.dot(w[y_hat, :], x)))
                tau = loss / (2 * (np.power(np.linalg.norm(x), 2)))
                w[y, :] += tau * x
                w[y_hat, :] -= tau * x

    pa = []
    # get the test classification
    for test in test_x:
        test = np.array(test)
        classify = np.argmax(np.dot(w, test))
        pa.append(classify)
    return pa


def main():
    train_x, train_y, test_x = sys.argv[1], sys.argv[2], sys.argv[3]
    # change Red wine to '1', and white wine to '0'
    train_set = np.loadtxt(train_x, delimiter=',', converters={11: lambda d: 1 if d == b'R' else 0})
    test_set = np.loadtxt(test_x, delimiter=',', converters={11: lambda d: 1 if d == b'R' else 0})
    # put all the labels into array
    labels_arr = np.genfromtxt(train_y)

    # normalize train set and test set
    for i in range(12):
        test_set[:, i] = (test_set[:, i] - train_set[:, i].min()) / (train_set[:, i].max() - train_set[:, i].min())

    for i in range(12):
        train_set[:, i] = (train_set[:, i] - train_set[:, i].min()) / (train_set[:, i].max() - train_set[:, i].min())

    # when k=7 returns me 77.14%
    k = 7
    # run the knn algorithm
    knn_alg = knn(train_set, labels_arr, test_set, k)

    # run perceptron with more then 60% success
    perceptron_alg = mc_perceptron(train_set, labels_arr, test_set)

    # run passive-aggressive with more then 60% success
    pa_alg = passive_aggressive(train_set, labels_arr, test_set)

    for i in range(len(knn_alg)):
        print(f"knn: {knn_alg[i]}, perceptron: {perceptron_alg[i]}, pa: {pa_alg[i]}")


if __name__ == "__main__":
    main()