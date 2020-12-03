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
            """
            # append only the k smallest
            if len(dist_list) < k:
                dist_list.append(dist)
            else:
                dist_list.sort()
                len_ = len(dist_list) - 1
                remove_dist = dist_list[len_]
                if remove_dist > dist:
                    dist_list.remove(remove_dist)
                    dist_list.append(dist)
            """
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

        """
        # sort and get the k smallest
        k_smallest = np.argpartition(dist_list, k)
        # get the k nearest labels
        closes_labels = []
        for label in range(k):
            closes_labels.append(labels_arr[k_smallest[label]])
        """
        # assign a class to the test point according to the majority of the classes
        count = np.bincount(closes_labels)
        class_assign = np.argmax(count)
        new_knn_labels.append(class_assign)
    return new_knn_labels


def mc_perceptron(train_x, train_y):
    train_x = np.array(train_x)
    train_y = np.array(train_y)

    # initialize array of w
    w = np.zeros((3, 12))

    # initialize eta and epochs
    eta = 0.1
    epochs = 100

    # train
    for epoch in range(epochs):
        count_mis = 0
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
            else:
                count_mis += 1
        print("epoch: {}, success rate: {}".format(epoch, (count_mis / 355) * 100))
    return w


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

    # k=3 and k=5 returns me 77.14%
    k = 3
    # run the knn algorithm
    knn_alg = knn(train_set, labels_arr, test_set, k)
    #print(knn_alg)

    perceptron_alg = mc_perceptron(train_set, labels_arr)
    #print(perceptron_alg)

    # *********** todo print int the end:
    ## print(f"knn: {knn_yhat}, perceptron: {perceptron_yhat}, pa: {pa_yhat}")


if __name__ == "__main__":
    main()