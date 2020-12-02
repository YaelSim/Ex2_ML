import numpy as np
import random
# import scipy
import sys


def main():
    train_x, train_y = sys.argv[1], sys.argv[2]
    train = np.loadtxt(train_x, delimiter=',', converters={11: lambda d: 1 if d == b'R' else 0})
    labels = np.genfromtxt(train_y)

    # take 90% (320) examples to the training set
    # 10% (35) examples to the test set.
    x = train[35:]
    y = labels[35:]
    test = train[:35]
    real_labels = labels[:35]

    # min max normalization between 0-1 to test set.
    for i in range(12):
        test[:, i] = (test[:, i] - test[:, i].min()) / (x[:, i].max() - x[:, i].min())

    # min max normalization between 0-1 to training set.
    for i in range(12):
        x[:, i] = (x[:, i] - x[:, i].min()) / (x[:, i].max() - x[:, i].min())



    success_rate = []
    for j in range(3, 16, 2):
        knn_labels = knn(x, y, test, j)
        #print(knn_labels)
        s = compare_knn_labels_with_real_labels(knn_labels, real_labels)
        success_rate.append(s)
        print("{} : {}".format(j, success_rate[-1]))


def compare_knn_labels_with_real_labels(knn_labels, real_labels):
    count = 0
    size = len(knn_labels)
    for i in range(size):
        if knn_labels[i] == real_labels[i]:
            count += 1
    return (count / size) * 100


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
            result = int(labels_arr[int(ind2)])
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


if __name__ == '__main__':
    main()
