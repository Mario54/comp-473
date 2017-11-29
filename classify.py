import json
from optparse import OptionParser

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from feature_extraction import extract_features_from_file


def load_data(file):
    all_data = []
    classes = []

    with open(file) as data:
        data = json.load(data)

    for font_name, data_points in data.items():
        for point in data_points:
            classes.append(font_name)
            all_data.append(point)

    return all_data, classes


def print_probabilities(class_labels, predict_proba):
    probabilities = zip(sorted(list(set(class_labels))), predict_proba)
    for (c, p) in sorted(probabilities, key=lambda x: x[1], reverse=True):
        if p > 0.01:
            print("{}: {}%".format(c, round(p * 100, 2)))


def print_confusion_matrix(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


def normalized_confusion_matrix(y_test, labels):
    cm = confusion_matrix(y_test, labels)
    np.set_printoptions(precision=2)
    return cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


class KNNClassifier():
    def __init__(self):
        self.knn = KNeighborsClassifier(n_neighbors=3, weights='distance')

    def predict_one(self, training_data, classes, feature_vector):
        self.knn.fit(training_data, classes)

        return self.knn.predict_proba(feature_vector)[0]

    def predict_all(self, x_train, x_test, y_train, y_test):
        self.knn.fit(x_train, y_train)

        return self.knn.predict(x_test)


class GaussianClassifier():
    def __init__(self):
        self.gnb = GaussianNB()

    def predict_one(self, training_data, classes, feature_vector):
        self.gnb.fit(training_data, classes)

        return self.gnb.predict_proba(feature_vector)[0]

    def predict_all(self, x_train, x_test, y_train, y_test):
        self.gnb.fit(x_train, y_train)

        return self.gnb.predict(x_test)


class NeuralNetClassifier():
    def __init__(self):
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        self.clf = MLPClassifier(solver='lbfgs', alpha=1e-5, )

    def predict_one(self, training_data, labels, feature_vector):
        self.scaler.fit(training_data)
        x_train = self.scaler.transform(training_data)
        x = self.scaler.transform(feature_vector)

        self.clf.fit(x_train, labels)
        return self.clf.predict_proba(x)[0]

    def predict_all(self, x_train, x_test, y_train, y_test):
        self.scaler.fit(x_train)
        x_train = self.scaler.transform(x_train)
        x_test = self.scaler.transform(x_test)

        self.clf.fit(x_train, y_train)
        return self.clf.predict(x_test)


if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("--file", dest="file",
                      help="classify this file's font")
    parser.add_option("--data", dest="data_file",
                      help="classify data file")
    parser.add_option("--features-algo", dest="algo", default='text',
                      help="algorithm used to generate features")
    parser.add_option("--classifier", dest="classifier", default="knn")

    (options, args) = parser.parse_args()

    data_file = "data.json"

    if options.data_file is not None:
        data_file = options.data_file

    all_data, classes = load_data(data_file)

    classifiers = {
        "knn": KNNClassifier(),
        "gaussian": GaussianClassifier(),
        "neural-net": NeuralNetClassifier(),
    }

    if options.classifier in classifiers:

        classifier = classifiers[options.classifier]

        if options.file is not None:
            feature_vector = np.asarray(extract_features_from_file(options.file, algo=options.algo)).reshape(1, -1)
            prediction = classifier.predict_one(all_data, classes, feature_vector)
            print_probabilities(classes, prediction)
        else:
            print("classifier: {}".format(options.classifier))
            x_train, x_test, y_train, y_test = train_test_split(all_data, classes, test_size=0.5)
            predictions = classifier.predict_all(x_train, x_test, y_train, y_test)

            print()
            print(classification_report(y_test, predictions))
            print(confusion_matrix(y_test, predictions))
            print(normalized_confusion_matrix(y_test, predictions))


    else:
        print("'{}' algorithm is not supported".format(options.classifier))
        exit()

        # if options.classifier == "neural-net":
        #     from sklearn.preprocessing import StandardScaler
        #
        #     scaler = StandardScaler()
        #     clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
        #                         )
        #
        #     if options.file is not None:
        #         scaler.fit(all_data)
        #         all_data = scaler.transform(all_data)
        #         clf.fit(all_data, classes)
        #         feature_vector = np.asarray(extract_features_from_file(options.file, algo=options.algo)).reshape(1, -1)
        #         print_probabilities(classes, clf.predict_proba(feature_vector)[0])
        #     else:
        #         x_train, x_test, y_train, y_test = train_test_split(all_data, classes, test_size=0.33)
        #
        #         # Don't cheat - fit only on training data
        #         scaler.fit(x_train)
        #         x_train = scaler.transform(x_train)
        #         # apply same transformation to test data
        #         x_test = scaler.transform(x_test)
        #
        #         clf.fit(x_train, y_train)
        #         pred = clf.predict(x_test)
        #
        #         print(classification_report(y_test, pred))
        #         cm = confusion_matrix(y_test, pred)
        #
        #         np.set_printoptions(precision=2)
        #         normalized_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #
        #         print(normalized_cm)
        #         print_cm(cm, sorted(list(set(classes))))
        #
        # elif options.classifier == "svm":
        #     svc = SVC(kernel='rbf', class_weight='balanced')
        #
        #     if options.file is not None:
        #         svc.fit(all_data, classes)
        #         feature_vector = np.asarray(extract_features_from_file(options.file, algo=options.algo)).reshape(1, -1)
        #         print(svc.predict(feature_vector)[0])
        #     else:
        #         x_train, x_test, y_train, y_test = train_test_split(all_data, classes, test_size=0.33)
        #
        #         svc.fit(x_train, y_train)
        #
        #         pred = svc.predict(x_test)
        #
        #         print(classification_report(y_test, pred))
        #         cm = confusion_matrix(y_test, pred)
        #
        #         np.set_printoptions(precision=2)
        #         normalized_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #
        #         print(normalized_cm)
        #         print_cm(cm, sorted(list(set(classes))))
        #
