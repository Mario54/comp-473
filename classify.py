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


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
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


if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("--file", dest="file",
                      help="classify this file's font")
    parser.add_option("--data", dest="data_file",
                      help="classify data file")
    parser.add_option("--features-algo", dest="algo", default='gabor',
                      help="algorithm used to generate features")
    parser.add_option("--classifier", dest="classifier", default="knn")

    (options, args) = parser.parse_args()

    data_file = "data.json"

    if options.data_file is not None:
        data_file = options.data_file

    all_data, classes = load_data(data_file)
    if options.classifier == "knn":

        knn = KNeighborsClassifier(n_neighbors=3, weights='distance')

        if options.file is not None:
            knn.fit(all_data, classes)
            feature_vector = np.asarray(extract_features_from_file(options.file, algo=options.algo)).reshape(1, -1)
            print_probabilities(classes, knn.predict_proba(feature_vector)[0])
        else:
            x_train, x_test, y_train, y_test = train_test_split(all_data, classes, test_size=0.5)
            knn.fit(x_train, y_train)
            pred = knn.predict(x_test)

            print()
            print(classification_report(y_test, pred))
            print(confusion_matrix(y_test, pred))

    elif options.classifier == "neural-net":
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                            )

        if options.file is not None:
            scaler.fit(all_data)
            all_data = scaler.transform(all_data)
            clf.fit(all_data, classes)
            feature_vector = np.asarray(extract_features_from_file(options.file, algo=options.algo)).reshape(1, -1)
            print_probabilities(classes, clf.predict_proba(feature_vector)[0])
        else:
            x_train, x_test, y_train, y_test = train_test_split(all_data, classes, test_size=0.33)

            # Don't cheat - fit only on training data
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            # apply same transformation to test data
            x_test = scaler.transform(x_test)

            clf.fit(x_train, y_train)
            pred = clf.predict(x_test)

            print(classification_report(y_test, pred))
            cm = confusion_matrix(y_test, pred)

            np.set_printoptions(precision=2)
            normalized_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            print(normalized_cm)
            print_cm(cm, sorted(list(set(classes))))

    elif options.classifier == "svm":
        svc = SVC(kernel='rbf', class_weight='balanced')

        if options.file is not None:
            svc.fit(all_data, classes)
            feature_vector = np.asarray(extract_features_from_file(options.file, algo=options.algo)).reshape(1, -1)
            print( svc.predict(feature_vector)[0])
        else:
            x_train, x_test, y_train, y_test = train_test_split(all_data, classes, test_size=0.33)

            svc.fit(x_train, y_train)

            pred = svc.predict(x_test)

            print(classification_report(y_test, pred))
            cm = confusion_matrix(y_test, pred)

            np.set_printoptions(precision=2)
            normalized_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            print(normalized_cm)
            print_cm(cm, sorted(list(set(classes))))

    elif options.classifier == "gaussian":
        gnb = GaussianNB()

        if options.file is not None:
            # print(classes)
            gnb.fit(all_data, classes)
            feature_vector = np.asarray(extract_features_from_file(options.file, algo=options.algo)).reshape(1, -1)
            print(feature_vector)
            print_probabilities(classes, gnb.predict_proba(feature_vector)[0])
            print(gnb.theta_)
            # print(gnb.sigma_)
        else:
            x_train, x_test, y_train, y_test = train_test_split(all_data, classes, test_size=0.33)
            y_pred = gnb.fit(x_train, y_train)

            pred = gnb.predict(x_test)

            print(classification_report(y_test, pred))
            cm = confusion_matrix(y_test, pred)

            np.set_printoptions(precision=2)
            normalized_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            print(normalized_cm)
            # print_cm(cm, sorted(list(set(classes))))
