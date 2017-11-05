from optparse import OptionParser

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import json

from feature_extraction import extract_features_from_file


def load_data():
    all_data = []
    classes = []

    for font_name, data_points in data.items():
        for point in data_points:
            # print(i)
            classes.append(font_name)
            all_data.append(point)

    return all_data, classes


parser = OptionParser()
parser.add_option("--file", dest="file",
                  help="classify this file's font")
parser.add_option("--data", dest="data_file",
                  help="classify data file")

(options, args) = parser.parse_args()

data_file = "data.json"

if options.data_file is not None:
    data_file = options.data_file

with open(data_file) as data_file:
    data = json.load(data_file)

knn = KNeighborsClassifier(n_neighbors=1)

all_data, classes = load_data()

if options.file is not None:
    knn.fit(all_data, classes)
    feature_vector = np.asarray(extract_features_from_file(options.file)).reshape(1, -1)
    print(knn.predict(feature_vector)[0])
else:
    x_train, x_test, y_train, y_test = train_test_split(all_data, classes, test_size=0.33, random_state=42)
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)

    print()
    print(classification_report(y_test, pred))

