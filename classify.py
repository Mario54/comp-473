from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import json


with open('data.json') as data_file:
    data = json.load(data_file)

all_data = []
classes = []

i = 0

for font_name, data_points in data.items():
    for point in data_points:
        i += 1
        # print(i)
        classes.append(font_name)
        all_data.append(point)

# print("Splitting")
x_train, x_test, y_train, y_test = train_test_split(all_data, classes, test_size=0.33, random_state=42)

# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)

# print("Fitting...")
# fitting the model
knn.fit(x_train, y_train)

# print("Predicting...")
# predict the response
pred = knn.predict(x_test)

# print()
# print(confusion_matrix(y_test, pred))

print()
print(classification_report(y_test, pred))