from sklearn.model_selection import train_test_split
from sklearn import datasets
iris = datasets.load_iris()
x = iris.data
y = iris.target
print("Iris Data set loaded..")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
print("Dataset is split into training and testing...")
print("Size of training data and its label", x_train.shape, y_train.shape)
print("Size of testing data and its label", x_test.shape, y_test.shape)
for i in range(len(iris.target_names)):
    print("Label", i, "-", str(iris.target_names[i]))
from sklearn.neighbors import KNeighborsClassifier as knn
classifier = knn(n_neighbors=1)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print("Results of Classification using K-nn with K=1")
for r in range(0,len(x_test)):
    print("Sample:", str(x_test[r]), "Actual-label:", str(y_test[r]), "Predicted-label:", str(y_pred[r]))
print("Classification Accuracy:", classifier.score(x_test, y_test))
from sklearn.metrics import classification_report, confusion_matrix
print('Confusion Matrix')
print(confusion_matrix(y_test,y_pred))
print('Accuracy Metrics')
print(classification_report(y_test,y_pred))
