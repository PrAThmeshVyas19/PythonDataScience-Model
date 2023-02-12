from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

def SVM():
    cancer = datasets.load_breast_cancer()

    print("Features of cancer dataset is : ",cancer.feature_names)

    print("Labels of cancer dataset is : ",cancer.target_names)

    print("Shape of dataset is : ",cancer.data.shape)

    print("First 5 records are : ")
    print(cancer.data[0:5])

    print("Target of dataset : ",cancer.target)

    X_train,X_test,Y_train,Y_test = train_test_split(cancer.data , cancer.target , test_size = 0.3 , random_state = 109)

    clf = svm.SVC(kernel = 'linear')

    clf.fit(X_train , Y_train)

    Y_pred = clf.predict(X_test)

    print("Accuracy of the model is : ",metrics.accuracy_score(Y_test , Y_pred)*100)


def main():
    print("_________________________Support vector Machine ______________________________")

    SVM()

if __name__ == "__main__":
    main()