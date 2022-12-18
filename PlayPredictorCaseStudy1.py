import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

def MarvellousPlayPredictor(data_path):

    # Step 1 : Load Data
    data = pd.read_csv(data_path , index_col=0)

    print("Size of Actual dataset : ",len(data))

    # Step 2 : Clean , Prepare and manipulate data
    feature_names = ['Whether' , 'Temperature']

    print("Names of features : ",feature_names)

    whether = data.Whether
    Temperature = data.Temperature
    play = data.Play

    # creating label 
    le = preprocessing.LabelEncoder()

    # Converting string labels into numbers
    temp_encoded = le.fit_transform(Temperature)
    label = le.fit_transform(play)
    whether_encoded = le.fit_transform(whether)

    # conbining whether and temp into single list of tuples
    features = list(zip(whether_encoded,temp_encoded))

    # Step 3 : Train data
    model = KNeighborsClassifier(n_neighbors=3)

    # Train the model  using the training sets
    model.fit(features , label)

    #Step 4 : Test Data
    predicted = model.predict([[0,2]])
    print(predicted)

    if predicted:
        print("Yes")

    else:
        print("No")


def main():
    print("Machine Learning Application")

    print("Play Predictor Application using KNN Algorithm")

    MarvellousPlayPredictor("PlayPredictor.csv")



if __name__ == "__main__":
    main()