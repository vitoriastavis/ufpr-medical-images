import cv2 as cv
import numpy as np
import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from joblib import load
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def read_files(ft_names_path, folds_path):
    """
    Read feature names and fold files

    Params:
    ft_names_path: path to the feature names .txt file
    folds_path: path to the folds .joblib file

    Returns:
    ft_names: list of radiomics feature names
    folds: list of tuples, each tuple is one fold containing (imgs_names, labels)
    """

    ft_names = []

    # Open file and read the content in a list
    with open(ft_names_path, 'r') as f:
        for line in f:

            # Remove linebreak
            x = line[:-1]

            # Add feature name to the list
            ft_names.append(str(x))

    # Load folds
    folds = load(folds_path)

    return ft_names, folds


def read_features(ft_path, fold_images, label):
    """
    Read features from files

    Params:
    ft_path: path to the feature names .txt file
    fold_images: list of image names for a fold
    label

    Returns:
    features: list of features
    names: list of names
    labels: list of labels
    """

    # Lists to return
    features = []
    names = []
    labels = []

    # For feature file in the path
    for ft_file in os.listdir(ft_path):

        # List of features for this image
        ft_o = []

        # Get patient name from the file
        patient = ft_file.split("-")[0]

        # Look for the patient in the fold images_names
        for k, image_name in enumerate(data):
            if patient in image_name:
                # Try to open file and read the content to a list
                try:
                    #print(f'achei o {patient} no fold')
                    with open(os.path.join(ft_path, ft_file), 'r') as f:
                        for line in f:

                            # Remove linebreak
                            x = line[:-1]

                            # Add current feature to the list
                            ft_o.append(float(x))

                    names.append(image_name)
                    labels.append(label[k])
                except:
                    pass

                # Break if patient found
                break

        if len(ft_o) != 0:
            features.append(ft_o)

    return features, names, labels

def create_dfs(fold, ft_names):
    """
    Create dataframes for the data

    Params:
    fold: tuple of (imgs_names, labels)

    Returns:
    ft_names: list of radiomics feature names
    folds: list of tuples, each tuple is one fold containing (imgs, labels)
    """

    x = fold[0]
    y = fold[1]

    # Dataframe for Otsu's features
    ft_o, index_o, label_o = read_features('features_o', x, y)
    df_o = pd.DataFrame(ft_o, columns = ft_names, index = index_o)
    df_o.insert(loc=1, column='label', value=label_o)

    # Dataframe for adaptive features
    ft_a, index_a, label_a = read_features('features_a', x, y)
    df_a = pd.DataFrame(ft_a, columns = ft_names, index = index_a)
    df_a.insert(loc=1, column='label', value=label_a)

    return df_o, df_a

def knn_classifier(clf, dataframes=None, threshold='', data={}, verbose=True):
    """
    classify features using knn

    Params:
    threshold: name of threshold filter
    data: optional dictionary data

    Returns:
    f1_list: list of f1 scores to create a dataframe
    """

    f1_list = []

    ans = {"accuracy": [], "recall": [], "f1_score": []}
    if verbose:
        print(f'Classifying KNN - {threshold}')

    # 5-fold cross validation
    for i in range(len(dataframes)):
        # Create dataframe from current folder
        df_o, df_a = dataframes[i]
        if (threshold == 'otsu'):
            df = df_o
        elif (threshold == 'adaptive'):
            df = df_a
        else:
            df = data

        # Predictor variables
        X = df.drop(['label'], axis=1).values

        # Target variables
        y = df['label'].values

        # Creates classifier with selected parameters
        neigh = KNeighborsClassifier(n_neighbors=7, metric='euclidean')

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Standardize features by removing the mean and scaling to unit variance
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        if clf = 'knn':

        # Fit the model
        neigh.fit(X_train, y_train)

        # Get the score
        score = neigh.score(X_test, y_test)

        # Predicting
        y_pred = neigh.predict(X_test)

        # Creates confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        if verbose:
            print(f'\t\t KNN - Fold {i+1} | threshold {threshold}')

            print (f'Confusion Matrix \n {cm}')
            print(classification_report(y_test, y_pred))

        # Get classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        recall = report['macro avg']['recall']
        f1 = report['macro avg']['f1-score']

        # List of metrics
        ans["accuracy"].append(score)
        ans["recall"].append(recall)
        ans["f1_score"].append(f1)

        f1_list.append(f1)

    # Print the results
    results = {"accuracy": round(np.median(ans["accuracy"]), 2), "recall": round(np.median(ans["recall"]), 2), "f1_score": round(np.median(ans["f1_score"]), 2)}
    
    if verbose:
        print("Classification done!")
        print(results)

    return f1_list


# The variable 'folds' is a list of tuples
# folds[0] is the first fold
# in each fold, there are (image_names, labels)
# which are used to build the dataframes below
# also using ft_names, which are names of the features to use as column names
ft_names, folds = read_files('ft_names.txt', 'folds.joblib')

dataframes = []
for i in range(len(folds)):
    df_o, df_a = create_dfs(folds[i], ft_names)
    dataframes.append((df_o, df_a))

    