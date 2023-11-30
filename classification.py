import numpy as np
import os, subprocess, sys
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from joblib import load
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

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
        for k, image_name in enumerate(fold_images):
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

def create_dfs(path_o, path_a, fold):
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
    ft_o, index_o, label_o = read_features(path_o, x, y)
    df_o = pd.DataFrame(ft_o, index = index_o)
    df_o.insert(loc=1, column='label', value=label_o)

    # Dataframe for adaptive features
    ft_a, index_a, label_a = read_features(path_a, x, y)
    df_a = pd.DataFrame(ft_a, index = index_a)
    df_a.insert(loc=1, column='label', value=label_a)

    return df_o, df_a

def classifier(method, dataframes=None, threshold='', verbose=True):
    """
    classify features using knn

    Params:
    threshold: name of threshold filter
    data: optional dictionary data

    Returns:
    f1_list: list of f1 scores to create a dataframe
    """

    f1_list = []

    ans = {"accuracy": [], "precision": [], "recall": [], "f1_score": []}

    if verbose:
        print("-----------------------------------")
        print(f'Classifying with {method} - {threshold} \n')

    # 5-fold cross validation
    for i in range(len(dataframes)):
        # Create dataframe from current folder
        (df_o, df_a) = dataframes[i]
        if (threshold == 'otsu'):
            df = df_o
        elif (threshold == 'adaptive'):
            df = df_a

        # Predictor variables
        X = df.drop(['label'], axis=1).values

        # Target variables
        y = df['label'].values

        # Creates classifier with selected parameters
        if method == 'knn':
            clf = KNeighborsClassifier(n_neighbors=7, metric='euclidean')
        elif method == 'mlp':
            clf = MLPClassifier(max_iter=300, activation='logistic', solver='adam')
        else:
            print('invalid method, try knn or mlp')
            exit()

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Standardize features by removing the mean and scaling to unit variance
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Fit the model
        clf.fit(X_train, y_train)

        # Get the score
        score = clf.score(X_test, y_test)

        # Predicting
        y_pred = clf.predict(X_test)

        # Creates confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        if verbose:           
            print(f'\t\t {method} - Fold {i+1} | threshold {threshold}')
            print (f'Confusion Matrix \n {cm}')
            print(classification_report(y_test, y_pred))     

        # Get classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        precision = report['macro avg']['precision']
        recall = report['macro avg']['recall']
        f1 = round(report['macro avg']['f1-score'], 3)

        # List of metrics
        ans["accuracy"].append(round(score, 3))
        ans["precision"].append(round(precision, 3))
        ans["recall"].append(round(recall, 3))
        ans["f1_score"].append(f1)

        f1_list.append(f1)

    # Print the results
    results = {"accuracy": round(np.mean(ans["accuracy"]), 3),
               "precision": round(np.mean(ans["precision"]), 3),
               "recall": round(np.mean(ans["recall"]), 3),
               "f1_score": round(np.mean(ans["f1_score"]), 3)}

    if verbose:
        print("Classification done! Average metrics:")
        print(results)
        print()
        print("-----------------------------------")

    return f1_list

def check_parameters(args):       

    if len(args) != 6: 
        print("\nUsage: python classification.py features_o features_a features_op features_ap folds.tar.gz\n")
        print("features_o: path for directory of features Otsu, images not preprocessed")
        print("features_a: path for directory of features adaptive, images not preprocessed")
        print("features_op: path for directory of features Otsu, images preprocessed")
        print("features_ap: path for directory of features adaptive, images not preprocessed")
        print("folds.tar.gz: .tar.gz file containing a folds.joblib file that has folds information\n")
        sys.exit(1)      
 
    paths = []
    
    for path in args[1:5]:
        if os.path.isdir(path):  
            paths.append(path)
        elif path[-7:] == '.tar.gz':
            command = f'tar xf {path}'
            subprocess.run(command, shell = True, executable="/bin/bash")
            paths.append(path[:-7])
        else:
            print(f'Argument provided {path} is not a directory nor a .tar.gz')
            sys.exit(1)   
            
    if args[5][-7:] == '.joblib': 
        paths.append(args[5])
    elif args[5][-7:] == '.tar.gz':
        command = f'tar xf {args[5]}'    
        subprocess.run(command, shell = True, executable="/bin/bash")
        paths.append('folds.joblib')
    else:
        print(f'{args[5]} is neither a .tar.gz or a .joblib')
        sys.exit(1)    
        
    return paths

if __name__ == '__main__':

    args = sys.argv
    
    # Verify all the arguments for 
    paths = check_parameters(args)
        
    # The variable 'folds' is a list of tuples
    # folds[0] is the first fold
    # in each fold, there are (image_names, labels)
    # which are used to build the dataframes below
    folds = load(paths[-1])

    print('Preparing data...')
    # Create dfs for non preprocessed images
    dataframes_np = []
    for i in range(len(folds)):
        df_o, df_a = create_dfs(paths[0], paths[1], folds[i])
        dataframes_np.append((df_o, df_a))

    # Create dfs for preprocessed images
    dataframes_p = []
    for i in range(len(folds)):
        df_o, df_a = create_dfs(paths[2], paths[3], folds[i])
        dataframes_p.append((df_o, df_a))

    # Print information along classifications
    verbose = True
    
    # Classification for non processed images
    f1_knn_otsu_np = classifier('knn', dataframes_np, 'otsu', verbose)
    f1_knn_adapt_np = classifier('knn', dataframes_np, 'adaptive', verbose)
    f1_mlp_otsu_np = classifier('mlp', dataframes_np, 'otsu', verbose)
    f1_mlp_adapt_np = classifier('mlp', dataframes_np, 'adaptive', verbose)

    # Classification for processed images
    f1_knn_otsu_p = classifier('knn', dataframes_p, 'otsu', verbose)
    f1_knn_adapt_p = classifier('knn', dataframes_p, 'adaptive', verbose)
    f1_mlp_otsu_p = classifier('mlp', dataframes_p, 'otsu', verbose)
    f1_mlp_adapt_p = classifier('mlp', dataframes_p, 'adaptive', verbose)

    header = ['Otsu (NP)', 'Adaptive (NP)']
    levels = ['Not preprocessed', 'Preprocessed']

    # Create result dataframes
    results_np_knn = pd.DataFrame(np.transpose([f1_knn_otsu_np, f1_knn_adapt_np]), columns = header)
    results_np_mlp = pd.DataFrame(np.transpose([f1_mlp_otsu_np, f1_mlp_adapt_np]), columns = header)

    results_p_knn = pd.DataFrame(np.transpose([f1_knn_otsu_p, f1_knn_adapt_p]), columns = header)
    results_p_mlp = pd.DataFrame(np.transpose([f1_mlp_otsu_p, f1_mlp_adapt_p]), columns = header)

    results_knn = pd.concat([results_np_knn, results_p_knn], axis=1, keys =levels)
    results_mlp = pd.concat([results_np_mlp, results_p_mlp], axis=1, keys=levels)

    results_knn.index = [i+1 for i in results_knn.index]
    results_knn.index.name = 'Fold'
    results_knn.to_csv('results_knn.csv', header=True, index=True)

    results_mlp.index = [i+1 for i in results_mlp.index]
    results_mlp.index.name = 'Fold'
    results_mlp.to_csv('results_mlp.csv', header=True, index=True)

    # Print results
    print('\t\t\tKNN - F1-score')
    print(results_knn)
    print()

    print('\t\t\tMLP - F1-score')
    print(results_knn)