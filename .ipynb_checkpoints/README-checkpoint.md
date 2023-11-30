# medical-images-processing

Immunohistochemistry image classification

Usage: python3 classification.py verbose
to show all confusion matrices and classification reports

or python3 classification.py noverbose
to hide all confusion matrices and classification reports

The script results in two tables: results_knn.csv and results_mlp.csv

(If you don't have directories with features for Otsu's threshold and adaptive threshold for both not
processed and for processes images, and a folds.tar.gz file, run feature_extraction.tar.gz first)
