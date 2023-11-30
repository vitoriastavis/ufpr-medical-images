# medical-images-processing

Immunohistochemistry image classification

Usage: python3 classification.py features_o.tar.gz features_a.tar.gz features_op.tar.gz features_ap.tar.gz folds.tar.gz

features_o.tar.gz: compressed file with directory of features Otsu, images not preprocessed
features_a.tar.gz: compressed file with directory of features adaptive, images not preprocessed
features_op.tar.gz: compressed file with directory of features Otsu, images preprocessed
features_ap.tar.gz: compressed file with directory of features adaptive, images not preprocessed
folds.tar.gz: .tar.gz file containing a folds.joblib file that has information
              dividing the image names in folds for classification

Alternative usage:
python3 classification.py features_o features_a features_op features_ap folds.joblib
if your files are already unpacked
    
The script results in to tables: results_knn.csv and results_mlp.csv

If you want to see only those two final tables and wish to hide the confusiom matrices
and classification reports for all classifications, set verbose = False!

(If you don't have directories with features for Otsu's threshold and adaptive threshold for both not
processed and for processes images, and a folds.tar.gz file, run feature_extraction.tar.gz first)
