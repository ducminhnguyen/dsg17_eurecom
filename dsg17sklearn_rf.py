import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import sklearn.ensemble as ens
import warnings
warnings.filterwarnings('ignore')
def readData(path, nrows = 1000):
#     data = np.genfromtxt(path, delimiter=',')
    input_data = pd.read_csv(path, nrows=nrows)
    input_data["genre_id"] = input_data["genre_id"].astype("category")
    input_data["media_id"] = input_data["media_id"].astype("category")
    input_data["album_id"] = input_data["album_id"].astype("category")
    input_data["context_type"] = input_data["context_type"].astype("category")
    input_data["platform_name"] = input_data["platform_name"].astype("category")
    input_data["platform_family"] = input_data["platform_family"].astype("category")
    input_data["listen_type"] = input_data["listen_type"].astype("category")
    input_data["user_gender"] = input_data["user_gender"].astype("category")
    input_data["user_id"] = input_data["user_id"].astype("category")
    input_data["artist_id"] = input_data["artist_id"].astype("category")
    return input_data
    
train_data = readData("./train.csv", nrows=None)
# train_data = train_data.sample(frac=0.3)
print(train_data.shape)
classifier = ens.RandomForestClassifier(n_estimators=30, criterion='gini', max_depth=None, min_samples_split=2, 
                                        min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', 
                                        max_leaf_nodes=None, bootstrap=True, oob_score=False, 
                                        n_jobs=2, random_state=None, verbose=0, warm_start=False, class_weight=None)
classifier.fit(train_data.drop(["is_listened"], axis=1).values, train_data["is_listened"].values)

import pickle as pkl
pkl.dump(classifier, open("dsg17_rf.pkl", "wb"))