import numpy as np
from random_forest_matching.RandomForestUtils import extract_feature
from sklearn.ensemble import RandomForestClassifier
import scipy.io

# load the dataset
train_data = scipy.io.loadmat("./voxelize_meshes/train_iminfo.mat")
train_paths = np.array(train_data['images'])
train_labels = np.array(range(300)) // 10

# get a training data X with size N_TRAIN x 1000
train_features = extract_feature(train_paths)

# run it through random forest X and Y with criterion = entropy
model = RandomForestClassifier(max_depth=200, random_state=0, criterion='entropy', max_leaf_nodes=5, min_samples_leaf=2)
model.fit(train_features, train_labels)


# load test_data

# extract feature from test data

# run it through the random forest get prediction


