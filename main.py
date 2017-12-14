import numpy as np
from random_forest_matching.RandomForestUtils import extract_feature
from sklearn.ensemble import RandomForestClassifier
import scipy.io

# load the dataset
TRAIN_FILE = './voxelize_meshes/train_im_paths.txt'
TEST_FILE = './voxelize_meshes/view_test_im_paths.txt'
train_paths = [line.rstrip('\n')for line in open(TRAIN_FILE)]

train_labels = np.array(range(300)) // 10

# get a training data X with size N_TRAIN x 1000
train_features = extract_feature(train_paths)

# run it through random forest X and Y with criterion = entropy
model = RandomForestClassifier(max_depth=200, random_state=0, criterion='entropy', max_leaf_nodes=5, min_samples_leaf=2)
model.fit(train_features, train_labels)

# load test_data
test_paths = [line.rstrip('\n')for line in open(TEST_FILE)]

# extract feature from test data
test_features = extract_feature(test_paths)

# run it through the random forest get prediction
out = model.predict(test_features)

print(out)

a=5

