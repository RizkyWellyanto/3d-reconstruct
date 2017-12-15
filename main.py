import numpy as np
from random_forest_matching.RandomForestUtils import extract_feature
from sklearn.ensemble import RandomForestClassifier
import scipy.io
from copy import deepcopy

# Return in-order list of classes, as well as the labeling of the class names of length N=class_length
def map_labels(class_name, class_list, class_set):
    """

    :param class_name: array of image filepaths for the class
    :param class_list: for ordering. Can pass in previous set to build on
    :param class_set: for uniqueness. Same as above
    :return:
    """
    class_obj_names = [name.split('\\')[0] for name in class_name]

    class_list = deepcopy(class_list)
    class_set = deepcopy(class_set)
    for obj in class_obj_names:
        if obj not in class_set:
            class_set.add(obj)
            class_list.append(obj)

    # Map obj_name -> enumerated
    class_labels = [class_list.index(class_obj_names[i]) for i in range(len(class_obj_names))]
    return class_labels, class_list, class_set, class_obj_names

# load the dataset
TRAIN_FILE          = './voxelize_meshes/data_for_python/train_im_paths_300.txt'
TRAIN_NAMES_FILE    = './voxelize_meshes/data_for_python/train_im_names_300.txt'
TEST_FILE           = './voxelize_meshes/data_for_python/test_im_paths_1800.txt'
TEST_NAMES_FILE     = './voxelize_meshes/data_for_python/test_im_names_1800.txt'

# Load train data
train_paths = [line.rstrip('\n') for line in open(TRAIN_FILE)]
train_names = [line.rstrip('\n') for line in open(TRAIN_NAMES_FILE)]

# load test_data
test_paths = [line.rstrip('\n') for line in open(TEST_FILE)]
test_names = [line.rstrip('\n') for line in open(TEST_NAMES_FILE)]

# Build labels for 10500 images, each image being indexed with the enumeration of the objects
train_labels, train_list, train_set, train_obj_names = map_labels(train_names, [], set())
test_labels, test_list, test_set, test_obj_names = map_labels(test_names, train_list, train_set) # Repeat with testing data

# get a training data X with size N_TRAIN x 1000
train_features = extract_feature(train_paths)

# run it through random forest X and Y with criterion = entropy
model = RandomForestClassifier(n_estimators=5,
                               max_depth=200,
                               random_state=0,
                               criterion='entropy',
                               max_leaf_nodes=5,
                               min_samples_leaf=2)
model.fit(train_features, train_labels)

# -------------------------------------------------------------------------------------- TESTING
# extract feature from test data
test_features = extract_feature(test_paths)

# run it through the random forest get prediction
out = model.predict(test_features)

print("Training class labels:")
for i in range(len(train_list)):
    print("{} - {}".format(i, train_list[i]))

# Create histogram of testing image counts, using dictionary. Key: obj_name, Value: obj_count
test_dict = dict.fromkeys(test_set, 0)
for i in range(len(test_obj_names)):
    obj = test_obj_names[i]
    test_dict[obj] += 1

# Count the amount of times the object has the correct label
predict_dict = dict.fromkeys(test_set, 0)
for i in range(len(test_obj_names)):
    obj = test_obj_names[i]
    print("{} - {} - {}".format(i, out[i], obj)) # Image#, Predicted Label, Object name
    if out[i] == test_labels[i]:
        predict_dict[obj] += 1

# Print correct predictions if nonzero. This is N_EXACT_MATCHES / N_TOTAL_TEST
print("Training depth maps: {}".format(len(train_names)))
print("Testing depth maps: {}".format(len(test_names)))
for k, v in sorted(predict_dict.items()):
    if v > 0:
        total_count = test_dict[k]
        print("{} = {} / {}".format(k, v, total_count))

