import numpy as np
from scipy.optimize import nnls
from sklearn.decomposition import NMF
import scipy.io

def get_entropy_features(data):
    """
    This function takes voxels training data and returns the entropy features

    :param  data            : N_TRAIN x 125000
    :return entropy_features: N_TRAIN x N_COMP
    """

    # set up dimension constants
    N_TRAIN = data.shape[0]
    N_COMP = 50

    # pre allocate the entropy features array
    entropy_features = np.zeros((N_TRAIN, N_COMP))

    # apply nnmf dimensionality reduction
    model = NMF(n_components=50, init='random', random_state=0)
    model.fit(data)
    H = model.components_

    # get the top 50 components using non-neg least squares
    for i in range(N_TRAIN):
        x, rnorm = nnls(H.transpose(), data[i])
        entropy_features[i, :] = x

    return entropy_features

def discretize_entropy_features(entropy_features):
    """
    This function takes the entropy feature and returned a discretized version of it based on 3 values
        2: Anything above than 66th percentile
        1: 33th to 66th percentile
        0: the rest of the matrices

    :param  entropy_features                : N_TRAIN x N_COMP
    :return discretized_entropy_features    : N_TRAIN x N_COMP
    """

    # preallocate memory for the discretized entropy feature
    discrete_entropy_features = np.ones(entropy_features.shape)

    # set values > 66th percentile as 2, > 33th percentile as 1, the rest as 0
    nonzero_entropy_features = entropy_features[entropy_features != 0]
    discrete_entropy_features[entropy_features < np.percentile(nonzero_entropy_features, 33)] = 0
    discrete_entropy_features[entropy_features > np.percentile(nonzero_entropy_features, 66)] = 2

    return discrete_entropy_features


# this is just for debugging purposes
data = scipy.io.loadmat("sampleVoxels.mat")
entropy_features = get_entropy_features(np.array(data['sampleVoxels']))
output = discretize_entropy_features(entropy_features)
print(output.shape)
print(output)