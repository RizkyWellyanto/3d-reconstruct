import numpy as np
import math
import skimage.io
import h5py
import pickle


def main():

    trainImages = pickle.load(open('trainImages_filenames.p', 'rb'))
    train_rf(trainImages)


def train_rf(iminfo):
    train_images = iminfo
    opts = {'numTrees': 5}

    # Prepare 3D features to calculate entropy from
    W = []

    forest_train(train_images, W, opts)


def forest_train(images, entropy_feature, opts):
    num_trees = 5
    Y = np.arange(len(images))
    tree_models = np.zeros(num_trees)

    for i in range(num_trees):
        tree_models[i] = tree_train(images, Y, entropy_feature, opts)

        print('Trained tree', i)


def tree_train(X, Y, entropy_feature, opts):
    model = {}
    maxImgs = 5
    # Extract random features
    num_fixed_coords_feat = 500
    num_match_coords_feat = 500

    model['fixed_coords'] = np.random.rand(2, num_fixed_coords_feat)
    model['match_coords'] = np.random.rand(4, num_match_coords_feat)
    Xf = extract_feature(X, model['fixed_coords'], model['match_coords'])
    print(Xf)

    return model


def extract_feature(images, fixed_coords, match_coords):
    # Extract randomly sampled silhouette feature from images
    feature = np.zeros((len(images), fixed_coords.shape[1] + match_coords.shape[1]))
    background = 255

    for i in range(len(images)):
        print("Extracted features from image ", i)
        I = skimage.io.imread(images[i])
        dim = I.shape
        if len(dim) > 2:
            I = skimage.color.rgb2gray(I)
        I = cut_image(I)

        # Fixed coordinates
        i_coords = np.floor(fixed_coords * I.shape[1]).astype(int)
        f = I[i_coords[0, :], i_coords[1, :]] != background
        feature[i, :fixed_coords.shape[1]] = np.transpose(f)

        # Matching coordinates
        i_coords = np.floor(match_coords * I.shape[1]).astype(int)
        f = I[i_coords[0, :], i_coords[1, :]] < I[i_coords[2, :], i_coords[3, :]]
        feature[i, fixed_coords.shape[1]:] = np.transpose(f)

    return feature


def cut_image(image):
    # Crop the image into square foreground image
    background = 255

    # Calculate max and min foreground row and column
    is_foreground = image != background
    col_sum = np.sum(is_foreground, 0)
    foreground_col_idx = np.nonzero(col_sum)
    max_col = max(foreground_col_idx[0])
    min_col = min(foreground_col_idx[0])
    row_sum = np.sum(is_foreground, 1)
    foreground_row_idx = np.nonzero(row_sum)
    max_row = max(foreground_row_idx[0])
    min_row = min(foreground_row_idx[0])

    # Cut the original image and prepare the square image to fill the data
    width = max_col - min_col + 1
    height = max_row - min_row + 1
    square_size = max([width, height])
    cut_image = image[min_row:max_row, min_col:max_col]
    square_image = np.uint16(np.ones((square_size, square_size)) * background)

    # Center-align the foreground image into the cut square
    pad_col = math.floor((square_size - width) / 2)
    pad_row = math.floor((square_size - height) / 2)
    square_image[pad_row:(pad_row + height - 1), pad_col:(pad_col + width - 1)] = cut_image
    return square_image


def get_filename(fn):
    fn_str = fn.value.tobytes()[::2].decode()
    return '../mesh_data_cvpr15' + fn_str[22:]


def read_matlab():
    f = h5py.File('../output/RF_Model_Train.mat')
    RF_Model_Train = f['RF_Model_Train']
    trainImages = RF_Model_Train['trainImages']
    imageFilenames = []
    for i in range(trainImages.shape[0]):
        filename = get_filename(f[trainImages[i, 0]])
        imageFilenames.append(filename)
    pickle.dump(imageFilenames, open('trainImages_filenames.p', 'wb'))


if __name__ == '__main__':
    main()
