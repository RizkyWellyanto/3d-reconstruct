%% Run this to produce iminfo and voxelized meshes on the training subset. Randomly permutes and subsamples the voxel matrix as well
close all

%% User configurable variables
VOXEL_SIZE = 50; % Size for voxelized representation in XYZ space (V_S x V_S x V_S)
N_SUBSAMPLE = 250; % Number of images to randomly subsample. 300 images in training subset

%% Setup path variables
RECONSTRUCT_DIR = 'C:\Users\unFearing\Documents\UIUC Senior Year\CS 445\3d-reconstruct\'; % Path to home dir of project
MESH_DATA = [RECONSTRUCT_DIR 'data\mesh_data_cvpr15\'];
TRAIN_DATA = [MESH_DATA 'Train2Subset\'];
CLASS = [TRAIN_DATA 'NovelClass'];
MODEL = [TRAIN_DATA 'NovelModel'];
VIEW = [TRAIN_DATA 'NovelView'];

% Add libraries
addpath([RECONSTRUCT_DIR 'ThirdParty\IO']) % for camera functions
addpath([RECONSTRUCT_DIR 'ThirdParty\JRock']) % Voxel/Mesh related high-level code
addpath([RECONSTRUCT_DIR 'ThirdParty\toolbox_graph']) % for 'read_mesh()' function
addpath([RECONSTRUCT_DIR 'ThirdParty\Voxelize\mesh2voxel']) % for 'polygon2voxel()' function

%% Begin voxelize program. Modeled off of 'Main/train_RF.m' lines 63:76 --------------------------------------
% Create iminfo structs. Warnings are due to training subset not containing all images (as expected). Ignore.
train_paths = {CLASS MODEL VIEW};
iminfo = generate_iminfo(train_paths);

% Voxelize the training meshes. Returns 1xN cell array, N = training set size, and contains 50x50x50 logicals.
% Better understood as Nx(50x50x50) logical matrices
voxelized_training_cells = voxelizeAllImgs(iminfo, iminfo.images, VOXEL_SIZE);

% Vectorize these logical voxels into one Nx125000 matrix
n_train = numel(voxelized_training_cells);
vox_train_vec = false(n_train, VOXEL_SIZE^3); % Init empty logical matrix, Nx125000 shape
for i=1:n_train
    vox_train_vec(i, :) = voxelized_training_cells{i}(:);
end

% Permute and return k unique integers in range [1,n], where n = n_train, k = min(N_SS, n)
perm = randperm(n_train, min(N_SUBSAMPLE, n_train)); 
subsample_vox = vox_train_vec(perm, :);

% Save training image info, original voxelized training matrix, & randomly permuted subsample voxelized training matrix
save('train_iminfo', 'iminfo');
save('train_vox_orig', 'vox_train_vec');
save('train_vox_perm', 'subsample_vox', 'perm'); % Save the permutation ordering as well