%% Run this to produce iminfo and voxelized meshes on the training subset. Randomly permutes and subsamples the voxel matrix as well
% Uses some code from Jrock08's repo, plus ThirdParty libraries (for Jrock08's code to run), as well as /Library/Util/Voxelize,
% to perform the complex mesh->voxelization process, which navigates the intricate file structure to find the 
% depth maps, camera viewing angle, etc for the iminfo struct beyond our scope.
close all

%% User configurable variables
VOXEL_SIZE = 50; % Size for voxelized representation in XYZ space (V_S x V_S x V_S)

%% Setup path variables
RECONSTRUCT_DIR = 'C:\Users\unFearing\Documents\UIUC Senior Year\CS 445\3d-reconstruct\'; % Path to home dir of project
MESH_DATA = [RECONSTRUCT_DIR 'data\mesh_data_cvpr15\'];
TRAIN_DATA = [MESH_DATA 'Train2\'];
TRAIN_CLASS = [TRAIN_DATA 'NovelClass'];
TRAIN_MODEL = [TRAIN_DATA 'NovelModel'];
TRAIN_VIEW = [TRAIN_DATA 'NovelView'];
TEST_DATA = [MESH_DATA 'TestSubset\'];
TEST_CLASS = [TEST_DATA 'NovelClass'];
TEST_MODEL = [TEST_DATA 'NovelModel'];
TEST_VIEW = [TEST_DATA 'NovelView'];

%{ 
Add libraries:
IO - reading files
GLCamera - camera funcitons 
JRock - high-level code by jrock08
toolbox_graph - for read_mesh()
Voxelie - for polygon2voxel() 
%}
addpath(genpath([RECONSTRUCT_DIR 'ThirdParty'])) % Generate all subfolder paths and add

%% Begin voxelize program. Modeled off of 'Main/train_RF.m' lines 63:76 --------------------------------------
% Create iminfo structs. Warnings are due to training subset not containing all images (as expected). Ignore.
train_paths = {TRAIN_CLASS TRAIN_MODEL TRAIN_VIEW};
train_iminfo = generate_iminfo(train_paths);
save('train_iminfo_10500', 'train_iminfo');
dlmcell('train_im_paths_10500.txt', train_iminfo.images(:))
dlmcell('train_im_names_10500.txt', train_iminfo.names(:))

test_paths = {TEST_CLASS TEST_MODEL TEST_VIEW};
test_iminfo = generate_iminfo(test_paths);
save('test_iminfo_1800', 'test_iminfo');
dlmcell('test_im_paths_1800.txt', test_iminfo.images(:))

%% Remainder of code is related to training process:
% Voxelize the training meshes. Returns 1xN cell array, N = training set size, and contains 50x50x50 logicals.
% Better understood as Nx(50x50x50) logical matrices
voxelized_training_cells = voxelizeAllImgs(train_iminfo, train_iminfo.images, VOXEL_SIZE);

% Vectorize these logical voxels into one Nx125000 matrix
n_train = numel(voxelized_training_cells);
vox_train_vec = false(n_train, VOXEL_SIZE^3); % Init empty logical matrix, Nx125000 shape
for i=1:n_train
    vox_train_vec(i, :) = voxelized_training_cells{i}(:);
end

% Permute and return k unique integers in range [1,n], where n = n_train, k = min(N_SS, n)
perm = randperm(n_train, min(N_SUBSAMPLE, n_train)); 
subsample_vox = vox_train_vec(perm, :);

% Save original voxelized training matrix, & randomly permuted subsample voxelized training matrix
save('train_vox_orig', 'vox_train_vec');
save('train_vox_perm', 'subsample_vox', 'perm'); % Save the permutation ordering as well