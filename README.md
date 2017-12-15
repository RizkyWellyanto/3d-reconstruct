# 3d-reconstruct

## Descrption
This repository contains code for a portion of the "Completing 3D Object Shape from One Depth Image" paper found here: http://vision.cs.uiuc.edu/projects/shaperecon/ . We are able to predict 3D meshes from 2D RGBD images, although the accuracy is not exactly quantifiable in its current state. However, viewing the images and predicted labels vs correct labels for the testing data, one can understand how the random forest works with the 2D RGBD images and why the outputs are not always close to correct.

## Code
The code we have written is primarily Python with occasional MATLAB. It is located in `/data_processing`, `/random_forest_matching`, `/voxelize_meshes`, and `main.py`. The main program uses the majority of the code, but for some functions we implemented, we were unable to integrate with the random forest classifier due to the scope and time of the project as well as the heavily-modified nature of the original random forest code provided in the paper. The `/ThirdParty` directory contains 3rd-party libraries used in the original paper, as well as some select higher-level abstraction code written by Jrock, plus a `/cell2text` library which was used for converting MATLAB cell arrays to text files.
