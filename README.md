# Fine Grained Action Recognition in Sports Videos
## cs229project submission - anagpal1@stanford.edu

## pose
This is where we annotate all our videos with 2D pose estimates and store a lot of intermediate pkl files for analysis. Also has analysis and feature generation code.

## pose_track
We annotate videos with tracking bounding boxes using deepsort. This is also where we have code that matches poses with deepsort's bounding boxes

## basket
This is the code we run on aws to estimate basket bounding box from basket holder's banner ad

## model
Code for feature formatting and setup of SVM and CNN train and test

## data_handling
scripts and code to run annotations and feature generation on hundreds of videos

## deepsort_opensource
externally linked submodule for tracking. Used by pose module

## pose_opensource
externally linked submodule for pose estimation. Used by pose_track module

