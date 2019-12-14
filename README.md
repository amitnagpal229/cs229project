Code is organized as follows:

Section 1:

pose - this directory structure was originally forked from the keras implementation of Cao et. al's multi purpose pose estimation open source model and code. Because pose was our most fundamental features for the project, most of our code was developed within this directory. Key files specific to this project are action.py, pose_track.py, pose_track_analysis.py, basket_bbox.py, filter.py, jumps_plot.py, truth_values.py, coverage.py, processing_action.py and the *.sh script files to run code over hundreds of videos.


deep_sort_pytorch - this was originally cloned as the opensource implementation of deep sort. See git for source of fork. Files related to work done for this project are bbox.py and *.sh files

pose_filter - only two files here are related to the project - filter.py and bbox_label.py

model - we used libsvm for all SVM training and keras for CNN training

