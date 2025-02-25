# Motor_Somatotopy_2025
Analysis and plotting code for Kunigk et al. motor somatotopy paper 

# OVERVIEW

This repository is a collection of all analysis code used with the somatotopy mapping task described in "Motor somatotopy impacts imagery strategy success in human intracortical brain-computer interfaces". It also contains code to plot the decoding analysis results. 

All plotting code (to recreate paper figures) is in "Somatotopy_Final_Figures.ipynb" which is a Jupyter notebook organized by figure.

Data is uploaded to DABI and located at <https://doi.org/10.18120/4eyf-r345>


# SOMATOTOPY MAPPING ANALYSIS

All code related to the task described in Figure 2 is in the .py files located in this repository. These include:

Movement_Data.py    -   Creation of structure (class) which contains task data, helper functions for channel/trial masking
MD_Helpers          -   Helper functions for main Movement_Data class object
MD_Tuning           -   Functions related to tuning significance analysis
MD_Dimensionality   -   Functions for performing dimensionality reduction-related analysis on movement data
MD_DOM              -   Functions related to depth of modulation analysis (mostly for plotting purposes)
MD_Classification   -   Functions related to naive Bayes classification analysis
MD_Plotting         -   Functions for plotting results of analyses
MD_Tuning_Old       -   Unused functions for calculating depth of modulation/tuning significance

The Somatotopy_Final_Figures.ipynb notebook utilizes most of the aforementioned functions to generate the final figures in the paper.


# DECODING ANALYSIS

The final figure plotting notebook assumes that decoder training and accuracy quantification was performed beforehand on the experimental session data provided via DABI. Data is assumed to be formatted as saved MATLAB variables, with individual index-matched arrays storing R^2 values from fully cross-validated decoder testing with recorded activity from both arrays, only the lateral array, and only the medial array. Final values from the paper may differ slightly with results from the notebook depending on the exact sessions used and on decoder training initialization/regularization/etc.

The DABI link contains experimental session data for both the 3D arm and hand task as well as the 2D cursor translation and click task. For recreation of Figures 4 and 5 in the paper, decoder training must be performed and the resulting R^2 values saved in MATLAB arrays for the provided notebook to process the data into figures.
