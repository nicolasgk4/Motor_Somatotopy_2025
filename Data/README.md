# Overview

Place data from DABI (https://doi.org/10.18120/4eyf-r345) in this folder.

Should include "Somatotopy_Mapping" and "Decoding_Analysis" folders.

# Somatotopy Mapping 

The Somatotopy_Final_Figures notebook should be able to recreate the figures from the paper with the data saved in this directory from DABI directly, without modification.

# Decoding Analysis

After decoding training/testing is complete, save data in this directory.

As-is, the data loaded in the Somatotopy_Final_Figures notebook was originally saved as arrays of $R^2$ values in MATLAB. Each session loaded saved the arrays with the following naming convention: "R2_{subject#}", "R2_{subject#}_block_lat", and "R2_{subject#}_block_med" (I don't know how to stop the markdown formatting, maybe load this as a .txt). The notebook may require modification to work with different data formats/variable names.
