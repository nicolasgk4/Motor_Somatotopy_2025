import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis

from MD_Helpers import create_reach_arr
from MD_Plotting import FA_plot

# Perform PCA on the reach data and produce a scree plot for identification of dimensionality
def PCA_scree(sess, plot_data=True):
    """
    Performs simple PCA on reach data from motor channels and produces a scree plot.

    Inputs:
        plot_data (bool): Whether to plot the data

    Returns:
        (np.ndarray): Transformed data
    """

    # Perform PCA
    PCA_reach = PCA()
    scores_PCA = PCA_reach.fit_transform(sess.FR_reach_O)

    if plot_data:
        # Scree plot
        fig = plt.figure(figsize=(6,3))
        ax = fig.add_subplot(111)
        ax.plot(PCA_reach.explained_variance_ratio_[:20]*100, 'o')
        ax.set_xlabel("Components")
        ax.set_ylabel("Variance Explained (%)")

    return PCA_reach.explained_variance_ratio_


def alignment_index(sess, data_type, n_components=2, plot_data=True, use_array="both"):
    """
    Function to calculate alignment index between n-dimensional target-averaged activity.

    Inputs:
        data_type (str):    Data to analyze. One of 3 options:
                                "FR": movement period firing rate matrix 
                                "FA": the same converted to 3-dimensional factor space
                                "Avg_FA": the average of "FA" across targets
        n_components (int): Number of components to analyze. Defaults to 2 based on visual data inspection.
        plot_data (bool):   Whether to plot resulting confusion matrix
        use_array (str): Which array to calculate data for. Can be "both" (default), "lateral", or "medial" to block out the other channels

    Returns:
        (np.ndarray):       2D numpy matrix of size (num_targets x num_targets) containing alingment indices
    """

    # Load data
    if data_type == "FR":
        if use_array == "both":
            FR_data = sess.FR_reach_O
        elif use_array == "lateral":
            test_lat_mask = sess.chan_mask * [[True]*128 + [False]*128][0]
            FR_data = create_reach_arr(sess.spikes, test_lat_mask)
        elif use_array == "medial":
            test_med_mask = sess.chan_mask * [[False]*128 + [True]*128][0]
            FR_data = create_reach_arr(sess.spikes, test_med_mask)

        test = FR_data.T

        trial_list = list(sess.spikes[sess.spikes["condition"] == "overt"].index)
        len_trial = sess.FA_reach_len

        # Add trial number indicator
        test = np.row_stack((test, np.array([[i] * len_trial for i in trial_list]).flatten()))

        # Add target number indicator
        test = np.row_stack((test, np.array([[sess.spikes["targ"][i]] * len_trial for i in trial_list]).flatten()))

        targ_data = []

        # Store firing rate activity data for each target as separate list elements
        for targ in sess.targ_dict.keys():
            targ_data.append(test[:-2, test[-1] == targ].T)

    elif data_type == "FA":
        test = FA_plot(sess, plot_data=False, use_array=use_array)[0]

        # Separate into target data
        targ_data = []
        for targ in sess.targ_dict.values():
            targ_data.append(test[test["Target"] == targ][["F1", "F2", "F3"]].to_numpy(dtype=float))

    elif data_type == "Avg_FA":
        test = FA_plot(sess, plot_data=False, use_array=use_array)[1]

        # Separate into target data
        targ_data = []
        for targ in sess.targ_dict.values():
            targ_data.append(test[test["Target"] == targ][["F1", "F2", "F3"]].to_numpy(dtype=float))
        


    # Compute individual PCAs for each target
    Xs = []
    Vs = []
    Cs = []

    Ss = []

    SVDs = []

    for targ in targ_data:
        pca = PCA(n_components=n_components)
        Xs.append(pca.fit_transform(targ))
        Vs.append(pca.components_.T)

        targ_cov = np.cov(targ.T)       
        Cs.append(targ_cov)

        U, S, Vh = np.linalg.svd(targ_cov, full_matrices=False)

        Ss.append(S)


    # Compare to other target subspaces
    alignments = []

    for i in range(len(Vs)):
        alignments.append([])
        for j in range(len(Vs)):
            # Alignment index calculation
            alignments[i].append(np.trace(np.matmul(Vs[i].T,np.matmul(Cs[j],Vs[i])))/sum(Ss[j][:2]))

    # Convert diagonal values to nan
    for i in range(len(alignments)):
        alignments[i][i] = np.nan

    # Z-score alignments
    alignments = np.array(alignments)
    # alignments = (alignments - np.nanmean(alignments))/np.nanstd(alignments)


    if plot_data:
        # Plot alignment matrix
        fig = px.imshow(alignments, zmin=0, zmax=1, text_auto=True, labels=dict(x="Projected Into", y="Trajectory", color="Alignment Index <br> (not Z-scored)"), color_continuous_scale="magma_r",
                        x=list(sess.targ_dict.values()), y=list(sess.targ_dict.values()))
        
        fig.update_xaxes(side="top")
        fig.update_layout(width=600, height=600)
        fig.show()

    return alignments