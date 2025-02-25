import numpy as np
import pandas as pd

from scipy.stats import ttest_1samp
from scipy.stats import ttest_rel

from MD_Plotting import channel_map

#Perform t-test analysis of baseline/reach mean firing rates    
def t_test_intra(sess, chan, targ, overt=True):
    """
    Perform t-test between sampled baseline and reach firing rates

    Inputs:
        chan (int): Desired channel
        targ (int): Desired movement target (1 left, 2 right)
        overt (bool): Whether to display overt or covert trials

    Outputs:
        scipy.stats.stats.Ttest_relResult: direct output of t-test function
    """
    base_start = 50
    reach_start = 25
    win_len = 50

    t_test_base = []
    t_test_reach = []

    if overt:
        overt_ = "overt"
    else:
        overt_ = "covert"

    spikes_test = sess.spikes.loc[(sess.spikes['targ'] == targ) & (sess.spikes['condition'] == overt_)]
    num_trials = len(spikes_test)

    for i in range(num_trials):
        t_test_base.append(spikes_test['FR'].iloc[i][base_start:base_start+win_len,chan-1])
        t_test_reach.append(spikes_test['FR'].iloc[i][spikes_test['epoch_starts'].iloc[i][1]+reach_start:spikes_test['epoch_starts'].iloc[i][1]+reach_start+win_len,chan-1])

    t_base_mean = np.zeros(num_trials)
    t_reach_mean = np.zeros(num_trials)

    for i in range(num_trials):
        t_base_mean[i] = np.mean(t_test_base[i])
        t_reach_mean[i] = np.mean(t_test_reach[i])

    #if test_type == 'W':
    #    ans = stats.wilcoxon(t_base_mean, t_reach_mean)
    #elif test_type == "T":
    #    ans = stats.ttest_rel(t_base_mean, t_reach_mean)
    
    return ttest_rel(t_base_mean,t_reach_mean)


def chan_peak(sess, chan, targ, overt=True, bin_size=30):
    """
    Calculates peak % change in FR for a given channel and a given target and performs a 1-sample t test
    with a null hypothesis expected value of 0.

    Inputs:
        chan (int): Desired channel
        targ (int): Desired movement target (1 left, 2 right, 0 either)
        overt (bool): Whether to plot overt or covert trials
        bin_size (int): Size of bins to use for moving average

    Returns:
        np.ndarray: Array of peak change in FR values for the given channel and target
    """

    if overt:
        overt_ = "overt"
    else:
        overt_ = "covert"

    num_trials = len(sess.spikes.loc[(sess.spikes['targ'] == targ) & (sess.spikes['condition'] == overt_)]['mean_base'])

    peaks = np.zeros(num_trials)
    for i in range(num_trials):
        peaks[i] = max(absolute_peak(sess, chan, targ, i+1, overt=overt, bin_size=bin_size), key=abs)

    # Need at least 5 good trials
    if len(peaks[~np.isnan(peaks)]) < 5:
        return np.nan, peaks

    p_val = ttest_1samp(peaks, 0, nan_policy="omit")[1]
    
    if np.ma.is_masked(p_val):
        p_val = 1
        
    return p_val, peaks