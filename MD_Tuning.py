import numpy as np
import pandas as pd

from scipy.stats import ttest_1samp

from MD_Plotting import channel_map

# Old significance methods
from MD_Tuning_Old import t_test_intra, chan_peak

# Calculate significance of tuning based on peak reach FR method
def absolute_peak(sess, chan, targ, trial, overt=True, bin_size=30):
    """
    Calculates the absolute change in firing rate for a given trial and channel.

    Inputs:
        chan (int): Desired channel
        targ (int:) Desired movement target (1 left, 2 right)
        trial (int): Desired trial (in a list of all trials that satisfy the desired target and condition, i.e. out of 40)
        overt (bool): Whether to display overt or covert trials
        bin_size (int): Size of bins to use for moving average

    Returns:
        np.ndarray: Array containing binned averages
    """

    if overt:
        overt_ = "overt"
    else:
        overt_ = "covert"

    trial_chosen = sess.spikes.loc[(sess.spikes['targ'] == targ) & (sess.spikes['condition'] == overt_)].iloc[trial-1]

    test_FR = trial_chosen["FR"][:,chan-1]

    # Adding low firing protection
    if max(test_FR) < 10:
        return [np.nan]

    # Estimates, but not bad
    reach_start = trial_chosen["epoch_starts"][1]
    reach_end = trial_chosen["epoch_starts"][2] 
    test_reach = test_FR[reach_start:reach_end]

    test_base = trial_chosen["mean_base"][chan-1]

    # if test_base == 0:
    #     test_base = 0.5

    test_change = test_reach - test_base # Subtract mean baseline firing rate
    # test_norm = test_change / test_base # Normalize by mean baseline FR
    # test_norm is now the reach FR represented as change from baseline (multiply by 100 for % change)
    
    # num_bins = int(np.floor(150/bin_size)) # lose some end samples if not divisor of 150

    # peak = np.zeros(num_bins)

    # for i in range(num_bins): # [bin_size]-sample moving window, up to last possible [bin_size]-sample window
    #     peak[i] = np.mean(test_norm[i*bin_size:(i+1)*bin_size]) # Calculate mean value in [bin_size]-sample window

    window_length = 25 # Samples
    window_starts = [15, 25, 35, 45, 55] # 300, 500, 700, 900, 1100 ms

    peak = np.zeros(len(window_starts))

    for i in range(len(window_starts)):
        peak[i] = np.mean(test_change[window_starts[i]:window_starts[i]+window_length])

    return peak

# Significance for a trial and channel
def window_peak_trial(sess, chan, targ, trial, overt=True, bin_size=25):
    """
    Calculates the change in firing rate for a given channel, target, and trial.

    Inputs:
        chan (int): Desired channel
        targ (int:) Desired movement target (1 left, 2 right)
        trial (int): Desired trial (in a list of all trials that satisfy the desired target and condition, i.e. out of 40)
        overt (bool): Whether to display overt or covert trials
        bin_size (int): Size of bins to use for window (odd or adds 1)

    Returns:
        (float): Average baseline activity (total spikes / bin_size)
        (float): Average movement activity (total spikes / bin_size)
    """

    trace = sess.spikes.loc[(sess.spikes["condition"] == "overt") & (sess.spikes["targ"] == targ)]["data"].iloc[trial-1][:,chan-1]

    move_idx = sess.DM_movement_index[targ]

    base_trace = trace[125:150]
    move_trace = trace[move_idx-12:move_idx+12+1] # 25 sample window

    return np.nansum(base_trace)/0.5, np.nansum(move_trace)/0.5
    

# Significance across trials for a channel
def window_peak_chan(sess, chan, targ, overt=True, bin_size=30):
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

    bases = np.zeros(num_trials)
    moves = np.zeros(num_trials)
    for i in range(num_trials):
        [bases[i], moves[i]] = window_peak_trial(sess, chan, targ, i+1, overt=overt, bin_size=bin_size)

    p_val = ttest_1samp(moves-bases, 0, nan_policy="omit")[1]
    
    if np.ma.is_masked(p_val):
        p_val = 1
        
    return p_val, bases, moves


# Main plotting
def chan_sig_map(sess, threshold, medial=False, overt=True, annotate='', sig_method="window", plot=True):
    """
    Display heat maps for t-test significance comparison of mean baseline FR and mean reach FR

    Inputs:
        threshold (float): p-value for maximum limit on heat map
        medial (bool): Which array to display (False - lateral)        
        overt (bool): Whether to display overt or covert trials
        annotate (string): How to annotate the plots. 3 options:
                            'chan_num': channel number
                            'significance': tuning significance per channel
                            '': (default) no annotation            
        sig_method (string): Which method to use when calculating significance ("window" is best choice now)
        plot (bool): Whether to plot result
        
    Returns:
        Dataframe input for heatmap
    """
    if medial:
        motor_chan = (np.unique(sess.med_map)[~np.isnan(np.unique(sess.med_map))]).astype(int)
    else:
        motor_chan = np.unique(sess.lat_map)[~np.isnan(np.unique(sess.lat_map))].astype(int)

    targets = list(sess.targ_dict.values())
    motor_tuning = pd.DataFrame(index=motor_chan,columns=targets)


    if sig_method == 'peak':
        for chan in motor_chan:
            # Make sure channel is not below FR threshold
            for targ in targets:
                if not sess.chan_mask[chan-1]:
                    motor_tuning[targ][chan] = np.nan
                else:
                    motor_tuning[targ][chan] = chan_peak(sess, chan, sess.targ_num_dict[targ], overt=overt)[0]
    elif sig_method == 'sample':
        for chan in motor_chan:
            for targ in targets:
                if not sess.chan_mask[chan-1]:
                    motor_tuning[targ][chan] = np.nan
                else:
                    motor_tuning[targ][chan] = t_test_intra(chan, sess.targ_num_dict[targ], overt=overt)[0]
    elif sig_method == 'window':
        for chan in motor_chan:
            # Make sure channel is not below FR threshold
            for targ in targets:
                if not sess.chan_mask[chan-1]:
                    motor_tuning.loc[chan, targ] = np.nan
                else:
                    motor_tuning.loc[chan, targ] = window_peak_chan(sess, chan, sess.targ_num_dict[targ], overt=overt)[0]
            

    if sig_method == "peak":
        method = 'Boolean Significance'
    else:
        method = "Tuning Significance"
        
    if plot:
        channel_map(sess, motor_tuning, method, threshold, overt=overt, annotate=annotate, medial=medial)

    return motor_tuning