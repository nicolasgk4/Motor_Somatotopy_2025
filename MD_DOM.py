import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from MD_Tuning import absolute_peak, chan_sig_map
from MD_Plotting import channel_map

# Calculate depth of modulation for a single trial
def depth_modulation(sess, chan, targ, trial, method="window", bin_size=30, overt=True):
    """
    Calculate depth of modulation for a single trial.
    
    Inputs:
        chan (int): Desired channel
        targ (int): Desired movement target (1-4, alphabetical order)
        trial (int): Desired trial (in a list of all trials that satisfy the desired target and condition)
        method (string): Method to use when calculating DOM. 2 options:
            peak - calculate (abs max peak reach FR) - (mean baseline FR)
            sample - calculate (mean reach FR in sample window) - (mean baseline FR in sample window)
        bin_size (int): Size of bins to use for moving average - for peak method
        overt (bool): Whether to display overt or covert trials
        
    Outputs:
        float: Depth of modulation for selected trial
    """

    if overt:
        overt_ = "overt"
    else:
        overt_ = "covert"

    spikes_test = sess.spikes.loc[(sess.spikes['targ'] == targ) & (sess.spikes['condition'] == overt_)].iloc[trial-1]


    reach_start = spikes_test["epoch_starts"][1]
    reach_end = spikes_test["epoch_starts"][2] 

    if method == "peak":
        # OLD
        # data = spikes_test["FR"][:, chan-1]
        # data_reach = data[reach_start:reach_end]
        
        # max_FR = max(data[reach_start:reach_end])
        # max_idx = np.where(data == max_FR)[0][0]

        # reach_samp = data[(max_idx-25):(max_idx+25)]
        # base_avg = np.mean(data[0:reach_start])

        # DOM = max_FR - base_avg
        
        # New method - absolute max value of moving average
        data = spikes_test["FR"][:, chan-1]
        data_reach = data[reach_start:reach_end]

        reach_diff = data_reach - spikes_test["mean_base"][chan-1]
        
        # num_bins = int(np.floor(150/bin_size)) # lose some end samples if not divisor of 150

        # peak = np.zeros(num_bins)

        # for i in range(num_bins): # [bin_size]-sample moving window, up to last possible [bin_size]-sample window
        #     peak[i] = np.mean(reach_diff[i*bin_size:(i+1)*bin_size]) # Calculate mean value in [bin_size]-sample window

        # window_length = 25 # Samples
        # window_starts = [15, 25, 35, 45, 55] # 300, 500, 700, 900, 1100 ms

        # peak = np.zeros(len(window_starts))

        # for i in range(len(window_starts)):
        #     peak[i] = np.mean(reach_diff[window_starts[i]:window_starts[i]+window_length])

        DOM = max(absolute_peak(sess, chan, targ, trial, overt=overt), key=abs)
        
    elif method == "sample":
        data = spikes_test["FR"][:, chan-1]
        
        base_start = 50
        reach_win = reach_start + 25
        win_len = 50
        
        base_mean = np.mean(data[base_start:base_start+win_len])
        reach_mean = np.mean(data[reach_win:reach_win+win_len])
        
        DOM = reach_mean - base_mean
        

    elif method == "window":

        trace = spikes_test["FR"][:,chan-1]

        move_idx = sess.DM_movement_index[targ]

        base_trace = trace[125:150]
        move_trace = trace[move_idx-12:move_idx+12+1] # 25 sample window

        # Greatest change from baseline - positive or negative
        DOM_idx = np.nanargmax(abs(move_trace - np.nanmean(base_trace)))

        # Pull out value at that point and calculate change from baseline
        DOM = move_trace[DOM_idx] - np.nanmean(base_trace)
    
    else:
        print("METHOD ERROR")

    return DOM    
    
    
    
#Return average depth of modulation for specific channel, target, and condition across trials
def trial_modulation(sess, chan, targ, method="window", overt=True, plot=False):
    """
    Calculates average depth of modulation for specified channel,
    target, and condition.

    Inputs:
        chan (int): Desired channel
        targ (int): Desired movement target (1-4, alphabetical order)
        method (string): Method to use when calculating DOM. 2 options:
            peak - calculate (peak reach FR) - (mean baseline FR)
            sample - calculate (mean reach FR in sample window) - (mean baseline FR in sample window) 
        overt (bool): Whether to display overt or covert trials
        plot (bool): Whether to display plot of average depth of modulation
        
    Outputs:
        float: Average depth of modulation for specified channel, target, and condition
        float: Standard deviation of depth of modulation for specified channel, target, and condition
    """

    if overt:
        overt_ = "overt"
    else:
        overt_ = "covert"

    num_trials = len(sess.spikes.loc[(sess.spikes['targ'] == targ) & (sess.spikes['condition'] == overt_)])

    depths = np.zeros(num_trials)
    for i in range(num_trials):
        depths[i] = depth_modulation(sess, chan, targ, i+1, method=method, overt=overt)

    if plot:
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        plt.rc('font',size=12)
        ax.scatter(range(num_trials),depths)
        ax.set_ylim([min(depths)-5,max(depths)+5])
        ax.set_xlabel('Trial')
        ax.set_ylabel(r'$\Delta$ FR')
        ax.axhline(np.mean(depths),color='r',linestyle=':',label='Mean')
        ax.axhline(0,color='k',linestyle=':')
        ax.legend()

        if overt:
            ax.set_title(f"Depth of Modulation\nChannel {chan} - Overt {sess.targ_dict[targ]}",fontsize=18)
        else:
            ax.set_title(f"Depth of Modulation\nChannel {chan} - Covert {sess.targ_dict[targ]}",fontsize=18)

        plt.rc('font',size=18)
        ax.set_xticks(np.arange(1,num_trials+1,2))
        ax.set_xticklabels(range(2,num_trials+2,2))

    return np.mean(depths),np.std(depths),depths        
    
    
    
#Channel map for average depth of modulation for desired array for each target during specific condition
def depth_map(sess, my_max, overt=True, medial=True, method="window", annotate='', final=False, save=False, plot=True):
    """
    Displays heat maps showing the average depth of modulation 
    for each channel on a specified array for a specified condition

    Uses general channel_map function

    Takes in:
        my_max (float): Maximum value for heatmap
        overt (bool): Whether to display overt or covert trials
        medial (bool): Which array to display (False - lateral)
        method (string): Which method to use when calculating depth of modulation
            sample - calculate 
        annotate (string): How to annotate the plots. 3 options:
                            'chan_num': channel number
                            'significance': tuning significance per channel
                            '': (default) no annotation
        final (bool): Whether plot is for professional display
        save (bool): Whether to save plot
        plot (bool): Whether to plot result
        
    Returns:
        pd.DataFrame: DataFrame input for heatmap

    """

    if medial:            
        motor_chan = (np.unique(sess.med_map)[~np.isnan(np.unique(sess.med_map))]).astype(int)
    else:            
        motor_chan = np.unique(sess.lat_map)[~np.isnan(np.unique(sess.lat_map))].astype(int)

    targets = list(sess.targ_dict.values())
    motor_tuning = pd.DataFrame(index=motor_chan,columns=targets)

    for chan in motor_chan:
        for targ in targets:
            motor_tuning.loc[chan, targ] = trial_modulation(sess, chan,sess.targ_num_dict[targ],method=method,overt=overt)[0]

    if plot:
        channel_map(sess, motor_tuning, 'Depth of Modulation', my_max, overt=overt, annotate=annotate, annotate_map=chan_sig_map(sess, 0.05/sess.num_targ, medial=medial, overt=overt, plot=False), final=final, save=save, medial=medial)
    
    return motor_tuning
    
    
    
#Average baseline FR for every channel (both arrays) for given target and condition across trials
def trial_mean_base(sess, targ, overt=True):
    """
    Create array holding averaged mean baseline FR for each channel for a given target and condition.
    
    Inputs:
        targ (int): Desired movement target (1-4, alphabetical order)
        overt (bool): Whether to display overt or covert trials
    Output:
        np.ndarray: Array holding mean baseline FR for each channel for a given target and condition
    """

    if overt:
        overt_ = "overt"
    else:
        overt_ = "covert"

    base_test = sess.spikes.loc[(sess.spikes['targ'] == targ) & (sess.spikes['condition'] == overt_)]["mean_base"]

    num_trials = len(base_test)

    base_means = np.zeros(256)
    for i in range(256):
        chan_base = [] #Holds baseline means per trial for a channel
        for ii in range(num_trials):
            chan_base.append(base_test.iloc[ii][i]) #Add next trial's baseline mean for that channel
        base_means[i] = np.mean(chan_base) #Update output array with averaged mean baseline FR for channel i
            
    return base_means #256x1 array of averaged mean baseline FR for each channel for given target and condition        
    
    
    
#Channel map for baseline FR across each target for specified condition and array
def base_map(sess, my_max, medial=False, overt=True, plot=True):
    """
    Displays heat maps showing the average baseline FR 
    for each channel on a specified array for a specified condition.

    Inputs:
        my_max (float): Maximum value for heatmap
        medial (bool): Which array to display (False - lateral)
        overt (bool): Whether to display overt or covert trials
        plot (bool): Whether to plot result
    Outputs:
        Nothing

    Uses general channel_map function

    """

    if medial:            
        motor_chan = (np.unique(sess.med_map)[~np.isnan(np.unique(sess.med_map))]).astype(int)
    else:            
        motor_chan = np.unique(sess.lat_map)[~np.isnan(np.unique(sess.lat_map))].astype(int)

    targets = list(sess.targ_dict.values())

    base_vals = dict()
    for targ in targets:
        base_vals[targ] = trial_mean_base(sess, sess.targ_num_dict[targ], overt)[motor_chan-1]

    motor_tuning = pd.DataFrame(base_vals, index=motor_chan,columns=targets)

    if plot:
        channel_map(sess, motor_tuning, 'Baseline FR', my_max, overt=overt, medial=medial)
    
    return motor_tuning