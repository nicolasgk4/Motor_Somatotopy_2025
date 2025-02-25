import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import matplotlib.colors as colors
import matplotlib.gridspec as gridspec

from plotly.subplots import make_subplots

from scipy.stats import median_abs_deviation

from sklearn.decomposition import FactorAnalysis


from MD_Helpers import create_reach_arr


def channel_map(sess, motor_tuning, label, my_max, medial, overt=True, annotate=False, annotate_map=None, final=False, save=False):
    """
    Function for displaying heatmaps for a specified condition across all targets.

    Inputs:
        motor_tuning (pd.DataFrame): Dataframe containing relevant information per channel (rows) and target (columns)
        label (string): One of 3 options:
                            'Baseline FR': Averaged mean baseline firing rates for each channel per target
                            'Depth of Modulation': Average change in firing rate between peak FR during reach and mean 
                                baseline FR for each channel per target
                            'Tuning Significance': p values for each channel and target of t-test performed between 
                                baseline and reach samples for each trial
        my_max (float): Maximum value for heatmap
        overt (bool): Whether to display overt or covert trials
        annotate (string): How to annotate the plots. 3 options:
                            'chan_num': channel number
                            'significance': tuning significance per channel
                            '': (default) no annotation
        annotate_map (pd.DataFrame): DataFrame of annotations to be displayed on heatmap (if annotate is 'significance')
        final (bool): Whether plot is for professional display
        save (bool): Whether to save plot

    Returns:
        Dataframe input used for heatmap
    """

    # Pitt numbering: Anterior pedestal, lateral array, 1-128. Posterior pedestal, medial array, 129-256
    # Chicago numbering: Anterior pedestal, medial array, 1-128. Posterior pedestal, lateral array, 129-256

    if medial:
        motor_chan = (np.unique(sess.med_map)[~np.isnan(np.unique(sess.med_map))]).astype(int)
        motor_map = sess.med_map.copy()
    else:
        motor_chan = np.unique(sess.lat_map)[~np.isnan(np.unique(sess.lat_map))].astype(int)
        motor_map = sess.lat_map.copy()

    if sess.subject == "CRS02b":
        if not medial:
            motor_map = np.rot90(motor_map,axes=(1,0))
            

    motor_map = np.reshape(motor_map,100)

    # Create list of 10x10 arrays for plotting - same number as number of targets
    motor_plotting = []
    for i in range(sess.num_targ):
        motor_plotting.append(motor_map.copy())

    # Reshape input dataframe into plotting matrix
    for chan in motor_chan:
        for i in range(sess.num_targ):
            motor_plotting[i][np.where(motor_map == chan)] = motor_tuning[sess.targ_dict[i+1]][chan]

    # Reshape into 10x10 arrays
    for i in range(sess.num_targ):
        motor_plotting[i] = np.reshape(motor_plotting[i],(10,10))
    
    
    if medial:
        mela = 'Medial'
    else:
        mela = 'Lateral'
    
    cmax = my_max
    
    dcenter = None
    
    if label == 'Baseline FR':
        d_label = 'Baseline FR (Hz)'
        d_cmap = 'viridis'
        cmin = 0
    elif label == 'Depth of Modulation':
        d_label = r'$\Delta$ Firing Rate (Hz)'
        d_cmap = 'icefire'
        cmin = np.nanmin(motor_plotting)
        dcenter = 0
        
        divnorm = colors.TwoSlopeNorm(vmin=cmin, vcenter=0, vmax=cmax)
        
    elif label == 'Tuning Significance':
        d_label = 'p value'
        d_cmap = 'viridis_r'
        cmin = np.nanmin(motor_plotting)
        
    elif label == "Boolean Significance":
        d_label = "Significant"
        
        for i in range(sess.num_targ):
            motor_plotting[i] = motor_plotting[i] < my_max

        cmin = None
        vmin = None
        vmax = None
        d_cmap='mako'
    else:
        d_label = label
        d_cmap = 'viridis'
        cmin = 0
        
    # Annotation

    if annotate == 'chan_num':
        annot_map = np.reshape(motor_map, (10,10))

        annots = []
        for i in range(sess.num_targ):
            annots.append(annot_map)
        
        my_fmt = 'g'
        
    elif annotate == 'significance':
        motor_tuning = annotate_map
        
        annots = []
        for i in range(sess.num_targ):
            annots.append(np.squeeze(np.array([[''] * 100])))

        ct = 0
        for chan in motor_chan:
            for i in range(sess.num_targ):
                annots[i][np.where(motor_map == chan)] = '*' if np.array(list(motor_tuning[sess.targ_dict[i+1]]))[ct] < 0.05/sess.num_targ else ''

            ct += 1
        
        for i in range(sess.num_targ):
            annots[i] = np.reshape(annots[i], (10,10))
        
        my_fmt = ''
        
    else:
        annots = []
        for i in range(sess.num_targ):
            annots.append(False)
        my_fmt = 'g'
        
    # -----------
        
    if final:
        fig = plt.figure()
        fig.set_figheight(8)
        fig.set_figwidth(30/4 * sess.num_targ)
        sns.set(font_scale=1.4)

        gs = gridspec.GridSpec(2, sess.num_targ*2, height_ratios=[1, 0.05])

        for i in range(sess.num_targ-1):
            hm = sns.heatmap(motor_plotting[i], xticklabels=False, yticklabels=False, ax=fig.add_subplot(gs[0,i*2:i*2+2]), linewidth=0.5, cmap=d_cmap, square=True, fmt='g', vmax=cmax, vmin=cmin, cbar=False, center=dcenter)
            hm.set_title(sess.targ_dict[i+1])

        hm = sns.heatmap(motor_plotting[sess.num_targ-1], xticklabels=False, yticklabels=False, ax=fig.add_subplot(gs[0,(sess.num_targ-1)*2:(sess.num_targ-1)*2+2]), linewidth=0.5, cmap=d_cmap, square=True, fmt='g', vmax=cmax, vmin=cmin, cbar_ax=fig.add_subplot(gs[1, sess.num_targ-1:sess.num_targ+1], xticklabels=[]), cbar_kws={'orientation':'horizontal','label':d_label}, center=dcenter)
        hm.set_title(sess.targ_dict[sess.num_targ])
        
        fig.suptitle(label + '\n' + mela, y=1, fontsize=26, fontweight="bold", verticalalignment="top", horizontalalignment="center")
        
    else:
        fig = plt.figure()
        fig.set_figheight(8)
        fig.set_figwidth(30/4 * sess.num_targ)
        sns.set(font_scale=1.4)

        gs = gridspec.GridSpec(2, sess.num_targ*2, height_ratios=[1, 0.05])

        for i in range(sess.num_targ-1):
            hm = sns.heatmap(motor_plotting[i], xticklabels=False, yticklabels=False, ax=fig.add_subplot(gs[0,i*2:i*2+2]), linewidth=0.5, cmap=d_cmap, square=True, annot=annots[i], fmt=my_fmt, vmax=cmax, vmin=cmin, cbar=False, center=dcenter)
            hm.set_title(sess.targ_dict[i+1])

        # Separate last plot for colorbar
        hm = sns.heatmap(motor_plotting[sess.num_targ-1], xticklabels=False, yticklabels=False, ax=fig.add_subplot(gs[0,(sess.num_targ-1)*2:(sess.num_targ-1)*2+2]), linewidth=0.5, cmap=d_cmap, square=True, annot=annots[sess.num_targ-1], fmt=my_fmt, vmax=cmax, vmin=cmin, cbar_ax=fig.add_subplot(gs[1, sess.num_targ-1:sess.num_targ+1], xticklabels=[]), cbar_kws={'orientation':'horizontal','label':d_label}, center=dcenter)
        hm.set_title(sess.targ_dict[sess.num_targ])

        fig.axes[-1].set_yticks([])

        
        fig.suptitle(label + '\n' + mela, y=1, fontsize=26, fontweight="bold", verticalalignment="top", horizontalalignment="center")
        
        
    if save:
        if overt:
            save_name = "Overt_" + label.replace(" ", "_") + "_" + mela + "_" + sess.subject + ".pdf"
        else:
            save_name = "Covert_" + label.replace(" ", "_") + "_" + mela + "_" + sess.subject + ".pdf"
            
        print("Figure saved as " + save_name)
        
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        plt.savefig(save_name,dpi=300,bbox_inches='tight')

def channel_raster(sess, trial_num, disp_avg=False, save_fig=False):
    """
    Plots raster plot of all channels for a specified trial

    Inputs:
        trial_num (int): desired trial
        disp_avg (bool): whether to display average FR trace beneath plot
        save_fig (bool): whether to save the outputted figure as a .png
    Prints:
        Raster plot
    """
    data = sess.spikes['data']
    epochs = sess.spikes['epoch_starts']

    counts = data[trial_num]
    counts = np.transpose(counts)
    raster = []
    for i in range(0,256):
        raster.append(np.where(counts[i,:] > 0)[0]*0.02)

    #Plot raster across channels
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])

    ax.eventplot(raster)

    ax.axvline(x=epochs[trial_num][1]*0.02, ymin=0, ymax=1, color='k')
    ax.text(epochs[trial_num][1]*0.02 + 0.1,256,'Movement')

    ax.axvline(x=epochs[trial_num][2]*0.02, ymin=0, ymax=1, color='k')
    ax.text(epochs[trial_num][2]*0.02 + 0.1,256,'Rest')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Channel')

    if save_fig:
        plt.savefig("OC_Channel_Raster.png",dpi=300,bbox_inches='tight')
    
def trial_raster(sess, chan, targ, overt=True, disp_avg=True):
    """
    Plots raster plot of all trials for a specific channel and target

    Inputs:
        chan (int): Desired channel
        targ (int): Desired movement target (1 left, 2 right, 0 either)
        overt (bool): Whether to display overt or covert trials
        disp_avg (bool): whether to display average FR trace beneath plot
    Prints:
        Raster plot
    """
    if overt:
        overt_ = "overt"
    else:
        overt_ = "covert"

    data = sess.spikes.loc[(sess.spikes['targ'] == targ) & (sess.spikes['condition'] == overt_)]['data']
    num_trials = len(data)


    epochs = [151*0.02,302*0.02]
    raster = []
    for i in range(0,num_trials):
        raster.append(np.where(data.iloc[i][:,chan-1] > 0)[0]*0.02)

    #Plot raster across channels

    fig,axes = plt.subplots(2,1, gridspec_kw={'height_ratios': [5, 1]})

    axes[0].eventplot(raster, colors="b")

    axes[0].axvline(x=epochs[0], ymin=0, ymax=1, color='k')
    axes[0].text(epochs[0] + 0.1,-1,'Movement')

    axes[0].axvline(x=epochs[1], ymin=0, ymax=1, color='k')
    axes[0].text(epochs[1] + 0.1,-1,'Rest')

    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Trial')

    axes[0].set_ylim([0, num_trials])

    if overt:
        axes[0].set_title('Overt %s'%sess.targ_dict[targ] + ' - Ch %d'%chan)
    else:
        axes[0].set_title('Covert %s'%sess.targ_dict[targ] + ' - Ch %d'%chan)

    if disp_avg:
        avg_FR = np.mean(np.array(sess.FR_chan(chan, targ, overt, plot=False)), axis=0)
        
        axes[1].plot(avg_FR, "b")
        axes[1].set_ylabel("Avg FR (Hz)")
        axes[1].axvline(x=epochs[0]*50, ymin=0, ymax=1, color='k')

        axes[1].axvline(x=epochs[1]*50, ymin=0, ymax=1, color='k')
        
        axes[1].set_xticks([0, 100, 200, 300, 400])
        axes[1].set_xticklabels([0, 2, 4, 6, 8])

        axes[0].set_xticks([])
        axes[0].set_xlabel('')
        axes[1].set_xlabel("Time (s)")
        
        axes[1].locator_params(axis='y', nbins=3)

def FR_chan(sess, chan, targ, overt=True, norm=False, plot=True, show_samples=False, save=False):
    """
    Print firing rate plots for each trial of a specified condition/target combination for a specified channel

    Inputs:
        chan (int): Desired channel
        targ (int): Movement target (1 - Left, 2 - Right, 0 - either)
        overt (bool): Condition; whether to plot overt or covert trials
        norm (bool): Whether to plot normalized (z-scored) firing rates
        plot (bool): Whether to plot data
        show_samples (bool): Whether to display sample ranges used in original significance testing
        save (bool): Whether to save plot

    Returns:
        (np.ndarray): Data used in plot
    """

    if norm:
        norm_ = "norm_FR"
    else:
        norm_ = "FR"

    if overt:
        overt_ = "overt"
    else:
        overt_ = "covert"

    data1 = sess.spikes.loc[(sess.spikes['targ'] == targ) & (sess.spikes['condition'] == overt_)][norm_]
    num_trials = len(data1)

    epochs = [151*0.02,302*0.02]
    tvect = np.arange(0,(sess.spikes[norm_][1].shape[0]-1)*0.02+0.02,0.02) # Arbitrary trial

    plot_data = []
    for i in range(len(data1)):
        plot_data.append(data1.iloc[i][:,chan-1])

    if plot:
        fig,axes = plt.subplots(nrows=num_trials,ncols=1,figsize=(5,num_trials/20 * 16))

        base_start = 50 * 0.02 #1 second
        reach_start = epochs[0] + 25*0.025 #0.5 second

        for i in range(num_trials):
            axes[i].plot(tvect,plot_data[i])
            axes[i].locator_params(axis="y", nbins=3)
            #axes[i].set_ylim((0,1.5))

            axes[i].set_xticks([])
                

            if norm:
                min_lim = np.nanmin([np.nanmin(x) for x in plot_data])
            else:
                min_lim = 0

            max_lim = np.nanmax([np.nanmax(x) for x in plot_data])

            axes[i].set_ylim((min_lim,max_lim))

            axes[i].axvline(x=epochs[0], ymin=0, ymax=50, color='k')
            axes[i].axvline(x=epochs[1], ymin=0, ymax=50, color='k')

            axes[i].set_ylabel('T%d'%(i+1))

            if show_samples:
                axes[i].axvline(base_start,color='r')
                axes[i].axvline(base_start+50*0.02,color='r')
                axes[i].axvline(reach_start,color='r')
                axes[i].axvline(reach_start+50*0.02,color='r')

            # For new window DOM and significance testing
            axes[i].fill_between([125/50 + 0.02, 150/50 + 0.02], min_lim, np.nanmax([np.nanmax(x) for x in plot_data]), color='k', alpha=0.2)
            axes[i].fill_between([(sess.DM_movement_index[targ]-12)/50+0.02, (sess.DM_movement_index[targ]+12+1)/50+0.02], min_lim, max_lim, color='r', alpha=0.2)

        axes[i].set_xlabel("Time (s)")

        axes[0].text(epochs[0] + 0.1,max([max(x) for x in plot_data])*0.7,'Movement')
        axes[0].text(epochs[1] + 0.1,max([max(x) for x in plot_data])*0.7,'Rest')

        if overt:
            axes[0].set_title(f"Firing Rates - Overt {sess.targ_dict[targ]} - Channel {chan}")
        else:
            axes[0].set_title(f"Firing Rates - Covert {sess.targ_dict[targ]} - Channel {chan}")
            
        if save:
            if overt:
                save_name = "Overt_" + sess.targ_dict[targ] + "_Ch" + chan + "_" + sess.subject + ".pdf"
            else:
                save_name = "Covert_" + sess.targ_dict[targ] + "_Ch" + chan + "_" + sess.subject + ".pdf"
            print("Figure saved as " + save_name)
            plt.savefig(save_name, dpi=300, bbox_inches='tight')            

    return plot_data

def single_FR(sess, chan, targ, trial, overt=True, save=False, show_samples=False):
    """
    Plot firing rate plot with sample windows displayed. 
    Takes in:
        Channel #
        Target movement
        Overt/Covert (1/0)
        Trial # (when counting all trials that meet the above criteria)
        Whether to save figure or not (boolean)
        show_samples (bool): Whether to display sample ranges used in original significance testing
    Returns:
        Array of firing rate over time for the indicated trial

    E.g., for the 6th trial of overt wrist movement on channel 42, call
        single_FR(42,4,1,6)    
    """

    if overt:
        overt_ = "overt"
    else:
        overt_ = "covert"

    fig = plt.figure(figsize=(19,6))
    ax = fig.add_axes([0,0,1,1])

    tvect = np.arange(0,(sess.spikes['FR'][1].shape[0]-1)*0.02+0.02,0.02)

    data = sess.spikes.loc[(sess.spikes['targ'] == targ) & (sess.spikes['condition'] == overt_)]['FR'].iloc[trial-1][:,chan-1]

    ax.plot(tvect,data)
    ax.axvline(sess.spikes['epoch_starts'][1][1]*0.02,color='k')
    ax.axvline(sess.spikes['epoch_starts'][1][2]*0.02,color='k')

    #Proposed samples
    base_start = 50
    reach_start = 25

    if show_samples:
        ax.axvline(base_start*0.02,color='r')
        ax.axvline((base_start+50)*0.02,color='r')
    
        ax.axvline((sess.spikes['epoch_starts'][1][1]+reach_start)*0.02,color='r')
        ax.axvline((sess.spikes['epoch_starts'][1][1]+reach_start+50)*0.02,color='r')
        
    ax.tick_params(axis="x", which="major", labelsize = 18)

    plt.tick_params(axis='y',which='major',labelsize=18)

    ax.text((sess.spikes['epoch_starts'][1][1] + 2.5)*0.02,max(data),'Movement',fontsize=18)
    ax.text((sess.spikes['epoch_starts'][1][2] + 2.5)*0.02,max(data),'Rest',fontsize=18)

    ax.set_xlabel('Time (s)',fontsize=18)
    ax.set_ylabel('FR (Hz)',fontsize=18)

    if overt:
        ax.set_title(f"Firing Rate Channel {chan} - Overt {sess.targ_dict[targ]}, Trial {trial}",fontsize=18)
    else:
        ax.set_title(f"Firing Rate Channel {chan} - Covert {sess.targ_dict[targ]}, Trial {trial}",fontsize=18)

    if save:
        if overt:
            save_name = "Overt_" + sess.targ_dict[targ] + "_Ch" + str(chan) + "_Trial" + str(trial) + "_" + sess.subject + ".pdf"
        else:
            save_name = "Covert_" + sess.targ_dict[targ] + "_Ch" + str(chan) + "_Trial" + str(trial) + "_" + sess.subject + ".pdf"
        print("Figure saved as " + save_name)
        plt.savefig(save_name, dpi=300, bbox_inches='tight')

    return data

def change_FR(sess, chan, targ, overt=True, print_plots=False):
    """
    Calculates the percent changes in firing rates between baseline and reach phases from a different channel for a given target movement and condition. Prints these results (optional).
    
    Inputs:
        chan (int): Desired channel
        targ (int): Desired movement target (1-4, alphabetical order)
        overt (bool): Whether to display overt or covert trials
        print_plots (bool): Whether to print summary plots
        
    Returns:
        np.ndarray: Mean baseline FR for each trial
        np.ndarray: Mean reach FR for each trial
        np.ndarray: Percent change in FR for each trial

    E.g., for data from overt (1) wrist (4) movements on channel 42, call
        data = change_FR(42,4,1)  
    """
    base_start = 50
    reach_start = 25
    win_len = 50

    base = []
    reach = []

    if overt:
        overt_ = "overt"
    else:
        overt_ = "covert"

    spikes_test = sess.spikes.loc[(sess.spikes['targ'] == targ) & (sess.spikes['condition'] == overt_)]
    num_trials = len(spikes_test)

    for i in range(num_trials):
        base.append(spikes_test['FR'].iloc[i][base_start:base_start+win_len,chan-1])
        reach.append(spikes_test['FR'].iloc[i][spikes_test['epoch_starts'].iloc[i][1]+reach_start:spikes_test['epoch_starts'].iloc[i][1]+reach_start+win_len,chan-1])

    base_mean = np.zeros(num_trials)
    reach_mean = np.zeros(num_trials)

    for i in range(num_trials):
        base_mean[i] = np.mean(base[i])
        reach_mean[i] = np.mean(reach[i])


    diff_FR = np.zeros(num_trials)

    for i in range(num_trials):
        if base_mean[i] != 0:
            diff_FR[i] = ((reach_mean[i] - base_mean[i])/base_mean[i])*100 #%change in FR
        else:
            diff_FR[i] = np.nan #if mean baseline FR is 0

    if print_plots:
        if overt:
            print(f"Changes in Firing Rate Channel {chan} - Overt {sess.targ_dict[targ]}")
        else:
            print(f"Changes in Firing Rate Channel {chan} - Covert {sess.targ_dict[targ]}")

        ax_lim = max(np.append(base_mean,max(reach_mean))) * 1.1;

        # Histograms
        fig,axes = plt.subplots(1,3,figsize=(20,4))
        axes[0].hist(base_mean,rwidth=0.9)
        axes[0].set_title('Baseline FR')
        axes[0].set_xlim((0, ax_lim))
        axes[0].set_xlabel('Firing Rate (Hz)')
        axes[0].set_ylabel('Trials')

        axes[1].hist(reach_mean,rwidth=0.9)
        axes[1].set_title('Movement FR')
        axes[1].set_xlim((0, ax_lim))
        axes[1].set_xlabel('Firing Rate (Hz)')

        axes[2].hist(diff_FR,rwidth=0.9)
        axes[2].set_title('Percent Change in FR')
        axes[2].set_xlabel('Change in Firing Rate (%)')

        # Dotted line at 0
        axes[2].axvline(0,color='r',linestyle=':',label='No change')
        # Solid line at mean percent change
        axes[2].axvline(np.mean(diff_FR),color='k',label='Mean % Change')
        axes[2].legend()


        # Scatter plots

        # Mean reach vs baseline
        fig2,ax2 = plt.subplots(1,2,figsize=(20,6))

        ax2[0].scatter(base_mean,reach_mean)
        ax2[0].set_title('Mean Movement vs Mean Baseline FR')
        ax2[0].set_xlabel('Mean Baseline FR (Hz)')
        ax2[0].set_ylabel('Mean Movement FR (Hz)')

        ax2[0].set_xlim(0,ax_lim);
        ax2[0].set_ylim(0,ax_lim);

        ax2[0].plot(np.arange(0,ax_lim),np.arange(0,ax_lim),linestyle=':',color='r',label='No change')
        ax2[0].scatter(np.mean(base_mean),np.mean(reach_mean),color='r',s=150,label='Mean ')

        ax2[0].legend()

        #Percent change per trial
        ax2[1].plot(range(1,num_trials+1),diff_FR)
        ax2[1].set_title('Percent Change in FR per Trial')
        ax2[1].axhline(0,color='r',linestyle=':')

        ax2[1].set_xticks(range(1,num_trials+1))
        ax2[1].set_xlabel('Trial')
        ax2[1].set_ylabel('Percent Change in FR between Movement and Baseline')

        sess.FR_chan(chan,targ,overt)

    return base_mean, reach_mean, diff_FR

def plot_trajectories_3D(data, avg_data, colormap):
    """
    Function to plot 3D trajectories using plotly.

    Inputs:
        data (pd.DataFrame): Trajectory ata in a DataFrame organized with columns ["F1", "F2", "F3", "Trial", "Target"]
        avg_data (pd.DataFrame): Average trajectory data in a DataFrame organized with columns ["F1", "F2", "F3", "Target"]
        colormap (dict): Dictionary containing target names and color assignments

    Outputs:
        (plotly express subplots): Plotted data
    """
    
    # Plotting trajectories

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Individual Trials", "Average Trajectories"))

    fig.add_traces(
        px.line_3d(data, x="F1", y="F2", z="F3", labels={"F1":"F1", "F2":"F2", "F3":"F3"}, color="Target", color_discrete_map=colormap, hover_data=["Trial"]).data,
        rows=1, cols=1
    )

    fig.add_traces(
        px.line_3d(avg_data, x="F1", y="F2", z="F3", color="Target", color_discrete_map=colormap).data,
        rows=1, cols=2
    )

    # Add a labeled start point to each trajectory
    for targ in data["Target"].unique():
        fig.add_trace(go.Scatter3d(x=[avg_data[avg_data["Target"] == targ]["F1"].iloc[0]], y=[avg_data[avg_data["Target"] == targ]["F2"].iloc[0]], z=[avg_data[avg_data["Target"] == targ]["F3"].iloc[0]], mode="markers", marker=dict(size=5, color=colormap[targ]), name="Start"), row=1, col=2)

    fig.update_traces(row=1, col=1, showlegend=False)
    fig.update_layout(scene = dict(xaxis_title="F1", yaxis_title="F2", zaxis_title="F3"),
                    scene2 = dict(xaxis_title="F1", yaxis_title="F2", zaxis_title="F3"),
                    width=1000, height=600)

    # Hide start labels in legend
    for trace in fig['data']:
        if trace['name'] == "Start": trace['showlegend'] = False

    fig.show()

    return fig

def FA_plot(sess, epoch="reach", n_components=3, rotation="varimax", plot_data=True, use_array="both"):
    """
    Performs factor analysis on motor channel data during the 
    first 150 samples (3 seconds) of reach phase. Plots the resulting 
    trajectories and average trajectories. Also calculates and can plot the 
    same for individual arrays - with optional return of data.

    Inputs:
        epoch (string): The phase of the movement to analyze (for now, either "reach" or "all"). Uses specified class attribute
        n_components (int): The number of components to use for factor analysis
        rotation (string): The rotation method to use for factor analysis
        plot_data (bool): Whether to plot the data
        use_array (str): Which array to calculate data for. Can be "both" (default), "lateral", or "medial" to block out the other channels

    Outputs:
        (pd.DataFrame): Factor analysis data
        (pd.DataFrame): Average factor analysis data
        (plotly express subplots): Plotted data (if plot_data = True)
    """

    overt = True # HARD CODED TEMPORARILY OR MAYBE FOREVER WHO KNOWS

    if epoch == "reach":
        if use_array == "both":
            FR_data = sess.FR_reach_O
        elif use_array == "lateral":
            test_lat_mask = sess.chan_mask * [[True]*128 + [False]*128][0]
            FR_data = create_reach_arr(sess.spikes, test_lat_mask)
        elif use_array == "medial":
            test_med_mask = sess.chan_mask * [[False]*128 + [True]*128][0]
            FR_data = create_reach_arr(sess.spikes, test_med_mask)

        len_trial = sess.FA_reach_len # Length of reach phase sample taken
    elif epoch == "all":
        if use_array == "both":
            FR_data = sess.FR_all_O
        elif use_array == "lateral":
            test_lat_mask = sess.chan_mask * [[True]*128 + [False]*128][0]
            FR_data = create_reach_arr(sess.spikes, test_lat_mask, period="all")
        elif use_array == "medial":
            test_med_mask = sess.chan_mask * [[False]*128 + [True]*128][0]
            FR_data = create_reach_arr(sess.spikes, test_med_mask, period="all")

        len_trial = sess.spikes["data"].iloc[0].shape[0] # Length of an entire trial - should be the same across trials from upstream MATLAB processing

    # Create list of trials to use
    if overt:
        trial_list = list(sess.spikes[sess.spikes["condition"] == "overt"].index)


    len_reach = FR_data.shape[0] # Length of concatenated reach period array

    # Perform factor analysis
    FA_reach = FactorAnalysis(n_components=n_components, rotation=rotation)
    FA_reach.fit(FR_data) # Training on overt reach trials

    scores_reach = FA_reach.fit_transform(FR_data)

    # Append trial number and target to transformed data array
    scores_reach = scores_reach.T

    # Add trial number indicator
    FA_traj = np.row_stack((scores_reach, np.array([[i] * len_trial for i in trial_list]).flatten()))

    # Add target number indicator
    FA_traj = np.row_stack((FA_traj, np.array([[sess.spikes["targ"][i]] * len_trial for i in trial_list]).flatten()))

    # PLOTTING

    # Create dataframe for plotting
    FA = pd.DataFrame(data=FA_traj.T, columns=["F1", "F2", "F3", "Trial", "Target"])

    FA["Target"] = FA["Target"].map(sess.targ_dict)

    FA["Trial"] = FA["Trial"].astype(int)


    # Create average trajectories for each target

    # Pulling out specific target trajectories
    FA_list = []
    for targ in sess.targ_dict.values():
        FA_list.append(FA[FA["Target"] == targ][["F1", "F2", "F3"]])

    # Averaging trajectories
    FA_avg = []
    for i,targ_trials in enumerate(FA_list):
        FA_avg.append(np.hstack((np.array(targ_trials, dtype=object).reshape(len(targ_trials)//len_trial, len_trial, 3).mean(axis=0), np.asarray([sess.targ_dict[i+1]]*len_trial).reshape(len_trial,1))))

    FA_avg = np.vstack(FA_avg)

    # Recreating dataframe with target labels
    avg_FA = pd.DataFrame(data=FA_avg, columns=["F1", "F2", "F3", "Target"])

    if plot_data:
        fig = plot_trajectories_3D(FA, avg_FA, sess.my_color_map)
        return FA, avg_FA, FA_reach, fig

    return FA, avg_FA, FA_reach

def motion_artifact_plotting(sess, trial, plot_data=True, save=False):
    """
    Function to plot factor data and raw spike counts for a given trial to identify motion artifacts.

    Inputs:
        trial (int): The trial to plot
        plot_data (bool): Whether or not to plot the data
        save (bool): Whether or not to save the plot

    Returns:
        (bool): Whether a motion artifact is likely for this trial
    """

    if trial not in sess.spikes.index:
        print("Trial not in DataFrame")
        return

    # Perform factor analysis
    FA = FA_plot(sess, plot_data=False)[0]

    # Plotting
    fig,ax = plt.subplots(1,3, figsize=(20,5))

    trial_data = sess.spikes.loc[trial]
    data = trial_data["data"]
    FA_data = FA[FA["Trial"] == trial]

    trial_len = len(FA_data)

    ax[0].plot(FA_data["F1"], np.arange(1,trial_len+1), 'r', label="F1")
    ax[0].plot(FA_data["F2"], np.arange(1,trial_len+1), 'g', label="F2")
    ax[0].plot(FA_data["F3"], np.arange(1,trial_len+1), 'b', label="F3")
    ax[0].set_ylabel("Sample")
    ax[0].set_xlabel("Factor Magnitude")
    ax[0].set_title("Factor Traces")
    ax[0].invert_yaxis()
    ax[0].set_xlim(-10,10)
    ax[0].axvline(0, color="k")
    ax[0].legend()

    sns.heatmap(data, ax=ax[1])
    plt.xlabel("Channel")
    plt.ylabel("Sample")

    total_spikes = np.sum(data, axis=1)

    ax[2].plot(total_spikes, np.arange(1,len(data)+1))
    ax[2].invert_yaxis()
    ax[2].set_ylabel("Sample")
    ax[2].set_xlabel("Total Spikes")

    ax[2].axvline(median_abs_deviation(total_spikes)*5.5 + np.median(total_spikes), color="r")

    ax[2].axhline(trial_data["epoch_starts"][1], color="g")
    ax[2].axhline(trial_data["epoch_starts"][2], color="g")

    # Add an overall title
    plt.suptitle("Trial " + str(trial) + " | " + sess.targ_dict[trial_data["targ"]])

    # Make an additional plot to show what proportion of channels have spikes at each timepoint, separated into channels 1-128 and 129-256
    fig, ax = plt.subplots(1,3, figsize=(20,5))

    # Note: the way the proportion is calculated, it should not be treated as ground truth for CRS02b since those implants have 12 less channels each array (not 128 active)

    ax[0].plot(np.sum(data[:,0:128] > 0, axis=1)/128, np.arange(1,len(data)+1))
    ax[0].invert_yaxis()
    ax[0].set_ylabel("Sample")
    ax[0].set_xlabel("Proportion of Channels with Spikes")
    ax[0].set_title("Lateral")
    ax[0].set_xlim(0,1)

    ax[1].plot(np.sum(data[:,128:256] > 0, axis=1)/128, np.arange(1,len(data)+1))
    ax[1].invert_yaxis()
    ax[1].set_ylabel("Sample")
    ax[1].set_xlabel("Proportion of Channels with Spikes")
    ax[1].set_title("Medial")
    ax[1].set_xlim(0,1)

    ax[2].plot(np.sum(data > 0, axis=1)/256, np.arange(1,len(data)+1))
    ax[2].invert_yaxis()
    ax[2].set_ylabel("Sample")
    ax[2].set_xlabel("Proportion of Channels with Spikes")
    ax[2].set_title("Combined")
    ax[2].set_xlim(0,1)