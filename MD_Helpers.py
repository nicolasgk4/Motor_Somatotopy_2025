import numpy as np

def create_reach_arr(spikes, chan_mask, oc="overt", period="reach", start_offset=25, period_len=125):
    """
    Function to create FR_reach_O and FR_all_O attributes of MovementData class.

    Inputs:
        spikes (pd.DataFrame): Spikes dataframe containing MATLAB-processed data.
        chan_mask (np.ndarray): Mask to apply to firing rate matrix
        oc (string): Whether to create for overt ("overt") or covert ("covert") trials
        period (string): Whether to process data into concatenated reach period only or all periods
        start_offset (int): Only for period-specific arrays (i.e., not FR_all_O); what index to begin each trial's data with
        period_len (int): How many samples to pull from each trial's period

    Returns:
        (np.ndarray): FR_reach_O or FR_all_O, as specified
    """

    # Create reach period factor analysis array
    FR_arr = np.zeros(256)

    for i in spikes.index:
        if spikes["condition"][i] == oc:
            if period == "reach":
                FR_arr = np.row_stack((FR_arr, spikes["norm_FR"][i][spikes["epoch_starts"][i][1]:spikes["epoch_starts"][i][2], :][start_offset:start_offset+period_len,:]))
            elif period == "all":
                FR_arr = np.row_stack((FR_arr, spikes["norm_FR"][i]))

    FR_arr = FR_arr[1:, chan_mask] # Remove first row of zeros and apply mask to columns

    return FR_arr


def square_channels(sess, arr):
    """
    Convert 192xN array to a series of 10x10 arrays where values are placed in the appropriate
    position based on channel map per participant.

    Inputs:
        arr (array): 192xN array of data (for CRS02b, only 176 active channels but other channels still present in data)

    Returns:
        (list): List consisting of N 10x10 arrays of lateral channel data
        (list): List consisting of N 10x10 arrays of medial channel data
    """

    lateral_squares = []
    medial_squares = []

    for n in range(arr.shape[1]):
        medial_square = sess.med_map
        lateral_square = sess.lat_map
        for i in range(10):
            for j in range(10):
                if ~np.isnan(medial_square[i,j]):
                    medial_square[i,j] = arr[np.where(sess.motor_channels == medial_square[i,j])[0][0], n]
                if ~np.isnan(lateral_square[i,j]):
                    lateral_square[i,j] = arr[np.where(sess.motor_channels == lateral_square[i,j])[0][0], n]
        medial_squares.append(medial_square)
        lateral_squares.append(lateral_square)

    return lateral_squares, medial_squares


def avg_chan_FR(sess):
    """
    Function to calculate average firing rate across the entire experiment for all channels.
    Uses binned spike count data - sums over all trials and divides by session length in seconds.

    Inputs:
        None

    Returns:
        (list): List of average firing rates for all channels (length 256)
    """

    avg_FRs = np.zeros_like(sess.chan_mask, dtype=float)
    len_sess = sess.num_trials * sess.spikes.data.iloc[0].shape[0] / 50 # seconds, should be identical for all trials (MATLAB preprocessing)

    for chan in sess.motor_channels:
        total_spikes = np.nansum(np.array([sess.spikes.data.to_numpy()[i][:,chan-1] for i in range(len(sess.spikes))]))
        avg_FRs[chan-1] = (total_spikes / len_sess)

    return avg_FRs