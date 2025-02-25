import numpy as np

from sklearn.naive_bayes import GaussianNB


def NB_classification(sess, start_offset=None, period_len=25, use_array="both"):
    """
    Function to perform Naive Bayes classification of movement targets based on threshold crossing data.

    Inputs:
        start_offset (int): Offset from movement phase start to begin classification
        period_len (int): Length of period to classify
        use_array (str): Which array to calculate data for. Can be "both" (default), "lateral", or "medial" to block out the other channels

    Returns:
        (np.ndarray): Confusion matrix of classification log-likelihoods for each trial
        (np.ndarray): Previous LL output, z-scored by rows (true movement target)
        (np.ndarray): Confusion matrix of classification results
    """

    half_period = int(period_len/2)

    # Obtain threshold crossing data
    spikes_mat = np.zeros(256)
    for i in sess.spikes.index:
        if sess.spikes["condition"][i] == "overt":
            start_offset = sess.DM_movement_index[sess.spikes["targ"][i]] - half_period - sess.spikes["epoch_starts"][i][1]
            spikes_mat = np.row_stack((spikes_mat, sess.spikes["data"][i][sess.spikes["epoch_starts"][i][1]:sess.spikes["epoch_starts"][i][2], :][start_offset:start_offset+period_len,:]))

    if use_array=="both":
        spikes_mat = spikes_mat[1:, sess.chan_mask]
    elif use_array=="lateral":
        spikes_mat = spikes_mat[1:, sess.chan_mask * [[True]*128 + [False]*128][0]] 
    elif use_array=="medial":
        spikes_mat = spikes_mat[1:, sess.chan_mask * [[False]*128 + [True]*128][0]] 


    # Metadata

    trial_list = list(sess.spikes[sess.spikes["condition"] == "overt"].index)
    len_trial = period_len

    # Create trial number array for averaging
    trial_labels = np.array([[i] * len_trial for i in trial_list]).flatten()

    # Create target label array
    target_labels = np.array([sess.spikes["targ"][i] for i in trial_list]).flatten()

    # Average over chosen window to obtain one <1 x chan> vector per trial
    trial_data = []
    for trial in np.unique(trial_labels):
        trial_data.append(np.nansum(spikes_mat[trial_labels == trial, :], axis=0) / (period_len / 50))


    LLs = []
    predicted_targs = []

    correct = 0

    for i in range(len(np.unique(trial_labels))):
        train_data = np.row_stack([trial for trial_num,trial in enumerate(trial_data) if trial_num != i])
        train_labels = np.array([target for target_num,target in enumerate(target_labels) if target_num != i])

        test_data = trial_data[i].reshape(1,-1)
        test_label = target_labels[i]

        gnb = GaussianNB()
        gnb.fit(train_data, train_labels)

        target_pred = gnb.predict(test_data)
        predicted_targs.append(target_pred)
        
        LLs.append(gnb.predict_joint_log_proba(test_data))

        if target_pred == test_label:
            correct += 1

    LLs = np.concatenate(LLs)

    print(f"{correct}/{len(np.unique(trial_labels))} trials correctly classified")
    print(f"{correct/len(np.unique(trial_labels))*100}% accuracy")


    # Create a confusion matrix with predicted_targs and target_labels
    confusion_matrix = np.zeros((4,4))

    for i in range(len(np.unique(trial_labels))):
        confusion_matrix[target_labels[i]-1, predicted_targs[i]-1] += 1

    conf_mat = []
    z_mat = []
    for targ in sess.targ_dict.keys():
        targ_LLs = np.median(LLs[target_labels == targ], axis=0)
        conf_mat.append(targ_LLs)
        z_mat.append((targ_LLs-np.mean(targ_LLs))/np.std(targ_LLs))

    conf_mat = np.array(conf_mat)
    z_mat = np.array(z_mat)

    return conf_mat, z_mat, confusion_matrix