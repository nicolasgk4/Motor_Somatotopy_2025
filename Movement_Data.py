import numpy as np
import pandas as pd
from scipy.io import loadmat

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from MD_Helpers import create_reach_arr, avg_chan_FR


class Movement_Data:
    def __init__(self, subject, session=1, targets=["Elbow", "Grasp", "Shoulder", "Wrist"]):
        """
        Object to hold neural and metadata for overt/covert movement experiments.

        Inputs:
            subject (str): Subject ID
            session (int): Session number
            targets (list): List of target names as strings for the experiment as defined in Climber Data.TaskStateMasks.targ_idx.
                            This defines how many targets to plot as well as their names. Length of list should match number of targets in experiment,
                            regardless of their uniqueness. E.g., if there are 4 target numbers and the first 2 are "Elbow" and the last 2 are "Wrist", the list
                            should be ["Elbow", "Elbow", "Wrist", "Wrist"]. This target order is defined in the config file for the task.


        """

        self.subject = subject
        self.session = session

        # Define target dictionary ------------------------------------------------------------
        self.num_targ = len(np.unique(targets))
        self.targ_dict = dict(zip([i+1 for i,x in enumerate(targets)], targets))

        temp = []
        unique_dict = dict()

        for key,val in self.targ_dict.items():
            if val not in unique_dict.values():
                unique_dict[key] = val
            else:
                temp.append(key)

        # Reset numbering
        unique_dict = dict(zip([i+1 for i in range(len(unique_dict))], [x for x in unique_dict.values()]))

        # Swap keys and values
        unique_dict = dict(zip(unique_dict.values(), unique_dict.keys()))

        # For plotting
        self.my_color_map = {"Left":"#7BC148", "Right":"#F18721", "Elbow":"#7BC148", "Grasp":"#F18721", "Shoulder":"#24ADE4", "Wrist":"#F4C815", "Move":"#F4C815"}

        # -------------------------------------------------------------------------------------
        
        mat = loadmat(f".//Data//Somatotopy_Mapping//Movement_Data_{subject}_Session_{session}.mat")
        
        # Set subject       
        if subject == "P2":
            chan_map = [[np.nan, np.nan, 42, 58, 3, 13, 27, 97, np.nan, np.nan],
            [np.nan, 34, 44, 57, 4, 19, 29, 98, 107, np.nan],
            [33, 36, 51, 62, 7, 10, 31, 99, 108, 117],
            [35, 38, 53, 60, 5, 12, 18, 100, 109, 119],
            [37, 40, 50, 59, 6, 23, 22, 101, 110, 121],
            [39, 43, 46, 64, 9, 25, 24, 102, 111, 123],
            [41, 47, 56, 61, 17, 21, 26, 103, 113, 125],
            [45, 49, 55, 63, 15, 14, 28, 104, 112, 127],
            [np.nan, 48, 54, 2, 8, 16, 30, 105, 115, np.nan],
            [np.nan, np.nan, 52, 1, 11, 20, 32, 106, np.nan, np.nan]] 

            self.lat_map = np.array(chan_map)
            self.med_map = np.array(chan_map) + 128

        else:
            chan_map = [[np.nan, 38, 50, 59, 6, 23, 22, 101, 111, np.nan],
            [33, 40, 46, 64, 9, 25, 24, 102, 113, 128],
            [35, 43, 56, 61, 17, 21, 26, 103, 112, 114],
            [37, 47, 55, 63, 15, 14, 28, 104, 115, 116],
            [39, 49, 54, 2, 8, 16, 30, 105, 117, 118],
            [41, 48, 52, 1, 11, 20, 32, 106, 119, 120],
            [45, 42, 58, 3, 13, 27, 97, 107, 121, 122],
            [34, 44, 57, 4, 19, 29, 99, 108, 123, 124],
            [36, 51, 62, 7, 10, 31, 98, 109, 125, 126],
            [np.nan, 53, 60, 5, 12, 18, 100, 110, 127, np.nan]]; 

            if subject == "P3":
                self.lat_map = np.array(chan_map)
                self.med_map = np.array(chan_map) + 128

            elif subject == "P4":
                # Same map as P3
                self.lat_map = np.array(chan_map)
                self.med_map = np.array(chan_map) + 128
            
            elif subject == "C1":
                # Same map as P3, switched lateral and medial
                self.lat_map = np.array(chan_map) + 128
                self.med_map = np.array(chan_map)

            elif subject == "C2":
                # Same map as P3, switched lateral and medial
                self.lat_map = np.array(chan_map) + 128
                self.med_map = np.array(chan_map)


            
        # Create MATALB data DataFrame
        self.spikes = pd.DataFrame(index=np.arange(1, mat['D'][0].shape[0]+1, 1), columns=
                                   ["data", "epoch_starts", "targ", "FR", "mean_base", "norm_FR", "condition"])
        
        for i in range(1,mat['D'][0].shape[0]+1):
            self.spikes.loc[i, "data"] = mat['D'][0][i-1][0]
            self.spikes.loc[i, "epoch_starts"] = mat['D'][0][i-1][1][0]

            # Target definition
            targ = mat['D'][0][i-1][2][0][0]
            self.spikes.loc[i, "targ"] = unique_dict[self.targ_dict[targ]]

            self.spikes.loc[i, "FR"] = mat['D'][0][i-1]['FR']*50
            self.spikes.loc[i, "mean_base"] = np.nanmean(self.spikes.loc[i, "FR"][1:self.spikes.loc[i, "epoch_starts"][1]][:],axis=0)

            scaler = StandardScaler()
            scaler.fit(self.spikes.loc[i, "FR"])
            self.spikes.loc[i, "norm_FR"] = scaler.transform(self.spikes.loc[i, "FR"])

            self.spikes.loc[i, "condition"] = mat['D'][0][i-1]['oc'][0]


        # Redefine target dictionary to reflect assignment
        self.targ_dict = dict(zip(unique_dict.values(), unique_dict.keys()))


        self.num_trials = len(self.spikes) # Number of total trials
        self.num_trials_O = self.spikes[self.spikes["condition"] == "o"].shape[0] # Number of overt trials
        self.num_trials_C = self.spikes[self.spikes["condition"] == "c"].shape[0] # Number of covert trials

        self.targ_num_dict = unique_dict

        # HARD CODING ORDER FOR 4 MOVEMENT TASK
        if self.targ_dict == {1: "Elbow", 2: "Grasp", 3: "Shoulder", 4: "Wrist"}:
            self.targ_num_dict = {"Grasp": 1, "Wrist": 2, "Elbow": 3, "Shoulder": 4}

            for i in range(1, self.num_trials+1):
                self.spikes.loc[i, "targ"] = self.targ_num_dict[self.targ_dict[self.spikes["targ"][i]]]

            self.targ_dict = {1: "Grasp", 2: "Wrist", 3: "Elbow", 4: "Shoulder"}

        # Create mask to only use motor channels 
        chan_num = np.arange(1,257)
        chan_mask1 = chan_num < 65 
        chan_mask2 = chan_num > 96

        chan_maskL = chan_mask1 | chan_mask2

        chan_mask1 = chan_num < 65 + 128
        chan_mask2 = chan_num > 96 + 128

        chan_maskM = chan_mask1 | chan_mask2

        # Save channel mask - valid for all participants
        self.chan_mask = chan_maskL & chan_maskM

        # Save motor channel list - valid for all participants
        motor_chan = (np.unique(chan_map)[~np.isnan(np.unique(chan_map))]).astype(int)
        self.motor_channels = np.concatenate((motor_chan, motor_chan+128))

        # Correct for low-firing channels
        self.min_FR_thresh = 0.5 # Hz, across the entire experiment
        self.min_FR_mask = avg_chan_FR(self) >= self.min_FR_thresh
        self.chan_mask = self.chan_mask * self.min_FR_mask # Unmasked channel AND above minimum FR threshold

        self.FA_reach_len = 125 # Length of a single reach period - HARDCODED

        self.FR_reach_O = create_reach_arr(self.spikes, self.chan_mask, period_len=self.FA_reach_len)
        self.FR_all_O = create_reach_arr(self.spikes, self.chan_mask, period="all")

        # Calculate depth of modulation windows
        self.DM_movement_index = {}
        
        for targ in self.targ_dict.keys():
            pca = PCA(n_components=1)
            trace = pca.fit_transform(np.nanmean(np.array(self.spikes.loc[(self.spikes["targ"]==targ) & (self.spikes["condition"]=="overt")].FR.to_list()),axis=0))

            move_trace = trace[175:250]
            base_trace = trace[100:125]
    
            # Save the index of the maximum modulation
            self.DM_movement_index[targ] = np.nanargmax(move_trace) + 175


    # Class functions            

    def drop_trial(self, trial):
        """
        Function to remove a trial from the main DataFrame and all other analyses.

        Inputs:
            trial (int): Trial number to remove. Must be an existing index of the DataFrame (1-indexed)

        Returns:
            None
        """
        if trial in self.spikes.index:
            self.spikes = self.spikes.drop(trial)
            self.num_trials -= 1

            # Reapply minimum firing rate check (in case one trial saw tons of spikes in a channel, pushing it over the threshold on average)
            self.min_FR_mask = avg_chan_FR(self) >= self.min_FR_thresh
            self.chan_mask = self.chan_mask * self.min_FR_mask # Unmasked channel AND above minimum FR threshold

            # Recreate firing rate arrays 
            self.FR_reach_O = create_reach_arr(self.spikes, self.chan_mask)
            self.FR_all_O = create_reach_arr(self.spikes, self.chan_mask, period="all")
        else:
            print("Trial not in DataFrame")

    def drop_chan(self, chan):
        """
        Function to manually remove a channel or list of channels from the main DataFrame and all other analyses.

        Inputs:
            chan (list or int): Channel(s) to remove. 1-indexed.

        Returns:
            None
        """

        # If multiple channels passed in 
        if type(chan) == list:
            for i in range(len(self.chan_mask)):
                if i+1 in chan:
                    self.chan_mask[i] = False

        # If only 1 channel passed in
        else:
            self.chan_mask[chan-1] = False

        # Recreate firing rate arrays
        self.FR_reach_O = create_reach_arr(self.spikes, self.chan_mask)
        self.FR_all_O = create_reach_arr(self.spikes, self.chan_mask, period="all")