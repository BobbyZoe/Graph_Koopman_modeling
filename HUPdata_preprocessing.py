import os
from mne_bids import read_raw_bids, BIDSPath
import pandas as pd
import numpy as np
import glob
import pickle
from scipy.io import savemat

def data_preprocessing(subject, resample_sfreq, l_freq, h_freq, time_before_onset, time_after_onset, 
                      first_run_only=False):
    
    session = 'presurgery'
    task = 'ictal'
    
    bids_root = r'//10.20.37.22/dataset0/DATASETS/OpenNEURO/ds004100/'
    
    file_pattern = os.path.join(bids_root, f"sub-{subject}/ses-presurgery/ieeg/*.edf")
    all_files = sorted(glob.glob(file_pattern))
    
    # Keep ictal files and exclude interictal files
    ictal_files = [file for file in all_files if "task-ictal" in os.path.basename(file)]
    if not ictal_files:
        print(f"No ictal files found for subject {subject} in {bids_root}")
        return
    
    # Process only the first run if specified
    if first_run_only:
        ictal_files = ictal_files[:1]
        print(f"Only processing first run for subject {subject}")
    
    # Dynamically determine acquisition type
    first_file = ictal_files[0]
    if "acq-seeg" in os.path.basename(first_file):
        acquisition = "seeg"
    elif "acq-ecog" in os.path.basename(first_file):
        acquisition = "ecog"
    else:
        raise ValueError("Cannot determine acquisition type from file name!")
    
    suffix = 'ieeg'
    
    # Check sampling rate, skip subjects with sampling rate below 500 Hz
    first_bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        acquisition=acquisition,
        run=os.path.basename(first_file).split("_")[4].split("-")[1],
        suffix=suffix,
        root=bids_root,
        datatype='ieeg',
        extension=".edf"
    )
    
    # Read the first file to verify sampling rate
    temp_raw = read_raw_bids(first_bids_path)
    if temp_raw.info['sfreq'] < 500:
        print(f"Skipping subject {subject}: sampling rate lower than 500Hz")
        return
    
    print(f"Processing subject {subject} with sampling rate: {temp_raw.info['sfreq']}Hz")
    
    # Create output directory
    output_dir = rf"//10.20.37.22/dataset0/weiting/graph_koopman/preprocessed_HUPdata_500resample_{l_freq}_{h_freq}_100s"
    subject_dir = os.path.join(output_dir, subject)
    os.makedirs(subject_dir, exist_ok=True)
    
    # Process all runs
    soz_channel_indices = []
    soz_channel_names = []
    resect_channel_indices = []
    resect_channel_names = []
    good_channel_names = []
    
    for i, file in enumerate(ictal_files):
        run_id = os.path.basename(file).split("_")[4].split("-")[1]
        print(f"Processing run {run_id}")
        
        bids_path = BIDSPath(
            subject=subject,
            session=session,
            task=task,
            acquisition=acquisition,
            run=run_id,
            suffix=suffix,
            datatype='ieeg',
            root=bids_root,
            extension=".edf"
        )
        
        # Read events file
        event_tsv_path = str(bids_path)[:-8] + 'events.tsv'
        events_df = pd.read_csv(event_tsv_path, sep='\t')
        
        # Read channels file
        channel_tsv_path = str(bids_path)[:-8] + 'channels.tsv'
        channel_df = pd.read_csv(channel_tsv_path, sep='\t')
        
        # Read raw data
        raw = read_raw_bids(bids_path)
        raw.load_data()
        
        # Select channels with 'good' status
        with_good_channel_status = channel_df[channel_df['status'] == 'good']
        good_channels = with_good_channel_status['name'].tolist()
        
        # Extract channel metadata only from the first run
        if i == 0:
            good_channel_names = good_channels
            
            # Extract SOZ (Seizure Onset Zone) channel information
            soz_mask = with_good_channel_status['status_description'].str.contains('soz', na=False)
            soz_channel_names = with_good_channel_status[soz_mask]['name'].tolist()
            soz_channel_indices = [good_channels.index(name) for name in soz_channel_names if name in good_channels]
            
            # Extract resected channel information
            resect_mask = with_good_channel_status['status_description'].str.contains('resect', na=False)
            resect_channel_names = with_good_channel_status[resect_mask]['name'].tolist()
            resect_channel_indices = [good_channels.index(name) for name in resect_channel_names if name in good_channels]
        
        # Pick good channels for processing
        raw.pick(good_channels)
        
        # Notch filtering (60 Hz and harmonics, FIR method)
        raw.notch_filter(freqs=[60,180], method='fir')

        # Bandpass filtering (specified frequency range, FIR method)
        raw.filter(l_freq=l_freq, h_freq=h_freq, method='fir')
        
        # Resample to the target sampling rate
        raw.resample(sfreq=resample_sfreq, verbose=True)
        
        # Apply average reference
        raw.set_eeg_reference(ref_channels='average', verbose=True)
        
        # Get seizure onset time and crop data within the specified time window
        onset_time = events_df['onset'][0]
        start_time = max(0, onset_time - time_before_onset)
        end_time = min(raw.times[-1], onset_time + time_after_onset)
        cropped_raw = raw.copy().crop(tmin=start_time, tmax=end_time)
        
        # Get preprocessed data matrix
        data = cropped_raw.get_data()  # shape: (n_channels, n_samples)
        
        # Save data for the current run
        run_filename_pkl = os.path.join(subject_dir, f"run{int(run_id):02d}.pkl")
        run_filename_mat = os.path.join(subject_dir, f"run{int(run_id):02d}.mat")
        
        with open(run_filename_pkl, 'wb') as f:
            pickle.dump(data, f)
        savemat(run_filename_mat, {'data': data})
        
    
    # Save SOZ and resected channel information
    soz_info = {'soz_channel_indices': soz_channel_indices, 'soz_channel_names': soz_channel_names}
    resect_info = {'resect_channel_indices': resect_channel_indices, 'resect_channel_names': resect_channel_names}
    
    # Write output files
    soz_filename_pkl = os.path.join(subject_dir, "soz_info.pkl")
    soz_filename_mat = os.path.join(subject_dir, "soz_info.mat")
    resect_filename_pkl = os.path.join(subject_dir, "resect_info.pkl")
    resect_filename_mat = os.path.join(subject_dir, "resect_info.mat")
    
    with open(soz_filename_pkl, 'wb') as f:
        pickle.dump(soz_info, f)
    savemat(soz_filename_mat, soz_info)
    
    with open(resect_filename_pkl, 'wb') as f:
        pickle.dump(resect_info, f)
    savemat(resect_filename_mat, resect_info)
    

def main():
    
    # Automatically retrieve all subject directories
    bids_root = r'//10.20.37.22/dataset0/DATASETS/OpenNEURO/ds004100/'
    
    # Scan all folders starting with 'sub-'
    subject_dirs = glob.glob(os.path.join(bids_root, "sub-*"))
    subject_list = []
    
    for subject_dir in subject_dirs:
        if os.path.isdir(subject_dir):
            # Extract subject ID (remove 'sub-' prefix)
            subject_id = os.path.basename(subject_dir).replace("sub-", "")
            subject_list.append(subject_id)
    
    # Sort subject list alphanumerically
    subject_list = sorted(subject_list)
    
    # Preprocessing parameters
    resample_sfreq = 500.0
    l_freq = 80  
    h_freq = 200
    time_before_onset = 50
    time_after_onset = 50
    
    TEST_MODE = False  # True: process only the first run of one subject; False: process all runs of all subjects
    
    if TEST_MODE:
        # Test mode: process a single subject
        if subject_list:
            data_preprocessing(
                subject=subject_list[0],  # Process the first subject in the list
                resample_sfreq=resample_sfreq,
                l_freq=l_freq,
                h_freq=h_freq,
                time_before_onset=time_before_onset,
                time_after_onset=time_after_onset,
                first_run_only=True  # Process only the first run
            )
    else:
        # Production mode: process all subjects

        # Track processing status
        success_subjects = []
        failed_subjects = []
        skipped_subjects = []
        
        for idx, subject in enumerate(subject_list):
            print(f"\n{'='*60}")
            print(f"Processing progress: {idx+1}/{len(subject_list)} - Subject: {subject}")
            print(f"{'='*60}")
            
            try:
                data_preprocessing(
                    subject=subject,
                    resample_sfreq=resample_sfreq,
                    l_freq=l_freq,
                    h_freq=h_freq,
                    time_before_onset=time_before_onset,
                    time_after_onset=time_after_onset,
                    first_run_only=False  # Process all available runs
                )
                success_subjects.append(subject)
                
            except Exception as e:
                error_msg = str(e)
                if "sampling rate is 256Hz" in error_msg:
                    skipped_subjects.append(subject)
                    print(f" Skipping subject {subject}: sampling rate is 256 Hz")
                else:
                    failed_subjects.append((subject, error_msg))
                    print(f" Error occurred while processing subject {subject}: {error_msg}")
                continue
        

if __name__ == "__main__":
    main()