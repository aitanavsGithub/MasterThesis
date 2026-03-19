
import os                                        # file path handling
import numpy as np
import matplotlib.pyplot as plt  
from pprint import pprint 

import spikeinterface.full as si
import spikeinterface.extractors as se
import spikeinterface as sa
from spikeinterface.preprocessing import unsigned_to_signed, bandpass_filter, whiten, detect_and_remove_bad_channels
import spikeinterface.widgets as sw
from spikeinterface import load_sorting_analyzer
from helper import generate_patch

##############################    set parameters / load data    #############################
sorter_name = 'kilosort4'
num_patches = 16
print("Starting analysis...")

# load recording
recording_biocam = se.read_biocam("data\\data_raw.brw", fill_gaps_strategy="zeros", )  
print("Recording loaded!")

# preprocessing dictionary
preprocessor_dict = {'unsigned_to_signed': {'bit_depth': 12},
                     'bandpass_filter': {'freq_min': 100}, 
                     'detect_and_remove_bad_channels': {}, 
                     'common_reference': {'operator': 'average'}}  # 'remove_artifacts': {}  


##############################    preprocess data and run sorting on patches    #############################

# preprocess data
preprocessed_recording = sa.preprocessing.apply_preprocessing_pipeline(recording_biocam, preprocessor_dict)
print("Preprocessing done!")

# loop over patches and sort them
for i in range(num_patches):
     
    # this generates the correct patch given the geometrz of the probe, see scrap5.ipynb for details
    patch = generate_patch(1 + i*num_patches + i//4*15*64)        
    recording_biocam_select = preprocessed_recording.select_channels(patch)

    # run sorting

    sorting_biocam = si.run_sorter(
        sorter_name,
        recording_biocam_select,
        folder=f"sorting_output\\kilosort4_output_patch_{i}",
        verbose=True
    )

    # create analyzer and calculate extensions
    analyzer_biocam = si.create_sorting_analyzer(sorting=sorting_biocam, recording=recording_biocam_select, format="memory")
    analyzer_biocam.compute(["random_spikes", "waveforms", "templates", "noise_levels","spike_amplitudes", "unit_locations", "spike_locations", "template_metrics"])

    # save analyzer
    analyzer_biocam.save_as(folder=f'C:\\Users\\user-adm\\Aitana\\MasterThesis\\analyzer_output\\analyzer_patch_{i}', format='binary_folder')

    print(f"Patch {i} done!")

