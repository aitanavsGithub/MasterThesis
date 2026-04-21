import os                                        # file path handling
import numpy as np
import matplotlib.pyplot as plt  
from pprint import pprint 
import pandas as pd

import spikeinterface.full as si
import spikeinterface.extractors as se
from spikeinterface.extractors import read_mearec
import spikeinterface as sa
from spikeinterface.preprocessing import unsigned_to_signed, bandpass_filter, whiten, detect_and_remove_bad_channels
import spikeinterface.widgets as sw
from spikeinterface import load_sorting_analyzer
from helper import find_amplitude, extract_average_waveform


############################## load data/set params ##############################

datapath_results = "data\\excit_data\\20240627-iPCS-1806-div35-iNeurons_00.bxr"
datapath_raw = "data\\test_data\\testdata_raw.brw"
recording_biocam = se.read_biocam(datapath_raw, fill_gaps_strategy="zeros", )
samp_freq = recording_biocam.get_sampling_frequency()

if not os.path.exists(datapath_raw):
    print("Data file not found at path: " + datapath_raw)

#datapath = "data\\test_data\\testdata_raw.brw"
#results_biocam = se.read_biocam(datapath, fill_gaps_strategy="zeros", )
#results_mearec = se.read_mearec(file_path=datapath)
#analyzer_biocam = read_mearec(datapath)  

import h5py

f = h5py.File(datapath_results, "r")
print(list(f.keys()))

rec = se.read_binary(datapath_results, num_channels=4096, dtype="int16", sampling_frequency=samp_freq)

print(type(rec))
print(type(recording_biocam))