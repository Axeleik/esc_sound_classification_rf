import numpy as np
from scipy import signal
import scipy.io.wavfile as sci_wav
import os
import librosa
import librosa.feature as feat
import matplotlib.pyplot as plt
from librosa.effects import split
from sklearn.ensemble import RandomForestClassifier





def read_and_split_file(path_to_sound_files,output_path):
    """
    Reads files and splits each file into 40 files with 5 seconds each.

    :param path_to_sound_files: Path where the initial sound files are
    :param output_path: Path where we save the split sound files
    """


    def split_and_save(path_file,file_name,output_path):
        """
        we read one file and then save than save the split versions in the output folder with changed names

        :param path_file:path of initial file
        :param file_name: name of whole file
        :param output_path: path where we save the output files
        """

        #read file
        rate,wave=sci_wav.read(path_file+file_name)

        #write all the split files with "_{number}" added before the ".wav" ending
        [sci_wav.write(output_path+file_name[:-4]+"_{}.wav".format(int(nr/80000)),16000,wave[nr:nr+80000])
        for nr in range(0,3200000,80000)]

    #load all files in the input folder
    sound_file_names=np.array(os.listdir(path_to_sound_files))

    #split and save for each of the files
    [split_and_save(path_to_sound_files,file_name,output_path)
     for file_name in sound_file_names]

