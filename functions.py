import numpy as np
from scipy import signal
import scipy.io.wavfile as sci_wav
import os
import librosa
import librosa.feature as feat
import matplotlib.pyplot as plt
from librosa.effects import split
from sklearn.ensemble import RandomForestClassifier
import pickle
from vigra import readHDF5,writeHDF5
import sklearn


def get_length_array():

    length_array = [
        ["chroma_stft", 12],
        ["chroma_cqt", 12],
        ["chroma_cens", 12],
        ["malspectrogram", 128],
        ["mfcc", 20],
        ["rmse", 1],
        ["spectral_centroid", 1],
        ["spectral_bandwidth", 1],
        ["chroma_contrast", 7],
        ["spectral_flatness", 1],
        ["spectral_rolloff", 1],
        ["poly_features", 2],
        ["tonnetz", 6],
        ["zero_crossing_rate", 1]]

    return length_array

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


def import_sounds(path_to_sound_files,scipy=True):
    """
    function to load the files

    :param path_to_sound_files: path for the sound files
    :return:
    sound_waves: loaded files
    sound_types: their names
    """

    #get sound file names
    sorted_file_names=np.sort(os.listdir(path_to_sound_files))

    #get actual types
    sound_types=np.array([name[4:-4] for name in sorted_file_names])

    # load sound_files
    # using scipy for now because librosa is messing up the files
    if scipy==True:
        print("Loading files")
        sound_waves = [sci_wav.read(path_to_sound_files+name) for name in sorted_file_names]

    else:
        sound_waves = np.array([librosa.load(path_to_sound_files+name, None) for name in sorted_file_names])


    return sound_waves,sound_types,sorted_file_names


def extract_features(soundwave,sampling_rate,sound_name="test",feature_list=[]):
    """
    extracts features with help of librosa
    :param soundwave: extracted soundwave from file
    :param sampling_rate: sampling rate
    :param feature_list: list of features to compute
    :param sound_name: type of sound, i.e. dog
    :return: np.array of all features for the soundwave
    """
    print("Computing features for ",sound_name)

    if len(feature_list)==0:
        feature_list=["chroma_stft","chroma_cqt","chroma_cens","melspectrogram",
                      "mfcc","rmse","spectral_centroid","spectral_bandwidth",
                      "spectral_contrast","spectral_flatness","spectral_rolloff",
                      "poly_features","tonnetz","zero_crossing_rate"]

    features=[]


    #feature_len
    #"chroma_stft":12
    if "chroma_stft" in feature_list:
        features.append(feat.chroma_stft(soundwave, sampling_rate))

    #"chroma_cqt":12
    if "chroma_cqt" in feature_list:
        features.append(feat.chroma_cqt(soundwave, sampling_rate))

    #"chroma_cens":12
    if "chroma_cens" in feature_list:
        features.append(feat.chroma_cens(soundwave, sampling_rate))

    #"malspectrogram":128
    if "melspectrogram" in feature_list:
        features.append(feat.melspectrogram(soundwave, sampling_rate))

    #"mfcc":20
    if "mfcc" in feature_list:
        features.append(feat.mfcc(soundwave, sampling_rate))

    #"rmse":1
    if "rmse" in feature_list:
        features.append(feat.rmse(soundwave))

    #"spectral_centroid":1
    if "spectral_centroid" in feature_list:
        features.append(feat.spectral_centroid(soundwave, sampling_rate))

    #"spectral_bandwidth":1
    if "spectral_bandwidth" in feature_list:
        features.append(feat.spectral_bandwidth(soundwave, sampling_rate))

    #"chroma_contrast":7
    if "spectral_contrast" in feature_list:
        features.append(feat.spectral_contrast(soundwave, sampling_rate))

    #"spectral_flatness":1
    if "spectral_flatness" in feature_list:
        features.append(feat.spectral_flatness(soundwave))

    #"spectral_rolloff":1
    if "spectral_rolloff" in feature_list:
        features.append(feat.spectral_rolloff(soundwave, sampling_rate))

    #"poly_features":2
    if "poly_features" in feature_list:
        features.append(feat.poly_features(soundwave, sampling_rate))

    #"tonnetz":6
    if "tonnetz" in feature_list:
        features.append(feat.tonnetz(soundwave, sampling_rate))

    #"zero_crossing_rate":1
    if "zero_crossing_rate" in feature_list:
        features.append(feat.zero_crossing_rate(soundwave))


    return np.concatenate(features)



def save_all_features_for_all_files(path_to_files,features_path=None,classes_path=None):
    """
    computes all paths for all soundfiles and saves them in a feature_vector to a file with the names if wanted

    :param path_to_files:
    :param path_to_save:
    :return: 1. array with all features for all soundwaves and 2. all the according classes
    """

    if features_path!=None and classes_path!=None:
        if os.path.exists(features_path) and os.path.exists(classes_path):
            print("Features and classes exist, loading")
            features_of_all=readHDF5(features_path, "features")
            soundtypes=np.load(classes_path)

            return features_of_all,soundtypes

    soundwaves,soundtypes,actual_file_names=import_sounds(path_to_files)

    features_of_all=np.array([extract_features(np.float64(soundwave),samplingrate,soundtypes[idx_soundwave]) for idx_soundwave,(samplingrate,soundwave)
                      in enumerate(soundwaves)])

    if features_path!=None and classes_path!=None:
        print("Saving features")

        writeHDF5(features_of_all,features_path,"features",compression="gzip")
        np.save(classes_path,soundtypes)

    return features_of_all,soundtypes


def load_random_forest_classifier(features, labels, save_path=None):
    """
    If rf already exists, load rf
    If not, ,train the random forest classifier with features and labels
    :param features: features
    :param labels: labels
    :return: trained random forest classifier
    """



    #load rf if it was already trained
    if save_path!=None:
        if os.path.exists(save_path):
            print("Rf exists, loading")
            rf=pickle.load(open(save_path, 'rb'))
            return rf

    print("Training rf")
    # Initialize and fit rf
    rf = RandomForestClassifier(500, n_jobs=32)
    rf.fit(features, labels)

    # save rf
    if save_path!=None:
        print("Saving rf")
        pickle.dump(rf, open(save_path, 'wb'))

    return rf

def train_and_predict_with_rf(features_train,classes_train,features_test,save_path=None,test=True):
    """

    :param features_train: features for training
    :param features_test: features for prediction
    :param classes_train: according training labels
    :return: predictions
    """

    classes=[classes_train[idx][:-2] for idx in range(0,len(classes_train),40)]
    classes_train=np.concatenate([[cl]*40 for cl in classes])


    # convert features (np.concatenate(features) does not work on any axis in desirable way)
    features_train=np.array([np.concatenate(feature) for feature in features_train])
    features_test=np.array([np.concatenate(feature) for feature in features_test])


    #Train rf and predict
    rf = load_random_forest_classifier(features_train,classes_train,save_path)

    if test:
        rf_predictions=rf.predict(features_test)
        return rf_predictions

    return rf

def test_feature_importances(rf):

    print("Analysing feature importances: ")

    length_array=get_length_array()

    f_imp=rf.feature_importances_

    type_arr=[]
    impact_arr=[]

    abs_len=0
    for type,length in length_array:
        type_arr.append(type)
        impact_arr.append(np.sum(f_imp[abs_len:abs_len+(length*157)]))
        abs_len+=length*157

    impact_arr=np.array(impact_arr)
    type_arr=np.array(type_arr)

    sort_mask=np.argsort(impact_arr)[::-1]

    type_arr=type_arr[sort_mask]
    impact_arr=impact_arr[sort_mask]

    for idx, val in enumerate(impact_arr):
        print(type_arr[idx],": ",impact_arr[idx])


def k_fold_cross_validation(features,classes,k_fold=10):

    print("Doing {}-fold cross-validation".format(k_fold))

    classes=[classes[idx][:-2] for idx in range(0,len(classes),40)]
    classes=np.concatenate([[cl]*40 for cl in classes])
    features=np.array([np.concatenate(feature) for feature in features])

    rf = RandomForestClassifier(500, n_jobs=32)


    return sklearn.model_selection.cross_val_score(rf, features, classes, cv=k_fold)


def k_fold_feature_exclusion(features,classes,excluded_features):

    length_array=get_length_array()

    print("Excluding following features:")
    print(excluded_features)

    abs_len=0
    for feature_type,feature_length in length_array:

        if feature_type in excluded_features:
            features=np.delete(features,np.s_[abs_len:abs_len+feature_length],axis=1)
        else:
            abs_len+=feature_length

    return np.mean(k_fold_cross_validation(features,classes,10))

def eval_downwards_upwards(features,classes,save_path):


    excluded_features_downwards = np.array(["malspectrogram", "mfcc", "chroma_stft", "chroma_contrast", "chroma_cens",
                                            "chroma_cqt", "tonnetz", "poly_features", "spectral_bandwidth", "rmse",
                                            "spectral_centroid", "spectral_rolloff", "spectral_flatness",
                                            "zero_crossing_rate"])

    excluded_features_upwards = excluded_features_downwards[::-1]





    print("STARTING DOWNWARDS")
    results_downwards = []
    for idx in range(1, len(excluded_features_downwards)):
        excluded_features = excluded_features_downwards[idx:]

        print("-------------------------------")
        print("Downwards idx={}".format(idx))
        results_downwards.append(k_fold_feature_exclusion(features, classes, excluded_features))
        print("Result: ",results_downwards[idx-1])
        print("-------------------------------")

    print("-------------------------------")
    print("-------------------------------")

    print("RESULTS DOWNWARDS: ",results_downwards)
    writeHDF5(results_downwards,save_path,"results_downwards")
    print("-------------------------------")
    print("-------------------------------")



    print("STARTING UPWARDS")

    results_upwards = []
    for idx in range(1, len(excluded_features_upwards)):
        excluded_features = excluded_features_upwards[idx:]

        print("-------------------------------")
        print("Downwards idx={}".format(idx))
        print("Features to exclude:")
        print(excluded_features)

        results_upwards.append(k_fold_feature_exclusion(features, classes, excluded_features))
        print("Result: ",results_upwards[idx-1])
        print("-------------------------------")

    print("-------------------------------")
    print("-------------------------------")
    print("RESULTS UPWARDS: ",results_upwards)
    writeHDF5(results_upwards, save_path, "results_upwards")
    print("-------------------------------")
    print("-------------------------------")

