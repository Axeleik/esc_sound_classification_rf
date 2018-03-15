import numpy as np
import sys
from functions import read_and_split_file












if __name__ == '__main__':

    initial_files_path="/mnt/localdata1/amatskev/esc_project_rf/sound_files/"
    split_files_path="/mnt/localdata1/amatskev/esc_project_rf/split_files/"

    read_and_split_file(initial_files_path,split_files_path)