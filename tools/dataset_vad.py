# -*- coding: utf-8 -*-
# Author: Hoozz
# Date: 2024/06/28

import os
import sys
import wave
import shutil
import pyvad
import numpy as np

DATASET_SRC_DIR = 'dataset/'
DATASET_DST_DIR = 'vad_output/'

def read_wave_data(file_path):
    with wave.open(file_path, 'rb') as f:
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        if framerate != 16000:
            print(f'Failed: Incorrect audio sampling rate: {framerate}', file=sys.stderr)
        file_data = f.readframes(-1)
        nframes = int(len(file_data) / nchannels / sampwidth)
        wave_data = np.frombuffer(file_data, dtype=np.short)
        wave_data.shape = -1, nchannels
        wave_data = wave_data.T
        # Here we only take data from one channel
        wave_data = wave_data[0]
    return wave_data, framerate # , nframes, sampwidth

def save_to_wave(file_path, wave_data, framerate):
    # Open wave file
    with wave.open(file_path, 'wb') as f:
        # Parameters
        f.setparams((
            1,              # Number of channel
            2,              # Sample depth (bytes)
            framerate,      # Sample rate
            len(wave_data), # Number of frames
            'NONE',
            'not compressed'
        ))
        # Write data to file
        f.writeframesraw(wave_data)

def filter_out_silent_data(data, vact):
    if (len(data) != len(vact)):
        print(f'Failed: VAD data length does not match data length: {len(data)} != {len(vact)}', file=sys.stderr)
    new_data = data[vact == 1.0]
    return new_data

def find_wave_file(parent_dir):
    files = {}
    total = 0

    for dirpath, dirnames, filenames in os.walk(parent_dir):
        file_list = []
        for item in filenames:
            if 'wav' not in item:
                continue
            file_list.append(item)
            total += 1
        if len(file_list) != 0:
            files[dirpath] = file_list

    return files, total

if os.path.exists(DATASET_DST_DIR):
    # os.rmdir(DATASET_DST_DIR)
    shutil.rmtree(DATASET_DST_DIR)
    print(f'Re-create: {DATASET_DST_DIR}')
os.makedirs(DATASET_DST_DIR)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        DATASET_SRC_DIR = sys.argv[1]
    if len(sys.argv) > 2:
        DATASET_DST_DIR = sys.argv[2]

    # Find wav files
    files, files_total = find_wave_file(DATASET_SRC_DIR)

    count = 0
    for dirpath in files:
        file_list = files[dirpath]
        for file in file_list:
            count += 1
            path = os.path.join(dirpath, file)

            wave_data, framerate = read_wave_data(path)
            vact = pyvad.vad(wave_data, framerate, fs_vad=16000, hop_length=20, vad_mode=3)
            new_data = filter_out_silent_data(wave_data, vact)

            parent_dir = dirpath.replace(DATASET_SRC_DIR, DATASET_DST_DIR)
            if os.path.exists(parent_dir) != True:
                os.makedirs(parent_dir)
            output_file = os.path.join(parent_dir, file)
            save_to_wave(output_file, new_data, framerate)

            print(f'[{count}/{files_total}]  {path}  --->  {output_file}', file=sys.stderr)

    print(f'\nDone', file=sys.stderr)
