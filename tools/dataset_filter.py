# -*- coding: utf-8 -*-
# Author: Hoozz
# Date: 2024/06/27

import os
import sys
import wave
import shutil

DATASET_SRC_DIR = 'dataset/'
DATASET_DST_DIR = 'filter_output/'
DURATION_MIN_MS = 500       # >=500ms
DURATION_MAX_MS = 1500      # <=1500ms

def read_wave_duration(file_path):
    with wave.open(file_path, 'rb') as f:
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        duration_ms = nframes * 1000 // framerate
    return duration_ms

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
        duration_range = sys.argv[1].split(',')
        DURATION_MIN_MS = int(duration_range[0])
        DURATION_MAX_MS = int(duration_range[1])
    if len(sys.argv) > 2:
        DATASET_SRC_DIR = sys.argv[2]
    if len(sys.argv) > 3:
        DATASET_DST_DIR = sys.argv[3]

    # Find wav files
    files, files_total = find_wave_file(DATASET_SRC_DIR)

    drop_count = 0
    count = 0
    for dirpath in files:
        file_list = files[dirpath]
        for file in file_list:
            count += 1
            path = os.path.join(dirpath, file)
            print(f'[{count}/{files_total}]  {path}  ', end='', file=sys.stderr)
            duration_ms = read_wave_duration(path)
            if duration_ms >= DURATION_MIN_MS and duration_ms <= DURATION_MAX_MS:
                parent_dir = dirpath.replace(DATASET_SRC_DIR, DATASET_DST_DIR)
                if os.path.exists(parent_dir) != True:
                    os.makedirs(parent_dir)
                output_file = os.path.join(parent_dir, file)
                shutil.copy(path, output_file)
                print(f'--->  {output_file}', file=sys.stderr)
            else:
                drop_count += 1
                print(f'DROP ({duration_ms}ms)', file=sys.stderr)

    print(file=sys.stderr)
    print(f'File total: {files_total}', file=sys.stderr)
    print(f'Drop: {drop_count}', file=sys.stderr)
    print(f'Save: {files_total - drop_count}', file=sys.stderr)
