# -*- coding: utf-8 -*-
# Author: Hoozz
# Date: 2024/06/27

import os
import sys
import wave
import math

DATASET_DIR = 'dataset/'

def read_wave_duration(file_path):
    with wave.open(file_path, 'rb') as f:
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        duration_ms = nframes * 1000 // framerate
        # print(f'nchannels: {nchannels}')
        # print(f'sampwidth: {sampwidth}')
        # print(f'framerate: {framerate}')
        # print(f'nframes: {nframes}')
        # print(f'duration: {duration_ms} ms')
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

if __name__ == '__main__':
    if len(sys.argv) > 1:
        DATASET_DIR = sys.argv[1]

    # Find wav files
    files, files_total = find_wave_file(DATASET_DIR)

    duration_report = {}
    count = 0
    for dirpath in files:
        file_list = files[dirpath]
        duration_max = -math.inf, ''
        duration_min = math.inf, ''
        for file in file_list:
            count += 1
            path = os.path.join(dirpath, file)
            duration_ms = read_wave_duration(path)
            if duration_ms > duration_max[0]:
                duration_max = duration_ms, file
            if duration_ms < duration_min[0]:
                duration_min = duration_ms, file
            print(f'\rProgress: [{count} / {files_total}]', end='', file=sys.stderr)
            print(f'{duration_ms:8}ms {path}')
        duration_report[dirpath] = [duration_min, duration_max]

    # Report
    print(file=sys.stderr)
    print(f'File total: {files_total}', file=sys.stderr)
    for item in duration_report:
        report = duration_report[item]
        print(f'{item:　<15} min:{report[0][0]}ms max:{report[1][0]}ms (min:{report[0][1]} max:{report[1][1]})', file=sys.stderr)
