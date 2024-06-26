# -*- coding: utf-8 -*-
# Author: Hoozz
# Date: 2024/06/24

import wave
import socket as skt
import os
import datetime
import serial
import sys
import threading
import time

OUTPUT_DIR = 'output/'
CHANNEL = 1
DEPTH = 2
RATE = 16000

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_duration_ms(size):
    return int(size * 1000 / RATE / CHANNEL / DEPTH)

def save_to_wave(file_path, wave_data):
    # Open wave file
    with wave.open(file_path, 'wb') as f:
        nframes = int(len(wave_data) / CHANNEL / DEPTH)
        # Parameters
        f.setparams((
            CHANNEL,        # Number of channel
            DEPTH,          # Sample depth (bytes)
            RATE,           # Sample rate
            nframes,        # Number of frames
            'NONE',
            'not compressed'
        ))
        # Write data to file
        f.writeframesraw(wave_data)

def from_tcp():
    server = skt.socket(skt.AF_INET, skt.SOCK_STREAM)
    server.setsockopt(skt.SOL_SOCKET, skt.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', 17171))
    server.listen(1)

    print('Waiting for client connection...')

    while True:
        # Wait accept
        clientsocket, address = server.accept()
        clientsocket.settimeout(5.0)
        client_info = str(f'[{address[0]}:{address[1]}]: ')
        print(client_info, end='')
        buff = b''
        try:
            while(True):
                data = clientsocket.recv(1024)
                if data:
                    buff += data
                    print(f'\r{client_info}{len(buff):6} bytes', end='')
                else:
                    print(', ', end='')
                    break
            duration = get_duration_ms(len(buff))
            now_datetime = str(f'{datetime.datetime.now()}').replace(' ', '_').replace(':', '-')
            file_path = os.path.join(OUTPUT_DIR, f'{now_datetime}_{duration}ms.wav')
            save_to_wave(file_path, buff)
            print(f'Duration: {duration:4}ms, Output: {file_path}')
        except Exception as e:
            print(f'\r{client_info} Exception({e})')
        # Close socket
        clientsocket.close()
    # Close TCP
    server.close()

def from_serial(name):
    state = False
    # Open serial
    port = serial.Serial(name, 115200, timeout = 0.1)
    # port.timeout = 1.0

    print('Waiting for serial port data...')

    while True:
        buff = b''
        serial_info = str(f'{name}: ')
        while(True):
            data = port.read(1024)
            if data:
                buff += data
                print(f'\r{serial_info}{len(buff):6} bytes', end='')
            else:
                break
        if len(buff) > 0:
            print(', ', end='')
            duration = get_duration_ms(len(buff))
            now_datetime = str(f'{datetime.datetime.now()}').replace(' ', '_').replace(':', '-')
            file_path = os.path.join(OUTPUT_DIR, f'{now_datetime}_{duration}ms.wav')
            save_to_wave(file_path, buff)
            print(f'Duration: {duration:4}ms, Output: {file_path}')
        else:
            print('\r', '/' if state else '\\', end='')
            state = not state

    # Close serial
    port.close()

if __name__ == '__main__':
    target = from_tcp
    args = ()
    if len(sys.argv) > 1:
        target = from_serial
        args = (sys.argv[1],)

    thread = threading.Thread(target=target, args=args)
    thread.daemon = True
    thread.start()
    while True:
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            print('\nExited')
            exit(0)
