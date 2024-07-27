# Hoozz Play Micro-speech Train
This project is used to train the micro-speech model.

Original reference: [train_micro_speech_model.ipynb](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/micro_speech/train/train_micro_speech_model.ipynb)

## Environment

This script can be used for training on both Ubuntu and Windows. It depend on the following Python environment:
```shell
> python --version
Python 3.7.9

> pip --version
pip 24.0
```

## Initialize the training environment

Pull source code
```shell
git clone https://github.com/huxiangjs/hoozz_play_micro_speech_train.git
cd hoozz_play_micro_speech_train
git submodule update --init --recursive
```

Install all dependencies
```shell
pip install -r requirements.txt
```

## Prepare the dataset

We use the dataset provided by Google as the basic dataset, the download address is: [speech_commands_v0.02.tar.gz](https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz)


Create a **dataset/** directory and unzip the downloaded dataset to the **dataset/**  directory. The unzipped directory structure is roughly as follows:
```shell
.
|-- LICENSE
|-- README.md
|-- dataset
|   |-- LICENSE
|   |-- README.md
|   |-- _background_noise_
|   |-- backward
|   |-- ...
|   |-- yes
|   `-- zero
|-- model_config.py
|-- model_train.py
|-- requirements.txt
|-- speech_commands_v0.02.tar.gz
`-- tensorflow
    ...
```

Next, we need to prepare our own data set. The audio data has the following format requirements:
* File format: wav
* Sampling depth: 16 bits
* Channel: single channel
* Sampling rate: 16kHz

Once you are ready, name the folders according to the keywords and copy the data into the corresponding folders. After completing the above operations, your directory should look like this:
```shell
.
|-- ...
|-- dataset
|   |-- LICENSE
|   |-- README.md
|   |-- _background_noise_
|   |-- backward
|   |-- ...
|   |-- yes
|   |-- <your_keywords_001>
|   |   |-- <your_wav_001.wav>
|   |   |-- <your_wav_002.wav>
|   |   |-- ...
|   |-- <your_keywords_002>
|   |   |-- ...
|   |-- <your_keywords_003>
|   |   |-- ...
|   |-- <your_keywords_004>
|   |   |-- ...
|   |-- ...
|   `-- zero
|-- ...
`-- ...
```

## Adjust training profile

Open the `model_config.py` file and modify the following key parameters as needed:

1. Add your own keywords
   ```python
   WANTED_WORDS = "yes,no,<your_keywords_001>,<your_keywords_001>,..."
   ```

2. Adjust the number of training times and corresponding learning rate
   ```python
   TRAINING_STEPS = "12000,3000"
   LEARNING_RATE = "0.001,0.0001"
   ```

3. Adjust the maximum speech recognition time as needed (this time must be greater than the duration of the wav file). The above shows the configuration for recognizing speech with a maximum length of 1 second.
   ```python
   WINDOW_STRIDE_MS = 20 # Don't touch me! :-)
   ...
   FEATURE_COUNT = 49 # <-- Change it,  1000ms = 20ms * (49 + 1)
   CLIP_DURATION_MS = WINDOW_STRIDE_MS * (FEATURE_COUNT + 1) # Don't touch
   ```

4. Set the following option to true to automatically open the TensorBoard web page during training:
   ```python
   OPEN_WEB_AUTO = True
   ```

## Start training

Run the `python model_train.py` command to start training.
```shell
> python model_train.py
...
(few moments later...)
...
~~~~~~ All operations have been completed ~~~~~~~
TensorBoard URL: http://localhost:40176
Float model is 146136 bytes
Quantized model is 38240 bytes
Float model accuracy is 99.020979% (Number of test samples=715)
Quantized model accuracy is 99.160839% (Number of test samples=715)
Tflite model is 38240 bytes
[2024-06-29 17:57:51.332231 ~ 2024-06-29 19:30:24.322574] Time cost: 1:32:32.990343

Press any key to exit...
```

At this point, your model is output in the `models/` directory.
```shell
$ tree models/
models/
|-- float_model.tflite
|-- model.cc
|-- model.tflite
`-- saved_model
    |-- saved_model.pb
    `-- variables
        |-- variables.data-00000-of-00001
        `-- variables.index

2 directories, 6 files
```

* model.tflite: Tensorflow Lite Quantized Model
* model.cc: tensorflow lite micro quantization model c language format

**For the** `Voice LED` **project, you only need to copy model.cc to the project and overwrite the original file.**


## Good luck

(๑❛ᴗ❛๑)
