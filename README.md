# Hoozz Play Micro-speech Train
This project is used to train the micro-speech model.

Original reference: [train_micro_speech_model.ipynb](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/micro_speech/train/train_micro_speech_model.ipynb)

## Environment
```shell
> python --version
Python 3.7.9

> cat requirements.txt
absl-py==2.1.0
astor==0.8.1
gast==0.2.2
google-pasta==0.2.0
grpcio==1.62.2
h5py==2.10.0
importlib-metadata==6.7.0
Keras-Applications==1.0.8
Keras-Preprocessing==1.1.2
Markdown==3.4.4
MarkupSafe==2.1.5
numpy==1.18.5
opt-einsum==3.3.0
protobuf==3.20.0
six==1.16.0
tensorboard==1.15.0
tensorflow-estimator==1.15.1
tensorflow-gpu==1.15.5
termcolor==2.3.0
typing_extensions==4.7.1
Werkzeug==2.2.3
wrapt==1.16.0
zipp==3.15.0
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
|   |-- your_keywords_001
|   |   |-- your_wav_001.wav
|   |   |-- your_wav_002.wav
|   |   |-- ...
|   |-- your_keywords_002
|   |   |-- ...
|   |-- your_keywords_003
|   |   |-- ...
|   |-- your_keywords_004
|   |   |-- ...
|   |-- ...
|   `-- zero
|-- ...
`-- ...
```
