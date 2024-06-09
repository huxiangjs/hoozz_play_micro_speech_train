# -*- coding: utf-8 -*-

# refs: https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/micro_speech/train/train_micro_speech_model.ipynb

import model_config
import subprocess
import random
import webbrowser
import sys
import input_data
import models
import numpy as np
import tensorflow as tf
import datetime

model_settings = models.prepare_model_settings(
    len(input_data.prepare_words_list(model_config.WANTED_WORDS.split(','))),
    model_config.SAMPLE_RATE, model_config.CLIP_DURATION_MS, model_config.WINDOW_SIZE_MS,
    model_config.WINDOW_STRIDE, model_config.FEATURE_BIN_COUNT, model_config.PREPROCESS)
audio_processor = input_data.AudioProcessor(
    model_config.DATA_URL, model_config.DATASET_DIR,
    model_config.SILENT_PERCENTAGE, model_config.UNKNOWN_PERCENTAGE,
    model_config.WANTED_WORDS.split(','), model_config.VALIDATION_PERCENTAGE,
    model_config.TESTING_PERCENTAGE, model_settings, model_config.LOGS_DIR)

train_report = []

def model_quantize():
    with tf.Session() as sess:
        float_converter = tf.lite.TFLiteConverter.from_saved_model(model_config.SAVED_MODEL)
        float_tflite_model = float_converter.convert()
        float_tflite_model_size = open(model_config.FLOAT_MODEL_TFLITE, "wb").write(float_tflite_model)
        report = "Float model is %d bytes" % float_tflite_model_size
        print(report)
        train_report.append(report)

        converter = tf.lite.TFLiteConverter.from_saved_model(model_config.SAVED_MODEL)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.inference_input_type = tf.lite.constants.INT8
        converter.inference_output_type = tf.lite.constants.INT8
        def representative_dataset_gen():
            for i in range(100):
                data, _ = audio_processor.get_data(1, i*1, model_settings,
                                                   model_config.BACKGROUND_FREQUENCY, 
                                                   model_config.BACKGROUND_VOLUME_RANGE,
                                                   model_config.TIME_SHIFT_MS,
                                                   'testing',
                                                   sess)
            flattened_data = np.array(data.flatten(), dtype=np.float32).reshape(1, 1960)
            yield [flattened_data]
        converter.representative_dataset = representative_dataset_gen
        tflite_model = converter.convert()
        tflite_model_size = open(model_config.MODEL_TFLITE, "wb").write(tflite_model)
        report = "Quantized model is %d bytes" % tflite_model_size
        print(report)
        train_report.append(report)

# Helper function to run inference
def run_tflite_inference(tflite_model_path, model_type="Float"):
    # Load test data
    np.random.seed(0) # set random seed for reproducible test results.
    with tf.Session() as sess:
        test_data, test_labels = audio_processor.get_data(
            -1, 0, model_settings, model_config.BACKGROUND_FREQUENCY,
            model_config.BACKGROUND_VOLUME_RANGE,
            model_config.TIME_SHIFT_MS, 'testing', sess)
    test_data = np.expand_dims(test_data, axis=1).astype(np.float32)

    # Initialize the interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # For quantized models, manually quantize the input data from float to integer
    if model_type == "Quantized":
        input_scale, input_zero_point = input_details["quantization"]
        test_data = test_data / input_scale + input_zero_point
        test_data = test_data.astype(input_details["dtype"])

    correct_predictions = 0
    for i in range(len(test_data)):
        interpreter.set_tensor(input_details["index"], test_data[i])
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]
        top_prediction = output.argmax()
        correct_predictions += (top_prediction == test_labels[i])

    report = '%s model accuracy is %f%% (Number of test samples=%d)' % (
        model_type, (correct_predictions * 100) / len(test_data), len(test_data))
    print(report)
    train_report.append(report)

def model_formate_output():
    with open(model_config.MODEL_TFLITE_MICRO, 'w', encoding='utf-8') as fout:
        model_data = []
        with open(model_config.MODEL_TFLITE, 'rb') as fin:
            model_data = fin.read()

        report = "Tflite model is %d bytes" % len(model_data)
        print(report)
        train_report.append(report)

        all_data = list(model_data)
        out_data = ['static const unsigned char tflite_model_data[] = {\n\t']
        line_count = 0
        for item in all_data:
            line_count += 1
            s = str(f'0x{item:02x},')
            out_data.append(s)
            out_data.append('\n\t' if line_count % 12 == 0 else ' ')
        out_data[-1] = '\n' if out_data[-1] == ' ' else out_data[-1]
        out_data.append('};\n\n')
        out_data.append(str(f'static const unsigned int tflite_model_size = {len(model_data)};\n'))
        fout.writelines(''.join(out_data))

def main():
    start_time = datetime.datetime.now()
    data_url = model_config.DATA_URL if model_config.DATA_URL is not None else ''
    train_info = [
        'python',
        str(f'{model_config.SPEECH_COMMANDS_HOME}/train.py'),
        str(f'--data_url={data_url}'),
        str(f'--data_dir={model_config.DATASET_DIR}'),
        str(f'--wanted_words={model_config.WANTED_WORDS}'),
        str(f'--silence_percentage={model_config.SILENT_PERCENTAGE}'),
        str(f'--unknown_percentage={model_config.UNKNOWN_PERCENTAGE}'),
        str(f'--preprocess={model_config.PREPROCESS}'),
        str(f'--window_stride={model_config.WINDOW_STRIDE}'),
        str(f'--model_architecture={model_config.MODEL_ARCHITECTURE}'),
        str(f'--how_many_training_steps={model_config.TRAINING_STEPS}'),
        str(f'--learning_rate={model_config.LEARNING_RATE}'),
        str(f'--train_dir={model_config.TRAIN_DIR}'),
        str(f'--summaries_dir={model_config.LOGS_DIR}'),
        str(f'--verbosity={model_config.VERBOSITY}'),
        str(f'--eval_step_interval={model_config.EVAL_STEP_INTERVAL}'),
        str(f'--save_step_interval={model_config.SAVE_STEP_INTERVAL}'),
    ]
    train_process = subprocess.Popen(train_info, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    # print(train_process.stdout)

    tensorboard_host = 'localhost'
    tensorboard_port = int(random.uniform(6006, 65535))
    tensorboard_info = [
        'tensorboard',
        str(f'--host={tensorboard_host}'),
        str(f'--port={tensorboard_port}'),
        str(f'--logdir={model_config.LOGS_DIR}'),
    ]
    tensorboard_process = subprocess.Popen(tensorboard_info, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    tensorboard_url = str(f'http://{tensorboard_host}:{tensorboard_port}')
    if (model_config.OPEN_WEB_AUTO):
        webbrowser.open(tensorboard_url)
    # print(tensorboard_process.stdout)
    report = str(f'TensorBoard URL: {tensorboard_url}')
    train_report.append(report)

    train_process.wait()

    freeze_info = [
        'python',
        str(f'{model_config.SPEECH_COMMANDS_HOME}/freeze.py'),
        str(f'--wanted_words={model_config.WANTED_WORDS}'),
        str(f'--window_stride_ms={model_config.WINDOW_STRIDE}'),
        str(f'--preprocess={model_config.PREPROCESS}'),
        str(f'--model_architecture={model_config.MODEL_ARCHITECTURE}'),
        str(f'--start_checkpoint={model_config.TRAIN_DIR}{model_config.MODEL_ARCHITECTURE}.ckpt-{model_config.TOTAL_STEPS}'),
        str(f'--save_format=saved_model'),
        str(f'--output_file={model_config.SAVED_MODEL}'),
    ]
    freeze_process = subprocess.Popen(freeze_info, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    # print(freeze_process.stdout)

    freeze_process.wait()
    model_quantize()

    # Compute float model accuracy
    run_tflite_inference(model_config.FLOAT_MODEL_TFLITE)
    # Compute quantized model accuracy
    run_tflite_inference(model_config.MODEL_TFLITE, model_type='Quantized')

    model_formate_output()

    end_time = datetime.datetime.now()
    report = str(f'[{start_time} ~ {end_time}] Time cost: {end_time - start_time}')
    train_report.append(report)

    print('\n~~~~~~ All operations have been completed ~~~~~~~')
    print('\n'.join(train_report))

    input('\nPress any key to exit...')
    tensorboard_process.kill()

if __name__ == '__main__':
    main()
