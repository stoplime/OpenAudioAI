
from model import ABHUE, GlobalModule
from losses import DistanceClusterLoss
import preprocess
import torch
import torch.optim as optim
import torch.nn as nn
import os
import time
from k_means import Kmeans
import inference
import argparse
import importlib
from natsort.natsort import natsorted
from memCheck import using
import concurrent.futures
import training_parameters

def train(params):
    params.Model_Initialization()
    log = open(params.log_file_path, "a")

    prev_loss = 0
    data_index = 0
    for data in params.preprocessor.parseData(os.path.join(params.train_data_dir, params.data_file)):
        # Create window, if window not correct size, skip loop
        if not params.preprocessor.create_sliding_window(data):
            continue

        data_input, data_label = params.preprocessor.tensorfy()

        local_output = params.local_model(data_input)
        output = params.global_model(local_output)

        loss_value = params.loss_function(output, data_label)
        if loss_value.item() > prev_loss * 10 and prev_loss > 0.00000001:
            log_print("Gradient Spike from {} to {}, gradient has been reset".format(prev_loss, loss_value.item()), log=log)
            params.local_model.reset_gradients()
        else:
            loss_value.backward(retain_graph=True)
        params.optimizer.step()

        prev_loss = loss_value.item()
        log_print('[{}] Training loss: {}, {}'.format(
                                            (data_index + 1),
                                            format(round(loss_value.item() / (data_index + 1), 10), '.10f'), 
                                            using("Memory")),
            end='\r', flush=True, log=log)

        params.local_model.reset_gradients()
        data_index += 1
    # clear line
    log_print("", log=log)
    params.preprocessor.clear_sliding_window()
    params.global_model.reset_gradients()
    # Save the model after every training file
    torch.save(params.local_model.state_dict(), params.save_model_path)
    torch.save(params.global_model.state_dict(), params.save_model_global_path)

    # initialize for next iteration
    params.Reset_Model()
    log.close()

    return "Done"

def val(params):
    params.Model_Initialization()
    log = open(params.log_file_path, "a")
    batch_datas = []
    label_datas = []
    data_index = 0
    batch_count = 0

    for data in params.preprocessor.parseData(os.path.join(params.val_data_dir, params.data_file)):
        # Create window, if window not correct size, skip loop
        if not params.preprocessor.create_sliding_window(data):
            continue

        data_input, data_label = params.preprocessor.tensorfy()

        local_output = params.local_model(data_input)
        output = params.global_model(local_output)

        batch_datas.append((data_index, output.detach()))
        label_datas.append((data_index, data_label))

        loss_value = params.loss_function(output, data_label)

        num_speakers = inference.CheckNumOfSpeakers(label_datas)
        if data_index+1 >= params.batch_size or num_speakers >= params.max_speakers:
            km = Kmeans(k=num_speakers, size=200)
            km.run( batch_datas )
            score = inference.bestLabels(km.clusters, label_datas, num_speakers)
            
            log_print('[{}] Inference Score: {} \t Batch Size: {} \t Speakers: {} \t Val Loss: {}'.format(
                        (batch_count + 1), score, data_index+1, num_speakers, 
                        format(round(loss_value.item() / (data_index + 1), 10), '.10f')), end='\r', flush=True, log=log)
            batch_count += 1
            data_index = 0
            batch_datas = []
            label_datas = []
        else:
            data_index += 1
    # clear line
    log_print("", log=log)
    params.preprocessor.clear_sliding_window()
    params.Reset_Model()
    log.close()

    return "Done"

def Log_Initialize(params, log):
    # ********************* Log To File *********************
    log_print("Settings A{}:".format(params.model_id), log=log)
    log_print("Dynamic HyperParams:", log=log)
    log_print("Recurrent Model:", params.recurrent_model, log=log)
    log_print("Window Size:", params.window_size, log=log)
    log_print("Dropout:", params.dropout, log=log)
    log_print("Prev/Post Stack Size:", params.stack_size, log=log)
    log_print("", log=log)

    log_print("Const HyperParams:", log=log)
    log_print("Device:", params.device, log=log)
    log_print("Total Epochs:", params.epochs, log=log)
    log_print("Batch Size:", params.batch_size, log=log)
    log_print("Max Speakers:", params.max_speakers, log=log)
    log_print("", log=log)
    # ********************* *********************

def log_print(*obj, log, end='\n', flush=False):
    print(*obj, end=end, flush=flush, file=log)
    print(*obj, end=end, flush=flush)

def Run_Params(params):
    print("Running params")
    # Setup log
    log = open(params.log_file_path, "a")
    Log_Initialize(params, log)
    print("first log")

    for epoch in range(params.epochs):
        log_print("epoch:", epoch, log=log)
        print("first epoch")

        # training
        for data_file in natsorted(os.listdir(params.train_data_dir)):
            log_print("Training file:", data_file, log=log)
            print("first Training")
            log.close()

            params.data_file = data_file
            with concurrent.futures.ProcessPoolExecutor() as executor:
                future = executor.submit(train, params)
                executor.shutdown(wait=True)
            log = open(params.log_file_path, "a")
            log_print("File", data_file, "is Done", log=log)
        params.preprocessor.clear_sliding_window()

        # Validtion
        for data_file in natsorted(os.listdir(params.val_data_dir)):
            log_print("Validation file:", data_file, log=log)
            log.close()

            params.data_file = data_file
            with concurrent.futures.ProcessPoolExecutor() as executor:
                future = executor.submit(val, params)
                executor.shutdown(wait=True)
            log = open(params.log_file_path, "a")
            log_print("File", data_file, "is Done", log=log)
        params.preprocessor.clear_sliding_window()
    
    return "Done"

def Multi_Params_Initialization(start=0, end=24, Test_log=False):
    multi_params = []
    preprocess_param = None
    once = True
    for i in range(start, end):
        if i % 2 == 0:
            model_type = "lstm"
        else:
            model_type = "gru"
        
        sub_i = int(i/2)
        if sub_i % 3 == 0:
            window_size = 3
        elif sub_i % 3 == 1:
            window_size = 5
        else:
            window_size = 7

        no_stack = (i < 6)
        if no_stack:
            dropout = 0
            stack_size = 1
        else:
            stack_size = 2
            sub_stack_i = int((i-6)/6)
            if sub_stack_i % 3 == 0:
                dropout = 0.2
            elif sub_stack_i % 3 == 1:
                dropout = 0.5
            else:
                dropout = 0.7

        param = training_parameters.training_parameters()
        param.Hyperparameter_Initialization(
            device_id       = 0,
            model_id        = i,
            recurrent_model = model_type,
            window_size     = window_size,
            dropout         = dropout,
            stack_size      = stack_size
        )
        param.Path_Initialization()
        param.Device_Initialization()
        if once:
            once = False
            param.Preprocessing_Initialization()
            preprocess_param = param
        else:
            param.Preprocessing_Initialization(glove_data=preprocess_param.preprocessor.glove)
        multi_params.append(param)

    if Test_log:
        for params in multi_params:
            log = open(multi_params[0].log_file_path, "a")
            Log_Initialize(params, log)

    return multi_params

# @profile
def main():
    torch.multiprocessing.set_start_method('spawn', force=True)
    test_params = training_parameters.training_parameters()
    test_params.Hyperparameter_Initialization(
        device_id       = 0,
        model_id        = 0,
        recurrent_model = "lstm",
        window_size     = 3,
        dropout         = 0,
        stack_size      = 1
    )
    test_params.Path_Initialization(testing=True)
    test_params.Device_Initialization()
    test_params.Preprocessing_Initialization()

    Run_Params(test_params)

    # multi_params = Multi_Params_Initialization(end=3)

    # print("done multi_params")
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     future = executor.submit(Run_Params, multi_params)
    #     executor.shutdown(wait=True)
    #     print("future", future.done())

if __name__ == '__main__':
    main()
