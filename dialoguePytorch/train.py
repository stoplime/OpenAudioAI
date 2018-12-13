
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

    running_loss = 0
    data_index = 0
    for data in params.preprocessor.parseData(os.path.join(params.train_data_dir, params.data_file)):
        # Create window, if window not correct size, skip loop
        if not params.preprocessor.create_sliding_window(data):
            continue

        data_input, data_label = params.preprocessor.tensorfy()

        local_output = params.local_model(data_input)
        output = params.global_model(local_output)

        loss_value = params.loss_function(output, data_label)
        loss_value.backward(retain_graph=True)
        params.optimizer.step()

        running_loss += loss_value.item()
        print('[{}] Training loss: {}, {}'.format(
                                            (data_index + 1), 
                                            format(round(running_loss / (data_index + 1), 10), '.10f'), 
                                            using("Memory")),
            end='\r', flush=True, file=log)
        print('[{}] Training loss: {}, {}'.format(
                                            (data_index + 1), 
                                            format(round(running_loss / (data_index + 1), 10), '.10f'), 
                                            using("Memory")),
            end='\r', flush=True)

        params.local_model.reset_gradients()
        data_index += 1
    # clear line
    print("", file=log)
    print("")
    params.preprocessor.clear_sliding_window()
    params.global_model.reset_gradients()
    # Save the model after every training file
    torch.save(params.local_model.state_dict(), params.save_model_path)
    torch.save(params.global_model.state_dict(), params.save_model_global_path)

    # initialize for next iteration
    params.Reset_Model()

    return "Done"

# @profile
def main():
    test_params = training_parameters.training_parameters()
    test_params.Hyperparameter_Initialization()
    test_params.Path_Initialization()
    test_params.Preprocessing_Initialization()
    
    # Training log
    log = open(test_params.log_file_path, "a")
    # ********************* Log To File *********************
    print("Settings:", file=log)
    print("Dynamic HyperParams:", file=log)
    print("Recurrent Model:", test_params.recurrent_model, file=log)
    print("Window Size:", test_params.window_size, file=log)
    print("Dropout:", test_params.dropout, file=log)
    print("Prev/Post Stack Size:", test_params.stack_size, file=log)
    print("", file=log)

    print("Const HyperParams:", file=log)
    print("Device:", test_params.device, file=log)
    print("Total Epochs:", test_params.epochs, file=log)
    print("Batch Size:", test_params.batch_size, file=log)
    print("Max Speakers:", test_params.max_speakers, file=log)
    print("", file=log)

    # ********************* Print to Terminal *********************
    print("Settings:")
    print("Dynamic HyperParams:")
    print("Recurrent Model:", test_params.recurrent_model)
    print("Window Size:", test_params.window_size)
    print("Dropout:", test_params.dropout)
    print("Prev/Post Stack Size:", test_params.stack_size)
    print("")

    print("Const HyperParams:")
    print("Device:", test_params.device)
    print("Total Epochs:", test_params.epochs)
    print("Batch Size:", test_params.batch_size)
    print("Max Speakers:", test_params.max_speakers)
    print("")

    # ********************* *********************
    for epoch in range(test_params.epochs):
        print("epoch:", epoch, file=log)
        print("epoch:", epoch)

        # training
        for data_file in natsorted(os.listdir(test_params.train_data_dir)):
            print("Training file:", data_file, file=log)
            print("Training file:", data_file)
            test_params.data_file = data_file
            with concurrent.futures.ProcessPoolExecutor() as executor:
                future = executor.submit(train, (test_params,))
                print(future)
                while not future.done():
                    time.sleep(1)
                
                try:
                    data = future.result()
                    print(data)
                except Exception as e:
                    print(e)

        # Validtion
        test_params.Model_Initialization()
        batch_datas = []
        label_datas = []
        data_index = 0
        batch_count = 0
        test_params.preprocessor.clear_sliding_window()

        for data_file in natsorted(os.listdir(test_params.val_data_dir)):
            print("Validation file:", data_file, file=log)
            print("Validation file:", data_file)

            for data in test_params.preprocessor.parseData(os.path.join(test_params.val_data_dir, data_file)):
                # Create window, if window not correct size, skip loop
                if not test_params.preprocessor.create_sliding_window(data):
                    continue

                data_input, data_label = test_params.preprocessor.tensorfy()

                local_output = test_params.local_model(data_input)
                output = test_params.global_model(local_output)

                batch_datas.append((data_index, output.detach()))
                label_datas.append((data_index, data_label))

                num_speakers = inference.CheckNumOfSpeakers(label_datas)
                if data_index+1 >= test_params.batch_size or num_speakers >= test_params.max_speakers:
                    km = Kmeans(k=num_speakers, size=200)
                    km.run( batch_datas )
                    score = inference.bestLabels(km.clusters, label_datas, num_speakers)
                    # round(score / (data_index + 1), 10)
                    print('[{}] Inference Score: {} \t Batch Size: {} \t Speakers: {}'.format((batch_count + 1), score, data_index+1, num_speakers), file=log)
                    print('[{}] Inference Score: {} \t Batch Size: {} \t Speakers: {}'.format((batch_count + 1), score, data_index+1, num_speakers))
                    batch_count += 1
                    data_index = 0
                    batch_datas = []
                    label_datas = []
                else:
                    data_index += 1
            print("", file=log)
            print("")
            test_params.preprocessor.clear_sliding_window()

if __name__ == '__main__':
    main()
