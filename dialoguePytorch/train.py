
from model import ABHUE
from losses import DistanceClusterLoss
import preprocess
import torch
import torch.optim as optim
import torch.nn as nn
import os
import time
from k_means import Kmeans
import inference

PATH = os.path.abspath(os.path.dirname(__file__))

train_data_dir = os.path.join(PATH, "..", "data", "train")
val_data_dir = os.path.join(PATH, "..", "data", "val")

load_form_save = False
epochs = 30
window_size = 5
batch_size = 32
max_speakers = 10

time_stamp = str(time.strftime("%Y_%m_%d-%H_%M_%S"))

save_model_name = "model_" + time_stamp + ".pt"
save_model_path = os.path.join(PATH, "saves", save_model_name)

log_file_name = "training_log_" + time_stamp + ".log"
log_file_path = os.path.join(PATH, "logs", log_file_name)

# def train(data, model, loss_function, optimizer, verbose=1):


def main():
    # test data path
    # dataPath = "/home/stoplime/workspace/audiobook/OpenAudioAI/data/train/train_0"
    
    # initialize preprocess and model
    preprocessor = preprocess.PreProcess(window_size)
    model = ABHUE()

    # Set Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device)
    model = model.to(device)
    
    # Create Save Dir or load saved model
    if not os.path.exists(os.path.dirname(save_model_path)):
        os.makedirs(os.path.dirname(save_model_path))
    elif load_form_save:
        model.load_state_dict(torch.load(save_model_path))

    # create log file directory
    if not os.path.exists(os.path.dirname(log_file_path)):
        os.makedirs(os.path.dirname(log_file_path))
    
    # Set optimizer
    optimizer = optim.Adam(model.parameters())
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    exp_lr_scheduler.step() # update lr scheduler epoch

    # Set model to train mode
    model.train()

    # Define loss function
    loss_function = DistanceClusterLoss(batch_size)

    # Training log
    log = open(log_file_path, "a")
    print("Settings:", file=log)
    print("Device:", device, file=log)
    print("Total Epochs:", epochs, file=log)
    print("Window Size:", window_size, file=log)
    print("Batch Size:", batch_size, file=log)
    print("Max Speakers:", max_speakers, file=log)

    print("Settings:")
    print("Device:", device)
    print("Total Epochs:", epochs)
    print("Window Size:", window_size)
    print("Batch Size:", batch_size)
    print("Max Speakers:", max_speakers)

    for epoch in range(epochs):
        print("epoch:", epoch, file=log)
        print("epoch:", epoch)

        # training
        batch_outputs = []
        batch_labels = []
        for data_file in os.listdir(train_data_dir):
            print("Training file:", data_file, file=log)
            print("Training file:", data_file)
            running_loss = 0;
            data_idx = 0
            for data in preprocessor.parseData(os.path.join(train_data_dir, data_file)):
                if not preprocessor.create_sliding_window(data):
                    continue
                
                data_input, data_label = preprocessor.tensorfy()
                
                # print("data_input middle sentence", len(data_input[int((window_size - 1) / 2)]))
                output = model(data_input)
                # print("output", output.shape)
                
                # batch_outputs.append(output)
                # batch_labels.append(data_label)

                # if len(batch_labels) >= batch_size:
                    # backpropagate through batch
                loss_value = loss_function(output, data_label)
                loss_value.backward(retain_graph=True)
                optimizer.step()

                running_loss += loss_value.item()
                print('[{}] Training loss: {}'.format((data_idx + 1), round(running_loss / (data_idx + 1), 10)), end='\r', flush=True, file=log)
                print('[{}] Training loss: {}'.format((data_idx + 1), round(running_loss / (data_idx + 1), 10)), end='\r', flush=True)

                    # Clear batch
                    # batch_outputs = []
                    # batch_labels = []
                torch.save(model.state_dict(), save_model_path)
                data_idx += 1
            # clear line
            print("", file=log)
            print("")
            preprocessor.clear_sliding_window()
        
        # Validtion
        batch_datas = []
        label_datas = []
        data_idx = 0
        batch_count = 0
        preprocessor.clear_sliding_window()

        for data_file in sorted(os.listdir(val_data_dir)):
            print("Validation file:", data_file, file=log)
            print("Validation file:", data_file)

            for data in preprocessor.parseData(os.path.join(val_data_dir, data_file)):
                if not preprocessor.create_sliding_window(data):
                    continue
                # print("Run")
                
                data_input, data_label = preprocessor.tensorfy()

                output = model(data_input)
                batch_datas.append((data_idx, output.detach()))
                label_datas.append((data_idx, data_label))

                num_speakers = inference.CheckNumOfSpeakers(label_datas)
                if data_idx+1 >= batch_size or num_speakers >= max_speakers:
                    km = Kmeans(k=num_speakers, size=200)
                    km.run( batch_datas )
                    score = inference.bestLabels(km.clusters, label_datas, num_speakers)
                    # round(score / (data_idx + 1), 10)
                    print('[{}] Inference Score: {} \t Batch Size: {} \t Speakers: {}'.format((batch_count + 1), score, data_idx+1, num_speakers), file=log)
                    print('[{}] Inference Score: {} \t Batch Size: {} \t Speakers: {}'.format((batch_count + 1), score, data_idx+1, num_speakers))
                    batch_count += 1
                    data_idx = 0
                    batch_datas = []
                    label_datas = []
                else:
                    data_idx += 1
            print("", file=log)
            print("")
            preprocessor.clear_sliding_window()




if __name__ == '__main__':
    main()
