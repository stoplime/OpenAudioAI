
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

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

PATH = os.path.abspath(os.path.dirname(__file__))

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('id', default=-1,
                    help='Id of the model')
parser.add_argument('model', default="LSTM",
                    help='Model Type')
parser.add_argument('window', default=3,
                    help='Window Size')
parser.add_argument('dropout', default=0.2,
                    help='Dropout Rate')
parser.add_argument('stack', default=1,
                    help='Stack size of RNN')
parser.add_argument('device', default=0,
                    help='GPU device')
args = parser.parse_args()

train_data_dir = os.path.join(PATH, "..", "data", "train")
val_data_dir = os.path.join(PATH, "..", "data", "val")

load_form_save = False
epochs = 30
batch_size = 32
max_speakers = 10

dev = 0
model_id = 1
recurrent_model = "gru"    # lstm or gru
window_size = 7             # 3, 5, 7
dropout = 0.2               # 0.2, 0.5, 0.8
stack_size = 1              # 1 or 2

if args != None:
    model_id = int(args.id)
    recurrent_model = str(args.model)
    window_size = int(args.window)
    dropout = float(args.dropout)
    stack_size = int(args.stack)
    dev = str(args.device)

    print("Model ID:", model_id)
    print("Recurrent Model:", recurrent_model)
    print("Window Size:", window_size)
    print("Dropout:", dropout)
    print("Prev/Post Stack Size:", stack_size)
    print("")

time_stamp = str(time.strftime("%Y_%m_%d-%H_%M_%S"))

save_model_name = "A" + str(model_id) + "_model_" + time_stamp + ".pt"
save_model_global_name = "A" + str(model_id) + "_model_global_" + time_stamp + ".pt"
save_model_path = os.path.join(PATH, "saves", save_model_name)
save_model_global_path = os.path.join(PATH, "saves", save_model_global_name)

log_file_name = "A" + str(model_id) + "_training_log_" + time_stamp + ".log"
log_file_path = os.path.join(PATH, "logs", log_file_name)

# def train(data, model, loss_function, optimizer, verbose=1):

def main():
    # Set Device
    device = torch.device("cuda:"+str(dev) if torch.cuda.is_available() else "cpu")
    print("device", device)
    
    # initialize preprocess and model
    preprocessor = preprocess.PreProcess(window_size, dev=device)
    local_model = ABHUE(recurrent_model=recurrent_model, dropout=dropout, stack_size=stack_size, dev=device)
    global_model = GlobalModule(recurrent_model=recurrent_model, dropout=dropout, stack_size=stack_size, dev=device)

    # Create Save Dir or load saved model
    if not os.path.exists(os.path.dirname(save_model_path)):
        os.makedirs(os.path.dirname(save_model_path))
    elif load_form_save:
        local_model.load_state_dict(torch.load(save_model_path))
        global_model.load_state_dict(torch.load(save_model_global_path))

    # create log file directory
    if not os.path.exists(os.path.dirname(log_file_path)):
        os.makedirs(os.path.dirname(log_file_path))
    
    # Set optimizer
    optimizer = optim.Adam( list(local_model.parameters()) + list(global_model.parameters()) )
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Set model to train mode
    local_model.train()
    global_model.train()

    # Define loss function
    loss_function = DistanceClusterLoss(batch_size, dev=device)

    # Training log
    log = open(log_file_path, "a")
    # ********************* Log To File *********************
    print("Settings:", file=log)
    print("Dynamic HyperParams:", file=log)
    print("Recurrent Model:", recurrent_model, file=log)
    print("Window Size:", window_size, file=log)
    print("Dropout:", dropout, file=log)
    print("Prev/Post Stack Size:", stack_size, file=log)
    print("", file=log)

    print("Const HyperParams:", file=log)
    print("Device:", device, file=log)
    print("Total Epochs:", epochs, file=log)
    print("Batch Size:", batch_size, file=log)
    print("Max Speakers:", max_speakers, file=log)
    print("", file=log)

    # ********************* Print to Terminal *********************
    print("Settings:")
    print("Dynamic HyperParams:")
    print("Recurrent Model:", recurrent_model)
    print("Window Size:", window_size)
    print("Dropout:", dropout)
    print("Prev/Post Stack Size:", stack_size)
    print("")

    print("Const HyperParams:")
    print("Device:", device)
    print("Total Epochs:", epochs)
    print("Batch Size:", batch_size)
    print("Max Speakers:", max_speakers)
    print("")
    
    # ********************* *********************
    for epoch in range(epochs):
        print("epoch:", epoch, file=log)
        print("epoch:", epoch)

        # training
        batch_outputs = []
        batch_labels = []
        for data_file in os.listdir(train_data_dir):
            print("Training file:", data_file, file=log)
            print("Training file:", data_file)
            running_loss = 0
            data_idx = 0
            for data in preprocessor.parseData(os.path.join(train_data_dir, data_file)):
                # Create window, if window not correct size, skip loop
                if not preprocessor.create_sliding_window(data):
                    continue
                
                data_input, data_label = preprocessor.tensorfy()
                
                local_output = local_model(data_input)
                output = global_model(local_output)
                
                loss_value = loss_function(output, data_label)
                loss_value.backward(retain_graph=True)
                optimizer.step()

                running_loss += loss_value.item()
                print('[{}] Training loss: {}'.format((data_idx + 1), round(running_loss / (data_idx + 1), 10)), end='\r', flush=True, file=log)
                print('[{}] Training loss: {}'.format((data_idx + 1), round(running_loss / (data_idx + 1), 10)), end='\r', flush=True)

                local_model.reset_gradients()
                torch.save(local_model.state_dict(), save_model_path)
                torch.save(global_model.state_dict(), save_model_global_path)
                data_idx += 1
            # clear line
            print("", file=log)
            print("")
            preprocessor.clear_sliding_window()
            global_model.reset_gradients()
        
        # update lr scheduler epoch
        exp_lr_scheduler.step()

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
                # Create window, if window not correct size, skip loop
                if not preprocessor.create_sliding_window(data):
                    continue
                
                data_input, data_label = preprocessor.tensorfy()

                local_output = local_model(data_input)
                output = global_model(local_output)
                
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
