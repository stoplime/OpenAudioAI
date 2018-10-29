
from model import ABHUE
from losses import DistanceClusterLoss
import preprocess
import torch
import torch.optim as optim
import torch.nn as nn
import os

PATH = os.path.abspath(os.path.dirname(__file__))

data_dir = os.path.join(PATH, "..", "data", "train")

window_size = 3
batch_size = 64


# def train(data, model, loss_function, optimizer, verbose=1):


def main():
    # test data path
    # dataPath = "/home/stoplime/workspace/audiobook/OpenAudioAI/data/train/train_0"

    preprocessor = preprocess.PreProcess(window_size)
    model = ABHUE()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters())
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    exp_lr_scheduler.step() # update lr scheduler epoch
    model.train()

    loss_function = DistanceClusterLoss(batch_size)

    batch_outputs = []
    batch_labels = []
    for data_file in os.listdir(data_dir):
        print("Training file:", data_file)
        running_loss = 0;
        data_idx = 0
        for data in preprocessor.parseData(os.path.join(data_dir, data_file)):
            if not preprocessor.create_sliding_window(data):
                continue
            
            data_input, data_label = preprocessor.tensorfy()
            
            # print("data_input middle sentence", len(data_input[int((window_size - 1) / 2)]))
            model.train()
            output = model(data_input)
            # print("output", output.shape)
            
            batch_outputs.append(output)
            batch_labels.append(data_label)

            # if len(batch_labels) >= batch_size:
                # backpropagate through batch
            loss_value = loss_function(output, data_label)
            loss_value.backward()
            optimizer.step()

            running_loss += loss_value.item()
            print('[{}] Training loss: {}'.format((data_idx + 1), round(running_loss / (data_idx + 1), 10)), end='\r', flush=True)

                # Clear batch
                # batch_outputs = []
                # batch_labels = []
            
            data_idx += 1
        # clear line
        print("")


if __name__ == '__main__':
    main()
