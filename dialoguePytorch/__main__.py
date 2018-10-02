
from model import ABHUE
from losses import DistanceClusterLoss
import preprocess
import torch
import torch.optim as optim
import torch.nn as nn
import os

PATH = os.path.abspath(os.path.dirname(__file__))

def main():
    # test data path
    # dataPath = "/home/stoplime/workspace/audiobook/OpenAudioAI/data/train/train_0"
    dataPath = os.path.join(PATH, "..", "data", "train", "train_0")

    preprocessor = preprocess.PreProcess()
    model = ABHUE()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device)
    model = model.to(device)
    window_size = 3
    batch_size = 8

    optimizer = optim.Adam(model.parameters())
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    exp_lr_scheduler.step() # update lr scheduler epoch
    model.train()

    loss_function = DistanceClusterLoss()

    sliding_window = []
    batch_outputs = []
    batch_labels = []
    for data in preprocessor.parseData(dataPath):
        sliding_window.append(data)
        if len(sliding_window) < window_size:
            continue
        elif len(sliding_window) > window_size:
            sliding_window.pop(0)
        
        # print("sliding_window", len(sliding_window))

        data_input = []
        for i, sentence in enumerate(sliding_window):
            words = []
            for word in sentence[0]:
                # Converts the word imbedding into a pytorch tensor per word
                words.append(torch.tensor(word).to(device).unsqueeze(0).unsqueeze(0))
            # data_input will have a size of [3] for the sentence and in each sentence there will be a list of embedings of size words
            data_input.append(words)

            # if middle sentence
            if i == (len(sliding_window) - 1) / 2:
                # The label will come from per sentence[1]
                data_label = sentence[1]
        
        # print("data_input middle sentence", len(data_input[int((window_size - 1) / 2)]))
        model.zero_grad()
        output = model(data_input)
        # print("output", output.shape)
        
        batch_outputs.append(output)
        batch_labels.append(data_label)

        if len(batch_labels) >= batch_size:
            # backpropagate through batch
            loss_value = loss_function(batch_outputs, batch_labels)
            loss_value.backward()
            optimizer.step()
            # Clear batch
            batch_outputs = []
            batch_labels = []


if __name__ == '__main__':
    main()
