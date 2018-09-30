
from model import ABHUE
import preprocess
import torch
import torch.optim as optim
import torch.nn as nn


def main():
    dataPath = "/home/stoplime/workspace/audiobook/OpenAudioAI/data/train/train_0"
    preprocessor = preprocess.PreProcess()
    model = ABHUE()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device)
    model = model.to(device)
    window_size = 3

    optimizer = optim.Adam(model.parameters())
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    exp_lr_scheduler.step() # update lr scheduler epoch
    model.train()

    # TODO: Actually decide on a good loss function
    loss_function = nn.MSELoss()

    sliding_window = []
    for data in preprocessor.parseData(dataPath):
        sliding_window.append(data)
        if len(sliding_window) < window_size:
            continue
        elif len(sliding_window) > window_size:
            sliding_window.pop(0)
        
        print("sliding_window", len(sliding_window))

        data_input = []
        for sentence in sliding_window:
            words = []
            for word in sentence[0]:
                # Converts the word imbedding into a pytorch tensor per word
                words.append(torch.tensor(word).to(device).unsqueeze(0).unsqueeze(0))
            # data_input will have a size of [3] for the sentence and in each sentence there will be a list of embedings of size words
            data_input.append(words)
            # The labels will come from per sentence indexed 1
            data_label = torch.tensor(sentence[1], dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
        
        print("data_input middle sentence", len(data_input[int((window_size-1)/2)]))
        model.zero_grad()
        outputs = model(data_input)
        print("outputs", outputs.shape)
        loss_value = loss_function(outputs, data_label)
        loss_value.backward()
        optimizer.step()


if __name__ == '__main__':
    main()
