
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
    model = model.to(device)

    optimizer = optim.Adam(model.parameters())
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    exp_lr_scheduler.step() # update lr scheduler epoch
    model.train()

    # TODO: Actually decide on a good loss function
    loss = nn.MSELoss()

    for data in preprocessor.parseData(dataPath):
        data_input = torch.tensor(data[0]).to(device)
        data_label = torch.tensor(data[1], dtype=torch.int32).to(device)
        
        outputs = model(inputs)
        loss_value = loss(outputs, data_label)
        loss_value.backward()
        optimizer.step()


if __name__ == '__main__':
    main()
