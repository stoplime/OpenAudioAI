import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ABHUE(nn.Module):
    ''' Attention-Based Heirarchical Utterance Embedding
    '''
    def __init__(self, recurrent_model="lstm", dropout=0, stack_size=1, dev=torch.device("cpu")):
        super(ABHUE, self).__init__()
        self.device = dev
        self.input_size = 200
        self.hidden_size = 200
        self.stack_size = stack_size
        self.isLSTM = recurrent_model == "lstm"
        if self.isLSTM:
            self.context_rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
            self.target_rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)

            self.prev_rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True, dropout=dropout, num_layers=stack_size)
            self.post_rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True, dropout=dropout, num_layers=stack_size)
        else:
            self.context_rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
            self.target_rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)

            self.prev_rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True, dropout=dropout, num_layers=stack_size)
            self.post_rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True, dropout=dropout, num_layers=stack_size)
            
        self.fc = nn.Linear(self.hidden_size*2, self.input_size)

        self.context_rnn = self.context_rnn.to(self.device)
        self.target_rnn = self.target_rnn.to(self.device)

        self.prev_rnn = self.prev_rnn.to(self.device)
        self.post_rnn = self.post_rnn.to(self.device)

        self.fc = self.fc.to(self.device)

    def create_hidden(self, length, stack=False):
        if self.isLSTM:
            if not stack:
                hidden = (torch.randn(1, 1, length).to(self.device), torch.randn(1, 1, length).to(self.device))
            else:
                hidden = (torch.randn(self.stack_size, 1, length).to(self.device), torch.randn(self.stack_size, 1, length).to(self.device))
        else:
            if not stack:
                hidden = torch.randn(1, 1, length).to(self.device)
            else:
                hidden = torch.randn(self.stack_size, 1, length).to(self.device)
        return hidden

    def reset_gradients(self):
        self.context_rnn.zero_grad()
        self.target_rnn.zero_grad()
        self.prev_rnn.zero_grad()
        self.post_rnn.zero_grad()
        self.fc.zero_grad()

    # @profile
    def forward(self, sentences):
        '''
            # sentences: [sentence, word, direction*layers, batch, embedding]
            sentences: [[embedding] of len words] of len sentences
        '''
        sentence_embedding = []
        for i, sentence in enumerate(sentences):
            hidden = self.create_hidden(self.hidden_size)
            if i == ((len(sentences) - 1) / 2):
                for word in sentence:
                    out, hidden = self.target_rnn(word, hidden)
            else:
                for word in sentence:
                    out, hidden = self.context_rnn(word, hidden)
            del hidden
            sentence_embedding.append(out)

        hidden = self.create_hidden(self.hidden_size, stack=True)
        for i, s_embed in enumerate(sentence_embedding):
            prev_out, hidden = self.prev_rnn(s_embed, hidden)
            if i == ((len(sentence_embedding) - 1) / 2):
                break
        # hidden = hidden.detach()
        del hidden

        hidden = self.create_hidden(self.hidden_size, stack=True)
        for i, s_embed in reversed(list(enumerate(sentence_embedding))):
            post_out, hidden = self.post_rnn(s_embed, hidden)
            if i == ((len(sentence_embedding) - 1) / 2):
                break
        # hidden = hidden.detach()
        del hidden

        feature_vec = torch.squeeze(torch.cat((prev_out, post_out), 2))
        prediction = self.fc(feature_vec)
        del feature_vec

        return prediction

class GlobalModule(nn.Module):
    ''' The Global Module of the Attention-Based Heirarchical Utterance Embedding
    '''
    def __init__(self, recurrent_model="lstm", dropout=0, stack_size=1, dev=torch.device("cpu")):
        super(GlobalModule, self).__init__()
        self.device = dev
        self.local_prediction_size = 200
        self.hidden_size = 200
        self.stack_size = stack_size
        self.isLSTM = recurrent_model == "lstm"
        if self.isLSTM:
            self.global_rnn = nn.LSTM(input_size=self.local_prediction_size, hidden_size=self.hidden_size, batch_first=True, dropout=dropout, num_layers=stack_size)
        else:
            self.global_rnn = nn.GRU(input_size=self.local_prediction_size, hidden_size=self.hidden_size, batch_first=True, dropout=dropout, num_layers=stack_size)

        self.global_rnn = self.global_rnn.to(self.device)

        self.hidden = None

    def create_hidden(self, length, stack=False):
        if self.isLSTM:
            if not stack:
                hidden = (torch.randn(1, 1, length).to(self.device), torch.randn(1, 1, length).to(self.device))
            else:
                hidden = (torch.randn(self.stack_size, 1, length).to(self.device), torch.randn(self.stack_size, 1, length).to(self.device))
        else:
            if not stack:
                hidden = torch.randn(1, 1, length).to(self.device)
            else:
                hidden = torch.randn(self.stack_size, 1, length).to(self.device)
        return hidden

    def reset_gradients(self):
        self.global_rnn.zero_grad()
        if type(self.hidden) == tuple:
            self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        else:
            self.hidden = self.hidden.detach()

    def forward(self, local_prediction):
        '''
            local_prediction: tensor(200)
        '''
        local_prediction = local_prediction.unsqueeze(0).unsqueeze(0)
        self.hidden = self.create_hidden(self.hidden_size, stack=True)
        global_pred, self.hidden = self.global_rnn(local_prediction, self.hidden)

        return global_pred.squeeze()