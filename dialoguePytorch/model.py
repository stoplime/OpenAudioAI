import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ABHUE(nn.Module):
    ''' Attention-Based Heirarchical Utterance Embedding
    '''
    def __init__(self, recurrent_model="lstm", dropout=0, stack_size=1):
        super(ABHUE, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    def forward(self, sentences):
        '''
            # sentences: [sentence, word, direction*layers, batch, embedding]
            sentences: [[embedding] of len words] of len sentences
        '''
        hidden_shape = sentences[0][0].shape[2]
        # print("hidden_shape", hidden_shape)
        sentence_embedding = []
        # self.target_rnn.zero_grad()
        # self.context_rnn.zero_grad()
        for i, sentence in enumerate(sentences):
            hidden = self.create_hidden(hidden_shape)
            # hidden = torch.tensor.randn(1, 1, hidden_shape)
            if i == ((len(sentences) - 1) / 2):
                for word in sentence:
                    out, hidden = self.target_rnn(word, hidden)
            else:
                for word in sentence:
                    # print("word", word.shape)
                    # print("hidden", hidden[0].shape)
                    out, hidden = self.context_rnn(word, hidden)
                    # print("out", out.shape)
                    # print("hidden 2", hidden.shape)
            sentence_embedding.append(out)

        hidden = self.create_hidden(hidden_shape, stack=True)
        # self.prev_rnn.zero_grad()
        for i, s_embed in enumerate(sentence_embedding):
            # print("s_embed", s_embed.shape)
            prev_out, hidden = self.prev_rnn(s_embed, hidden)
            if i == ((len(sentence_embedding) - 1) / 2):
                break

        hidden = self.create_hidden(hidden_shape, stack=True)
        # self.post_rnn.zero_grad()
        for i, s_embed in reversed(list(enumerate(sentence_embedding))):
            post_out, hidden = self.post_rnn(s_embed, hidden)
            if i == ((len(sentence_embedding) - 1) / 2):
                break

        # print("prev_out", prev_out.shape)
        # print("post_out", post_out.shape)
        feature_vec = torch.squeeze(torch.cat((prev_out, post_out), 2))
        # print("feature_vec", feature_vec.shape)
        prediction = self.fc(feature_vec)

        return prediction

class GlobalModule(nn.Module):
    ''' The Global Module of the Attention-Based Heirarchical Utterance Embedding
    '''
    def __init__(self, recurrent_model="lstm", dropout=0, stack_size=1):
        super(GlobalModule, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.local_prediction_size = 200
        self.hidden_size = 200
        self.stack_size = stack_size
        self.isLSTM = recurrent_model == "lstm"
        if self.isLSTM:
            self.global_rnn = nn.LSTM(input_size=self.local_prediction_size, hidden_size=self.hidden_size, batch_first=True, dropout=dropout, num_layers=stack_size)
        else:
            self.global_rnn = nn.GRU(input_size=self.local_prediction_size, hidden_size=self.hidden_size, batch_first=True, dropout=dropout, num_layers=stack_size)

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

    def forward(self, local_prediction):
        '''
            local_prediction: tensor(200)
        '''
        local_prediction = local_prediction.unsqueeze(0).unsqueeze(0)
        hidden = self.create_hidden(self.hidden_size, stack=True)
        global_pred, hidden = self.global_rnn(local_prediction, hidden)

        return global_pred.squeeze()