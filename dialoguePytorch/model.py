import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ABHUE(nn.Module):
    ''' Attention-Based Heirarchical Utterance Embedding
    '''
    def __init__(self):
        super(ABHUE, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.utterance_size = 200
        self.hidden_size = 200
        self.batch = 8
        self.context_lstm = nn.LSTM(input_size=self.utterance_size, hidden_size=self.hidden_size, batch_first=True)
        self.main_lstm = nn.LSTM(input_size=self.utterance_size, hidden_size=self.hidden_size, batch_first=True)

        self.prev_lstm = nn.LSTM(input_size=self.utterance_size, hidden_size=self.hidden_size, batch_first=True)
        self.post_lstm = nn.LSTM(input_size=self.utterance_size, hidden_size=self.hidden_size, batch_first=True)

        self.fc = nn.Linear(self.hidden_size*2, self.utterance_size)

    def create_hidden(self, length):
        return (torch.randn(1, 1, length).to(self.device), torch.randn(1, 1, length).to(self.device))

    def forward(self, utterances):
        '''
            # utterances: [sentence, word, direction*layers, batch, embedding]
            utterances: [[embedding] of len words] of len sentences
        '''
        hidden_shape = utterances[0][0].shape[2]
        # print("hidden_shape", hidden_shape)
        sentence_embedding = []
        # self.main_lstm.zero_grad()
        # self.context_lstm.zero_grad()
        for i, sentence in enumerate(utterances):
            hidden = self.create_hidden(hidden_shape)
            # hidden = torch.tensor.randn(1, 1, hidden_shape)
            if i == ((len(utterances) - 1) / 2):
                for word in sentence:
                    out, hidden = self.main_lstm(word, hidden)
            else:
                for word in sentence:
                    # print("word", word.shape)
                    # print("hidden", hidden[0].shape)
                    out, hidden = self.context_lstm(word, hidden)
                    # print("out", out.shape)
                    # print("hidden 2", hidden.shape)
            sentence_embedding.append(out)

        hidden = self.create_hidden(hidden_shape)
        # self.prev_lstm.zero_grad()
        for i, s_embed in enumerate(sentence_embedding):
            # print("s_embed", s_embed.shape)
            prev_out, hidden = self.prev_lstm(s_embed, hidden)
            if i == ((len(sentence_embedding) - 1) / 2):
                break

        hidden = self.create_hidden(hidden_shape)
        # self.post_lstm.zero_grad()
        for i, s_embed in reversed(list(enumerate(sentence_embedding))):
            post_out, hidden = self.post_lstm(s_embed, hidden)
            if i == ((len(sentence_embedding) - 1) / 2):
                break

        # print("prev_out", prev_out.shape)
        # print("post_out", post_out.shape)
        feature_vec = torch.squeeze(torch.cat((prev_out, post_out), 2))
        # print("feature_vec", feature_vec.shape)
        prediction = self.fc(feature_vec)

        return prediction