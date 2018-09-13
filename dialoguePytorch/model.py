import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ABHUE(nn.Module):
    ''' Attention-Based Heirarchical Utterance Embedding
    '''
    def __init__(self):
        super(ABHUE, self).__init__()
        self.utternace_size = 200
        self.hidden_size = 200
        self.context_lstm = nn.LSTM(input_size=self.utterance_size, hidden_size=self.hidden_size, batch_first=True)
        self.main_lstm = nn.LSTM(input_size=self.utterance_size, hidden_size=self.hidden_size, batch_first=True)

        self.prev_lstm = nn.LSTM(input_size=self.utterance_size, hidden_size=self.hidden_size, batch_first=True)
        self.post_lstm = nn.LSTM(input_size=self.utterance_size, hidden_size=self.hidden_size, batch_first=True)

        self.fc = nn.Linear(self.hidden_size*2, self.utternace_size)

    def forward(self, utterances):
        '''
            utterances: [sentence, word, direction*layers, batch, embedding]
        '''
        hidden_shape = utterances[0].shape[1:]
        sentence_embedding = []
        for i, sentence in enumerate(utterances):
            hidden = torch.randn(hidden_shape)
            if i == ((len(utterances) - 1) / 2):
                for word in sentence:
                    out, hidden = self.main_lstm(word, hidden)
            else:
                for word in sentence:
                    out, hidden = self.context_lstm(word, hidden)
            sentence_embedding.append(out)

        hidden = torch.randn(hidden_shape)
        for i, s_embed in enumerate(sentence_embedding):
            prev_out, hidden = self.prev_lstm(s_embed, hidden)
            if i == ((len(sentence_embedding) - 1) / 2):
                break

        hidden = torch.randn(hidden_shape)
        for i, s_embed in reversed(list(enumerate(sentence_embedding))):
            post_out, hidden = self.post_lstm(s_embed, hidden)
            if i == ((len(sentence_embedding) - 1) / 2):
                break

        feature_vec = torch.cat((prev_out, post_out), 0)
        prediction = self.fc(feature_vec)

        return prediction