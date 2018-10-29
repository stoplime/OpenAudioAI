
import nltk
from glove_tokenizer import glove_tokenizer
import torch

class PreProcess(object):
    def __init__(self, window_size, glove_path=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.once = True
        self.glove = glove_tokenizer(glove_path)
        self.window_size = window_size
        self.sliding_window = []

    def parseData(self, dataFile):
        ''' Parse the data to a list of sentence and label
            Params:
            ------
            dataFile: string
                Path to the preprocessed data.
            ------
            Returns: List(sentence)
                sentence: List[embeddings, label]
                    embeddings: List(word embeddings)
        '''
        with open(dataFile, 'r') as file:
            lines = file.readlines()
            for line in lines:
                splits = line.split('%$*')
                words = nltk.word_tokenize(splits[0].strip())
                sentenceEmbedding = []
                # print(embeddings)
                for word in words:
                    embedding = self.glove.tokenize(word)
                    sentenceEmbedding.append(embedding)
                yield [sentenceEmbedding, int(splits[1].strip())]
    
    def create_sliding_window(self, sentence):
        self.sliding_window.append(sentence)
        if len(self.sliding_window) < self.window_size:
            return False
        elif len(self.sliding_window) > self.window_size:
            self.sliding_window.pop(0)
        return True

    def tensorfy(self):
        ''' Converts the sliding window to a list of tensors the model can use.
            Returns the list of tensors as well as the label of the middle sentence.
        '''
        data_input = []
        for i, sentence in enumerate(self.sliding_window):
            words = []
            for word in sentence[0]:
                # Converts the word imbedding into a pytorch tensor per word
                words.append(torch.tensor(word).to(self.device).unsqueeze(0).unsqueeze(0))
            # data_input will have a size of sliding window for the sentence and in each sentence there will be a list of embedings of size words
            data_input.append(words)

            # if middle sentence
            if i == (len(self.sliding_window) - 1) / 2:
                # The label will come from per sentence[1] => (sentence, label)
                data_label = sentence[1]
        return data_input, data_label


def main():
    dataPath = "/home/stoplime/workspace/audiobook/OpenAudioAI/data/train/train_0"

    preprocessor = PreProcess()
    data = preprocessor.parseData(dataPath)

if __name__ == '__main__':
    main()