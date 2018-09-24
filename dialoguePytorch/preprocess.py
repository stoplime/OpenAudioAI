

import nltk
from glove_tokenizer import glove_tokenizer

class PreProcess(object):
    def __init__(self, glove_path=None):
        self.once = True
        self.glove = glove_tokenizer(glove_path)

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

    def createEmbeddingDict(self):
        pass

def main():
    dataPath = "/home/stoplime/workspace/audiobook/OpenAudioAI/data/train/train_0"

    preprocessor = PreProcess()
    data = preprocessor.parseData(dataPath)

if __name__ == '__main__':
    main()