


class PreProcess(object):
    def __init__(self):
        pass

    def parseData(self, dataFile):
        ''' Parse the data to a list of sentence and label
            Params:
            ------
            dataFile: string
                Path to the preprocessed data.
            ------
            Returns: List(sentence)
                sentence: List[embeddings, label]
        '''
        data = []
        with open(dataFile, 'r') as file:
            lines = file.readlines()
            for line in lines:
                splits = line.split('%$*')
                embeddings = nltk.word_tokenize(splits[0].strip())
                sentenceEmbedding = []
                for embed in embeddings:
                    sentenceEmbedding.append(embed)
                data.append([sentenceEmbedding, splits[1].strip()])
        return data

    def createEmbeddingDict(self):
        pass

    