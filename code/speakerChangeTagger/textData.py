import os
from tqdm import tqdm
import pickle as p
import nltk
from collections import defaultdict
import random
import numpy as np

class Batch:
    '''
    0. should have extra utterances for padding, hence dimension-0 of utterances and labels
    are different. Specifically, utterances.d_0 = labels.d_0 + 2 * (windowSize - 1)
    1. all the seqs of a batch should come from one single show
    2. test and val also have batch
    '''
    def __init__(self):
        # list of utterances, each utterance is a list [[utt_0], [utt_1], ...]
        # all utterances are padded to maxLength using <pad>
        self.sequence = []
        self.labels = []
        # length of each sample in the batch, useful for RNN
        self.length = []
        # useful for debugging
        self.id = -1
class Sample:
    def __init__(self):
        # raw sentence, use nltk.word_tokenize later
        self.utterance = ''
        # 0: not a change point
        # 1: is a change point
        self.label = -1
        # sequence: convert utterance to sequence of word ids
        self.sequence = []
        # true length of this sample, max is self.args.maxLength
        self.length = -1
        # seq of words
        self.words = []

class Show:
    def __init__(self):
        self.samples = []
        self.batches = []


class TextData:
    '''
    TextData consists of train data, val data and test data
    each data consists of several Show object
    each Show object consists of several several sample object
    each sample object consists of a sentence text and a label
    '''
    def __init__(self, args):
        self.args = args
        self.trainShows = []
        self.valShows = []
        self.testShows = []
        self.preTrainedEmbedding = []
        # batches for train, val, and test
        # note: currently useless, since we want to keep all the batches of a show together during training
        self.trainBatches = []
        self.valBatches = []
        self.textBatches = []

        self.word2id = {}
        self.id2word = {}
        self.word2cnt = defaultdict(int)

        self.trainCnt = -1
        self.valCnt = -1
        self.testCnt = -1

        self.batchCnt = 0

        # for sentences shorter than self.args.maxLength, pad them
        # for words whose occurances are less than 10, pad them to <unknown>
        self.PAD_WORD = '<pad>'
        self.UNK_WORD = '<unknown>'
        self.pad = self.getWordId(self.PAD_WORD)
        self.unk = self.getWordId(self.UNK_WORD)


        self.createData()
        self.createDic()

        if self.args.preEmbedding == True:
            self.createEmbedding()

        cntLength = self.createSeq()
        self.avglenth  = cntLength*1.0/(self.trainCnt+self.testCnt+self.valCnt)

        print('average length = {}'.format(cntLength*1.0/(self.trainCnt+self.valCnt+self.testCnt)))

        self.createBatches()
        print('number of words = {}'.format(len(self.word2id)))

    def createEmbedding(self):
        with open(self.args.embeddingFile, 'rb') as embeddingFile:
            # a dictionary
            # key: word
            # value: 200-d embeddings
            word2embedding = p.load(embeddingFile)

        unknown = np.random.normal(scale=0.5, size=[self.args.embeddingSize]).tolist()

        unk_cnt = 0
        cnt = 0
        for id in range(len(self.id2word)):
            word = self.id2word[id]
            if word == self.UNK_WORD:
                self.preTrainedEmbedding.append(unknown)
                cnt += 1
            # if we have that embedding
            elif word == self.PAD_WORD:
                # note: add pad word later, should be untrainable
                continue
            elif word in word2embedding.keys():
                # note: some words do not have pretrained embedding
                # note: hence, they are pointed to self.unk
                # note: their position should be replace by the words to come
                embedding = word2embedding[word]
                self.preTrainedEmbedding.append(embedding)
                self.word2id[word] = len(self.preTrainedEmbedding)
            # if we don't have that embedding, make the word unknown
            # note: word2id is violated!
            else:
                #note, point the word to <unknown>
                self.word2id[word] = self.unk
                unk_cnt += 1
        # note: pad should be added to the head of the array later
        self.preTrainedEmbedding = np.array(self.preTrainedEmbedding).astype(np.float32)
        print('Total {} unknown words'.format(unk_cnt))
        assert cnt == 1


    def _create(self, directory, tag):
        '''
        :param directory: the data directory
        :param tag: should be 'train', 'test' or 'val'
        :return:
        '''
        print('creating {} data'.format(tag))

        def createSamples(fileName):
            file = open(fileName, 'r')

            lines = file.readlines()
            samples = []
            for line in lines:
                sample = Sample()
                splits = line.split('%$*')
                #print(line)
                sample.utterance = splits[0].strip()

                words = nltk.word_tokenize(sample.utterance)

                for word in words:
                    # note: do not generate wordId here, cnt the occurances first
                    self.word2cnt[word] += 1
                    sample.words.append(word)
                sample.label = int(splits[1].strip())
                samples.append(sample)

            file.close()
            return samples

        sampleCnt = 0
        fileNames = os.listdir(directory)
        shows = []
        #for fileName in fileNames:
        for fileName in tqdm(fileNames):
            if tag == 'train':
                fileName = os.path.join(self.args.trainData, fileName)
            elif tag == 'val':
                fileName = os.path.join(self.args.valData, fileName)
            elif tag == 'test':
                fileName = os.path.join(self.args.testData, fileName)
            show = Show()
            samples = createSamples(fileName)
            sampleCnt += len(samples)
            show.samples = samples
            shows.append(show)

        if tag == 'train':
            self.trainShows = shows
        elif tag == 'val':
            self.valShows = shows
        elif tag == 'test':
            self.testShows = shows
        return sampleCnt


    def createDic(self):
        '''
        create dictionaries
        word2id, id2word
        :return:
        '''
        cntLength = 0
        print('Creating sequnces of words')
        for shows in [self.trainShows, self.testShows, self.valShows]:
            for show in tqdm(shows):
                for sample in show.samples:
                    for word in sample.words:
                        if self.word2cnt[word] < self.args.minOccur:
                            wordId = self.unk
                            self.word2id[word] = self.unk
                        else:
                            wordId = self.getWordId(word)


    def createSeq(self):
        '''
        create sequence of wordIds, and pad the sequences to maxLength
        :return:
        '''
        cntLength = 0
        print('Creating sequnces of words')
        for shows in [self.trainShows, self.testShows, self.valShows]:
            for show in tqdm(shows):
                for sample in show.samples:
                    for word in sample.words:
                        wordId = self.word2id[word]
                        sample.sequence.append(wordId)

                    cntLength += len(sample.sequence)
                    #print(len(sample.sequence))
                    # record true length of each sample
                    if len(sample.sequence) > self.args.maxLength:
                        sample.length = self.args.maxLength
                    else:
                        sample.length = len(sample.sequence)

                    while len(sample.sequence) < self.args.maxLength:
                        sample.sequence.append(self.pad)
                    # restrict the max length of sequences
                    sample.sequence = sample.sequence[0:self.args.maxLength]
        return cntLength

    def createData(self):
        self.trainCnt = self._create(self.args.trainData, 'train')
        self.valCnt = self._create(self.args.valData, 'val')
        self.testCnt = self._create(self.args.testData, 'test')

        print('{} training utterances, {} validation utterances, {} test utterances'.format(
            self.trainCnt, self.valCnt, self.testCnt
        ))

    def getWordId(self, word):
        '''
        Create wordId for a word
        :param word:
        :return: wordId
        '''
        wordId = self.word2id.get(word, -1)
        #self.word2cnt[word] += 1

        if wordId == -1:
            wordId = len(self.id2word)
            self.word2id[word] = wordId
            self.id2word[wordId] = word
        return wordId

    def getWordFromID(self, wordId):

        word = self.id2word.get(wordId, "-1")

        if word == "-1":
            word = "PAD"
        return word
    
    def covertIdFeed2Words(self, feed):
        passage = []
        print(type(feed))
        for i, tensor in enumerate(feed):
            # print(tensor.name)
            if (tensor.name).find("input_utterances") != -1:
                sample = feed[tensor]
                # print("sample", sample)
                for line in sample:
                    sentence = []
                    hasSomething = False
                    for wordId in line:
                        # print(wordId)
                        word = self.getWordFromID(wordId)
                        if word != "<pad>":
                            sentence.append(word)
                            hasSomething = True
                        else:
                            sentence.append("")
                    if hasSomething:
                        passage.append(sentence)
                    
        return passage

    def getVocabularySize(self):
        '''
        get vocab size
        :return:
        '''

        size = len(self.word2id)
        return size

    def shuffle(self, shows):
        random.shuffle(shows)

    def getShowBatch(self, show):
        '''
        extract batch from a show
        :param show:
        :return: list of batch objects
        '''

        batches_sample = []
        batches = []
        samples = show.samples
        numSamples = len(samples)

        for i in range(0, numSamples, self.args.batchSize):
            batch_sample = samples[i:min(i+self.args.batchSize, numSamples)]
            batches_sample.append(batch_sample)

        pad_seq = []
        for i in range(self.args.maxLength):
            pad_seq.append(self.pad)

        pad_seqs = []
        for i in range(self.args.uttContextSize-1):
            pad_seqs.append(pad_seq)

        pad_length = []
        # the length of the padding sentence is useless, but do not set it to zero
        for i in range(self.args.uttContextSize-1):
            pad_length.append(2)

        for idx, batch_sample in enumerate(batches_sample):
            # this is the first batch of a show
            # pad_sentence * (self.uttContextSize - 1) + setences + next_batch_sentence * (self.uttContextSize)
            batch = Batch()
            batch.id = self.batchCnt
            self.batchCnt += 1
            if idx == 0:
                # extend with padding seqs first
                batch.sequence.extend(pad_seqs)
                batch.length.extend(pad_length)
            else:
                # extend with last batch_sample
                last_batch_sample = batches_sample[idx-1]
                samples_for_padding = last_batch_sample[-(self.args.uttContextSize - 1):]
                for sample in samples_for_padding:
                    batch.sequence.append(sample.sequence)
                    batch.length.append(sample.length)
            # then extend with true seqs
            for sample in batch_sample:
                batch.sequence.append(sample.sequence)
                batch.labels.append(sample.label)
                batch.length.append(sample.length)

            # this is the last batch of a show, pad with pad seqs
            if idx == len(batches_sample)-1:
                batch.sequence.extend(pad_seqs)
                batch.sequence.append(pad_seq)
                batch.length.extend(pad_length)
                batch.length.append(2)
            else:
                # not the last batch of a show, pad with next batch
                next_batch_sample = batches_sample[idx+1]
                # note restriction: self.args.uttContextSize <= self.args.batchSize
                # fixme: self.args.uttContextSize may be too long for next batch
                samples_for_padding = next_batch_sample[0:(self.args.uttContextSize)]
                for sample in samples_for_padding:
                    batch.sequence.append(sample.sequence)
                    batch.length.append(sample.length)

                numShorts = (self.args.uttContextSize) - len(samples_for_padding)
                while numShorts > 0:
                    batch.sequence.append(pad_seq)
                    numShorts -= 1
            # check
            if len(batch.labels) != len(batch.sequence) - 2*(self.args.uttContextSize - 1) - 1:
                print('Wrong padding! batch id = {}'.format(batch.id))
                assert False

            batches.append(batch)

        return batches

    def createBatches(self):
        '''
        create batches for training
        :return:
        '''
        print('Creating batches for train/val/show')

        for show in self.trainShows:
            show_batches = self.getShowBatch(show)
            show.batches = show_batches
            self.trainBatches.append(show_batches)

        for show in self.testShows:
            show_batches = self.getShowBatch(show)
            show.batches = show_batches
            self.textBatches.append(show_batches)

        for show in self.valShows:
            show_batches = self.getShowBatch(show)
            show.batches = show_batches
            self.valBatches.append(show_batches)

    def getBatches(self, tag='train'):
        '''

        :param tag: train or val or test
        :return:
        '''

        assert tag == 'train' or tag == 'val' or tag == 'test'

        batches = []
        if tag == 'train':
            self.shuffle(self.trainShows)
            for show in self.trainShows:
                batches.extend(show.batches)

        if tag == 'test':
            #self.shuffle(self.testShows)
            for show in self.testShows:
                batches.extend(show.batches)

        if tag == 'val':
            #self.shuffle(self.valShows)
            for show in self.valShows:
                batches.extend(show.batches)

        return batches
