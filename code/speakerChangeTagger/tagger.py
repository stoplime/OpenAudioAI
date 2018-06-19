import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import argparse
from speakerChangeTagger.textData import TextData
from speakerChangeTagger.RecurrentModel import RecurrentModel
from sklearn.metrics import confusion_matrix
import pickle as p

Join = os.path.join
Path = os.path.dirname(os.path.abspath(__file__))

class Tagger:
    def __init__(self):
        self.args = None

        self.textData = None
        self.model = None

        self.globalStep = 0

        self.summaryWriter = None
        self.outFile = None
        self.mergedSummary = None
        # tensorflow main session
        self.sess = None

    @staticmethod
    def parseArgs(args):

        parser = argparse.ArgumentParser()

        parser.add_argument('--resultDir', type=str, default='result', help='result directory')

        # data location
        dataArgs = parser.add_argument_group('Dataset options')

        dataArgs.add_argument('--trainData', type=str, default='data/train', help='training data location')
        dataArgs.add_argument('--valData', type=str, default='data/val', help='validation data location')
        dataArgs.add_argument('--testData', type=str, default='data/test', help='test data location')
        dataArgs.add_argument('--dataDir', type=str, default='data', help='dataset directory, save pkl here')
        dataArgs.add_argument('--datasetName', type=str, default='dataset', help='a TextData object')
        dataArgs.add_argument('--numClasses', type=int, default=2, help='number of classes for current dataset')
        dataArgs.add_argument('--summaryDir', type=str, default='summaries', help='directory of summaries')
        dataArgs.add_argument('--minOccur', type=int, default=0, help='min occurances for a word')

        # network options
        nnArgs = parser.add_argument_group('Network options')
        nnArgs.add_argument('--maxLength', type=int, default=30, help='maximum length for one utterance, useful for padding')
        nnArgs.add_argument('--wordLayers', type=int, default=1, help='CNN window size for words')
        nnArgs.add_argument('--wordUnits', type=int, default=200, help='CNN size for words')
        nnArgs.add_argument('--uttContextSize', type=int, default=3, help='up and down context size for the 2nd CNN')
        nnArgs.add_argument('--uttLayers', type=int, default=1, help='CNN windows size for utterances')
        nnArgs.add_argument('--uttUnits', type=int, default=200, help='CNN size for utterance')
        nnArgs.add_argument('--embeddingSize', type=int, default=200, help='embedding size')

        # training options
        trainingArgs = parser.add_argument_group('Training options')
        trainingArgs.add_argument('--dropOut', type=float, default=0.9, help='dropout rate for CNN')
        trainingArgs.add_argument('--learningRate', type=float, default=0.0009, help='learning rate')
        trainingArgs.add_argument('--batchSize', type=int, default=100, help='batch size')
        ## do not add dropOut in the test mode!
        trainingArgs.add_argument('--test', type=bool, default=False, help='if in test mode')
        trainingArgs.add_argument('--epochs', type=int, default=120, help='training epochs')
        trainingArgs.add_argument('--device', type=str, default='/gpu:1', help='use the second GPU as default')
        trainingArgs.add_argument('--preEmbedding', type=bool, default=False, help='whether or not to use the pretrained embedding')
        trainingArgs.add_argument('--embeddingFile', type=str, default='embeddings/200d.pkl', help='pretrained embeddings')

        # evaluation options
        evalArgs = parser.add_argument_group('Evaluation options')
        evalArgs.add_argument('--evalModel', default=False, action='store_true', help='indicates for an evaluation')
        evalArgs.add_argument('--modelPath', type=str, default=Join(Path, 'saves', 'savedModel.ckpt'), help='trained model path')
        return parser.parse_args(args)


    def constructFileName(self):
        #TODO
        file_name = str(self.args.maxLength) + '_' + str(self.args.wordLayers) + '_' + str(self.args.wordUnits)
        file_name += '_' + str(self.args.uttContextSize) + '_' + str(self.args.uttLayers) + '_' + str(self.args.uttUnits) + '_'
        file_name += str(self.args.embeddingSize) + '_' + str(self.args.dropOut) + '_' + str(self.args.learningRate)
        file_name += '_' + str(self.args.batchSize)
        if self.args.preEmbedding:
            file_name += '_True'
        else:
            file_name += '_False'

        return file_name

    def constructDatasetName(self):
        #TODO
        suffix = '-' + str(self.args.maxLength) + '-' + str(self.args.uttContextSize) + '-' + str(self.args.batchSize)
        if self.args.preEmbedding:
            suffix += '_True'
        else:
            suffix += '_False'

        self.args.datasetName += suffix + '.pkl'

    def constructDir(self, base):
        #TODO
        directory = []
        directory.append(base)
        # maxlength
        super = []
        super.append('maxLen_' + str(self.args.maxLength))
        super.append('_wL_' + str(self.args.wordLayers))
        super.append('_wRNN_' + str(self.args.wordUnits))
        super.append('_uCon_' + str(self.args.uttContextSize))
        super.append('_uL_' + str(self.args.uttLayers))
        super.append('_uRNN_' + str(self.args.uttUnits))
        super.append('_e_' + str(self.args.embeddingSize))
        if self.args.preEmbedding:
            super.append('_True')
        else:
            super.append('_False')


        base_0 = ''.join(super)

        dir_0 = os.path.join(base, base_0)
        if not os.path.exists(dir_0):
            os.makedirs(dir_0)

        dir_1 = os.path.join(dir_0, 'lr_' + str(self.args.learningRate))
        if not os.path.exists(dir_1):
            os.makedirs(dir_1)

        dir_2 = os.path.join(dir_1, 'dropout_' + str(self.args.dropOut))
        if not os.path.exists(dir_2):
            os.makedirs(dir_2)
        return dir_2

    def writeInfo(self, file):
        file.write('maxLength = {}\n'.format(self.args.maxLength))
        file.write('wordLayers = {}\n'.format(self.args.wordLayers))
        file.write('wordUnits = {}\n'.format(self.args.wordUnits))
        file.write('uttLayers = {}\n'.format(self.args.uttLayers))
        file.write('uttUnits = {}\n'.format(self.args.uttUnits))
        file.write('uttContextSize = {}\n'.format(self.args.uttContextSize))
        file.write('embeddingSize = {}\n'.format(self.args.embeddingSize))

        file.write('dropOut = {}\n'.format(self.args.dropOut))
        file.write('learning rate = {}\n'.format(self.args.learningRate))
        file.write('batch size = {}\n'.format(self.args.batchSize))
        file.write('embedding file = {}\n'.format(self.args.embeddingSize))

        file.flush()

    def main(self, args=None):
        print('TensorFlow v{}'.format(tf.__version__))

        # initialize args
        self.args = self.parseArgs(args)

        self.outFile = self.constructFileName()

        # note: for padding
        assert self.args.uttContextSize < self.args.batchSize

        # load data if exists, else create the dataset
        self.constructDatasetName()
        datasetFileName = os.path.join(self.args.dataDir, self.args.datasetName)
        if not os.path.exists(datasetFileName):
            self.textData = TextData(self.args)
            with open(datasetFileName, 'wb') as datasetFile:
                p.dump(self.textData, datasetFile)
            print('dataset created and saved to {}'.format(datasetFileName))
        else:
            with open(datasetFileName, 'rb') as datasetFile:
                self.textData = p.load(datasetFile)
            print('dataset loaded from {}'.format(datasetFileName))


        # note: since dropOut is not implemented yet, currently we only have one model

        # default session
        sessConfig = tf.ConfigProto(allow_soft_placement=True)
        sessConfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sessConfig)

        # summary writer
        self.args.summaryDir = self.constructDir('summaries')
        with tf.device(self.args.device):
            self.model = RecurrentModel(self.args, self.textData)
            self.summaryWriter = tf.summary.FileWriter(self.args.summaryDir, self.sess.graph)
            self.mergedSummary = tf.summary.merge_all()
            init = tf.global_variables_initializer()
            # initialize all global variables
            self.sess.run(init)
            if self.args.evalModel:
                self.evalModel(self.sess)
            else:
                self.train(self.sess)

    def evalModel(self, sess):
        print("Start Evaluation of Model")

        self.saver = tf.train.Saver()

        self.saver.restore(sess, self.args.modelPath)

        out = open(os.path.join(self.args.resultDir, self.outFile), 'w', 1)
        out.write(self.outFile + '\n')
        self.writeInfo(out)

        trainBatches = self.textData.getBatches('train')
        for nextBatch in tqdm(trainBatches):
            ops, feed_dict = self.model.step(nextBatch)
            _, loss, correct, predictions, vec = sess.run(ops, feed_dict)

            print('ops: {}\nfeed_dict: {}'.format(ops, feed_dict))
            print('loss: {}\ncorrect: {}\npredictions: {}\nvec: {}'.format(loss, correct, predictions, vec))

            out.write('ops: {}\nfeed_dict: {}'.format(ops, feed_dict))
            out.write('loss: {}\ncorrect: {}\npredictions: {}\nvec: {}'.format(loss, correct, predictions, vec))
        out.close()

    def train(self, sess):
        '''
        training loop
        :param sess: default sess for Tagger
        :return:
        '''

        print('Start training')
        
        self.saver = tf.train.Saver()

        out = open(os.path.join(self.args.resultDir, self.outFile), 'w', 1)
        out.write(self.outFile + '\n')
        self.writeInfo(out)

        for e in range(self.args.epochs):
            # training
            trainBatches = self.textData.getBatches('train')
            totalTrainLoss = 0.0
            allTrainCorrect = 0.0
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            for nextBatch in tqdm(trainBatches):
            # note, for batches in the end of a show, the length is not enough
            #for nextBatch in trainBatches:
                self.globalStep += 1
                ops, feed_dict = self.model.step(nextBatch)

                # use vec to fetch some vector from the graph, just for debug
                _, loss, correct, predictions, vec = sess.run(ops, feed_dict)
                #print(vec[0])
                #print(softmax_input.shape())

                #self.summaryWriter.add_summary(batchSummary, self.globalStep)

                totalTrainLoss += loss
                allTrainCorrect += correct
                true_positive, true_negative, false_postive, false_negative \
                    = self.calculate_F1(predictions, nextBatch.labels)
                TP += true_positive
                TN += true_negative
                FP += false_postive
                FN += false_negative

            precision = (TP*1.0)/(TP+FP)
            recall = (TP*1.0)/(TP+FN)

            f1 = 2*precision*recall/(precision+recall)

            trainAcc = allTrainCorrect/self.textData.trainCnt
            valAcc, valF1, valP, valR = self.test(sess, tag='val')
            testAcc, testF1, testP, testR = self.test(sess, tag='test')

            tf.summary.scalar(name='trainF1', tensor=f1)
            tf.summary.scalar(name='valF1', tensor=valF1)
            tf.summary.scalar(name='testF1', tensor=testF1)
            tf.summary.scalar(name='trainLoss', tensor=totalTrainLoss)
            tf.summary.scalar(name='trainAcc', tensor=trainAcc)
            tf.summary.scalar(name='valAcc', tensor=valAcc)
            tf.summary.scalar(name='testAcc', tensor=testAcc)

            print('epoch = {}/{}, trainAcc = {}, trainLoss = {}, valAcc = {}, testAcc = {}'
                  .format(e+1, self.args.epochs, trainAcc, totalTrainLoss, valAcc, testAcc))
            print('trainF1 = {}, valF1 = {}, testF1 = {}'.format(f1, valF1, testF1))
            print('trainP = {}, trainR = {}, valP = {}, valR = {}, testP = {}, testR = {}'.format(precision, recall,
                                                                                                  valP, valR, testP, testR))

            out.write('epoch = {}/{}, trainAcc = {}, trainLoss = {}, valAcc = {}, testAcc = {}\n'
                  .format(e+1, self.args.epochs, trainAcc, totalTrainLoss, valAcc, testAcc))
            out.write('               trainF1 = {}, valF1 = {}, testF1 = {}\n'
                      .format(f1, valF1, testF1))
            out.write('               trainP = {}, trainR = {}, valP = {}, valR = {}, testP = {}, testR = {}\n'
                      .format(precision, recall, valP, valR, testP, testR))
            out.flush()
        if not os.path.exists(Join(Path, "saves")):
            os.mkdir(Join(Path, "saves"))
        self.saver.save(sess, Join(Path, "saves", "savedModel.ckpt"))
        out.close()


    def test(self, sess, tag='val'):
        self.args.test = True

        batches = self.textData.getBatches(tag)
        allCorrect = 0.0

        TP = 0
        TN = 0
        FP = 0
        FN = 0

        for idx, nextBatch in enumerate(batches):
            ops, feed_dict = self.model.step(nextBatch, test=True)
            correct, predictions = sess.run(ops, feed_dict)
            true_positive, true_negative, false_postive, false_negative \
                = self.calculate_F1(predictions, nextBatch.labels)

            allCorrect += correct
            TP += true_positive
            TN += true_negative
            FP += false_postive
            FN += false_negative
        try:
            if TP == 0 and FP == 0:
                f1 = -1.0
                precision = (TP * 1.0) / (TP + FP)
                recall = (TP * 1.0) / (TP + FN)
            else:
                precision = (TP * 1.0) / (TP + FP)
                recall = (TP * 1.0) / (TP + FN)

                f1 = 2 * precision * recall/(precision + recall)
        except:
            f1 = -2.0
            precision = 0.0
            recall = 0.0


        if tag == 'val':
            acc = allCorrect / self.textData.valCnt
        else:
            acc = allCorrect / self.textData.testCnt

        self.args.test = False

        return acc, f1, precision, recall


    def calculate_F1(self, predictions, labels):
        '''
        calculate TP, FP, FN
        :return:
        '''
        # sometimes predictions and labels are all zeros or all ones, we need *labels* to guide
        matrix = confusion_matrix(y_true=labels, y_pred=predictions, labels=[0,1])

        true_positive = matrix[1][1]
        false_postive = matrix[0][1]

        true_negative = matrix[0][0]
        false_negative = matrix[1][0]

        return true_positive, true_negative, false_postive, false_negative

