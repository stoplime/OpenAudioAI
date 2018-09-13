'''
A recurrent neural network for speaker change detection
static attention
'''
import tensorflow as tf

# import torch

# class SpeakerModel(nn.Module):
#     def __init__(self, num_classes=10, points_per_class=2, preprocess=True):
#         super(SpeakerModel, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(32, 64, 5)
#         self.conv3 = nn.Conv2d(64, 128, 3)
#         self.conv4 = nn.Conv2d(128, 256, 3)
#         self.fc1 = nn.Linear(256, 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, num_classes*3*points_per_class)

#         self.num_classes = num_classes
#         self.points_per_class = points_per_class
#         self.preprocess = preprocess

#     def forward(self, x):
#         if self.preprocess:
#             if len(x.shape) < 4:
#                 x = x.reshape([1]+list(x.shape))
#             x = x/127.5 - 1
#             # x = nn.Upsample(size=(32, 32))(x)
#         x = self.pool(F.leaky_relu(self.conv1(x)))
#         x = F.leaky_relu(self.conv2(x))
#         x = self.pool(F.leaky_relu(self.conv3(x)))
#         x = F.leaky_relu(self.conv4(x))
#         x = F.avg_pool2d(x, kernel_size=x.size()[2:], stride=(1, 1))

#         # global average
#         x = x.view(-1, 256)
#         x = F.leaky_relu(self.fc1(x))
#         x = F.leaky_relu(self.fc2(x))
#         x = F.sigmoid(self.fc3(x))
#         x = x.view(-1, self.num_classes, self.points_per_class, 3)
#         return x

class RecurrentModel:
    def __init__(self, args, textData):
        self.args = args
        self.textData = textData

        self.input_utterances = None
        self.labels = None
        self.dropOutRate = None
        self.embedded = None
        self.length = None
        self.predictions = None
        self.padded_length = self.args.batchSize + 2*(self.args.uttContextSize-1) + 1
        self.loss = None
        # self.learning_rate = None

        self.correct = None
        self.accuracy = None
        self.correct_predictions = None

        # this is for debug
        self.vectors = None
        self.batchSize = None
        self.optOp = None

        self.twinLength = []

        for i in range(self.args.batchSize + self.args.uttContextSize):
            self.twinLength.append(self.args.uttContextSize)

        self.buildNetWork()

    def buildNetWork(self):
        # last_outputs is of shape [fake_batch_size, self.args.wordUnits]
        with tf.variable_scope('word'):
            last_outputs = self._buildWordNetwork()
            last_outputs_squeezed = tf.squeeze(last_outputs)
        with tf.variable_scope('utterance'):
            context_vectors = self._buildUttNetwork(last_outputs_squeezed)
        # use context_vectors_squeezed as inputs to softmax
        # shape: [true_batch_size, self.args.uttUnits]

        with tf.name_scope('output'):
            weights = tf.Variable(tf.truncated_normal([self.args.uttUnits * 4, self.args.numClasses], stddev=0.5),
                                  name='weights')
            biases = tf.Variable(tf.truncated_normal([self.args.numClasses], stddev=0.5), name='biases')
            logits = tf.add(tf.matmul(context_vectors, weights), biases, name='logits')

            self.out = tf.slice(logits, begin=[0, 0], size=[self.batchSize, self.args.numClasses], name='softmax_truncated')

            self.predictions = tf.argmax(self.out, axis=1, name='predictions')
        with tf.name_scope('loss'):
            # note: since we have the sparse version, we don't need to have one-hot labels
            loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.out, labels=self.labels, name='loss_')
            self.loss = tf.reduce_mean(loss_, name='loss')
            self._variableSummaries(self.loss)
            self.vectors = self.loss
        with tf.name_scope('evaluation'):
            self.correct_predictions = tf.equal(tf.cast(self.predictions, tf.int32), self.labels)
            self.correct = tf.reduce_sum(tf.cast(self.correct_predictions, tf.int32), name='numCorrect')
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32), name='accuracy')
            self._variableSummaries(self.accuracy)

        with tf.name_scope('backpropagation'):
            opt = tf.train.AdamOptimizer(learning_rate=self.args.learningRate, beta1=0.9, beta2=0.999,
                                               epsilon=1e-08)
            self.optOp = opt.minimize(self.loss)


    def _buildWordNetwork(self):
        with tf.name_scope('placeholders'):
            # in this implementation, we still have fake batch size in the input_utterances
            # fake batch_size = (self.args.uttContextSize - 1) + batch_size
            self.input_utterances = tf.placeholder(tf.int32, [None, self.args.maxLength], name='input_utterances')
            self.length = tf.placeholder(tf.int32, [self.padded_length], name='sequence_length')
            # self.labels use true batch_size
            self.labels = tf.placeholder(tf.int32, [None], name='labels')
            self.dropOutRate = tf.placeholder(tf.float32, (), name='dropOut')
            self.batchSize = tf.placeholder(tf.int32, (), name='true_batch_size')
            # self.learning_rate = tf.placeholder(tf.float32, shape=[])

        with tf.name_scope('embeddingLayer'):
            # whether or not to use the pretrained embeddings
            if self.args.preEmbedding == False:
                embeddings = tf.Variable(
                    tf.truncated_normal([self.textData.getVocabularySize()-1, self.args.embeddingSize], stddev=0.5),
                    name='embeddings')
            else:
                embeddings = tf.Variable(self.textData.preTrainedEmbedding, name='embedding')
            # note: for <pad>, embedding should be all zeros
            zero_embedding = tf.Variable(tf.zeros([1, self.args.embeddingSize]), name='padEmbedding', trainable=False)
            embeddings = tf.concat(values=[zero_embedding, embeddings], axis=0)
            #self._variableSummaries(embeddings)
            # self.embedded is a 3-dimentional matrix of shape [fake_batch_size, maxLength, embeddingSize]
            self.embedded = tf.nn.embedding_lookup(embeddings, self.input_utterances)

        with tf.name_scope('sentence_encoder'):
            with tf.name_scope('cell'):
                cell = tf.contrib.rnn.LSTMCell(num_units=self.args.wordUnits, state_is_tuple=True)
                cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.dropOutRate,
                                                         output_keep_prob=self.dropOutRate)
                multiCell = tf.contrib.rnn.MultiRNNCell([cell]*self.args.wordLayers, state_is_tuple=True)

            outputs, states = tf.nn.dynamic_rnn(cell=multiCell, inputs=self.embedded,
                                                sequence_length=self.length, dtype=tf.float32)

            # note: use tf.gather_nd to replace this function when gradient is implemented
            def last_relevant_old(output, length):
                #note, this implementation is unreasonably slow, deprecated
                batch_size = tf.shape(output)[0]
                max_length = tf.shape(output)[1]
                out_size = int(output.get_shape()[2])
                index = tf.range(0, batch_size) * max_length + (length - 1)
                flat = tf.reshape(output, [-1, out_size])
                relevant = tf.gather(flat, index, name='last_outputs')
                return relevant

            def last_relevant(output, length):
                slices = []
                for idx, l in enumerate(tf.unstack(length)):
                    last = tf.slice(output, begin=[idx, l - 1, 0], size=[1, 1, self.args.wordUnits])
                    slices.append(last)

                lasts = tf.concat(slices, 0)
                return lasts
            # fake_batch_size = self.args.batchSize + self.args.uttContextSize * 2 - 1
            # last_outputs is of shape [fake_batch_size, self.args.wordUnits]
            last_outputs = last_relevant(outputs, self.length)
            return last_outputs


    def _buildUttNetwork(self, last_outputs):
        with tf.name_scope('prepare_inputs'):
            utt_inputs = []
            utt_inputs_reverse = []
            for i in range(self.args.batchSize + self.args.uttContextSize):
                utt_input = tf.slice(last_outputs, begin=[i, 0], size=[self.args.uttContextSize, self.args.wordUnits])
                utt_input_reverse = tf.reverse(tensor=utt_input, axis=[0], name='utt_input_reverse')
                utt_inputs.append(utt_input)
                utt_inputs_reverse.append(utt_input_reverse)
            # shape: [self.args.batchSize+self.args.uttContextSize, self.args.uttContextSize, self.args.wordUnits]
            utt_inputs_pack = tf.stack(utt_inputs)
            utt_inputs_reverse_pack = tf.stack(utt_inputs_reverse)

        with tf.variable_scope('context_encoder_up'):
            with tf.name_scope('cell'):
                cell = tf.contrib.rnn.LSTMCell(num_units=self.args.uttUnits, state_is_tuple=True)
                cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.dropOutRate,
                                                         output_keep_prob=self.dropOutRate)
                multiCell = tf.contrib.rnn.MultiRNNCell([cell]*self.args.uttLayers, state_is_tuple=True)

            # outputs shape: [self.args.batchSize + self.args.uttContextSize, self.args.uttContextSize, self.args.uttUnits]
            outputs, states = tf.nn.dynamic_rnn(cell=multiCell, inputs=utt_inputs_pack,
                                                sequence_length=self.twinLength, dtype=tf.float32)

        with tf.variable_scope('context_encoder_down'):
            with tf.name_scope('cell'):
                cell2 = tf.contrib.rnn.LSTMCell(num_units=self.args.uttUnits, state_is_tuple=True)
                cell2 = tf.contrib.rnn.DropoutWrapper(cell2, input_keep_prob=self.dropOutRate,
                                                     output_keep_prob=self.dropOutRate)
                multiCell2 = tf.contrib.rnn.MultiRNNCell([cell2] * self.args.uttLayers, state_is_tuple=True)

            # the reverse context and the original context are encoded by the same RNN
            #tf.get_variable_scope().reuse_variables()



            # outputs shape: [self.args.batchSize + self.args.uttContextSize, self.args.uttContextSize(reversed), self.args.uttUnits]
            outputs_reverse, states_reverse = tf.nn.dynamic_rnn(cell=multiCell2, inputs=utt_inputs_reverse_pack,
                                                sequence_length=self.twinLength, dtype=tf.float32)
            # for debugging, check if the variables are reused
        #vb = tf.global_variables()
        # variables for attention
        def build_attention(to_context, from_last):

            weights_y = tf.Variable(tf.truncated_normal(shape=[self.args.uttUnits, self.args.uttUnits], stddev=0.5),
                                    name='weights_y')

            weights_h = tf.Variable(tf.truncated_normal(shape=[self.args.uttUnits, self.args.uttUnits], stddev=0.5),
                                    name='weights_h')

            e_l = tf.Variable(tf.ones(shape=[self.args.uttContextSize,]), name='e_l', trainable=False)

            weights_soft = tf.Variable(tf.truncated_normal([self.args.uttUnits, 1], stddev=0.5), name='weights_soft')

            # let outputs attend corresponding outputs_reverse

            # M_0 = W_y * Y

            d0, d1, d2 = to_context.get_shape()
            to_context_reshape = tf.reshape(to_context, shape=[-1, self.args.uttUnits],
                                         name='to_context_reshape')


            to_context_reshape = tf.reshape(to_context, shape=[-1, self.args.uttUnits],
                                         name='to_context_reshape')
            matrix_M_0 = tf.matmul(to_context_reshape, weights_y, name='M_0_intermediate')

            matrix_M_0 = tf.reshape(matrix_M_0, shape=[-1, self.args.uttContextSize, self.args.uttUnits], name='M_0')

            # M_1 = W_h * h_N * e_l
            matrix_M_1 = tf.matmul(from_last, weights_h, name='M_1_intermediate')

            matrix_M_1 = tf.expand_dims(matrix_M_1, axis=-1)
            e_l = tf.expand_dims(e_l, axis=0)


            matrix_M_1_reshape = tf.reshape(matrix_M_1, shape=[-1, 1])
            matrix_M_1 = tf.matmul(matrix_M_1_reshape, e_l)
            matrix_M_1 = tf.reshape(matrix_M_1, shape=[-1, self.args.uttContextSize, self.args.uttUnits])

            matrix = tf.add(matrix_M_0, matrix_M_1, name='M')

            # M = tanh(W_y * Y + W_h * h_N * e_l)
            matrix_activated = tf.tanh(matrix, name='matrix_tanh')

            # alpha = softmax(w_s * M)
            matrix_activated_reshape = tf.reshape(matrix_activated, shape=[-1, self.args.uttUnits])
            alpha = tf.matmul(matrix_activated_reshape, weights_soft)
            alpha = tf.reshape(alpha, shape=[-1, self.args.uttContextSize])
            alpha = tf.nn.softmax(logits=alpha, name='alpha')

            # r = Y * alpha
            # shape: [self.args.batchSize+self.args.uttContextSize, 1, self.args.uttContextSize]
            alpha_reshape = tf.reshape(alpha, shape=[-1, 1, self.args.uttContextSize], name='alpha_reshape')
            attention = tf.matmul(alpha_reshape, to_context, name='attention_up_down')
            # shape: [self.args.batchSize+self.args.uttContextSize, self.args.uttUnits]
            attention_squeeze = tf.squeeze(attention, name='attention_up_down_squeeze')

            return attention_squeeze

        with tf.name_scope('attention_prepare'):

            # -------------------------------------- attention ---------------------------------------------
            # for i in range(self.args.batchSize + self.args.uttContextSize):
            #   outputs[i, self.args.uttContextSize-1, ...] = attention(outputs_reverse[i+self.args.uttContextSize, ..., ...])
            #   outputs_reverse[i, self.args.uttContext-1, ...] = attention(outputs[i-self.args.uttContextSize, ..., ...])

            # prepare the attention vector
            outputs_last = tf.slice(outputs, begin=[0, self.args.uttContextSize - 1, 0],
                                    size=[self.args.batchSize + self.args.uttContextSize, 1, self.args.uttUnits])
            outputs_reverse_last = tf.slice(outputs_reverse, begin=[0, self.args.uttContextSize - 1, 0],
                                            size=[self.args.batchSize + self.args.uttContextSize, 1,
                                                  self.args.uttUnits])

            # shape: [self.args.batchSize + self.args.uttContextSize, self.args.uttUnits]
            outputs_last_squeeze = tf.squeeze(outputs_last)
            outputs_reverse_last_squeeze = tf.squeeze(outputs_reverse_last)


            up_context = tf.slice(outputs, begin=[0, 0, 0],
                                  size=[self.args.batchSize, self.args.uttContextSize, self.args.uttUnits], name='up')
            up_last = tf.slice(outputs_last_squeeze, begin=[0, 0],
                               size=[self.args.batchSize, self.args.uttUnits], name='up_last')

            down_context = tf.slice(outputs_reverse, begin=[self.args.uttContextSize, 0, 0],
                                    size=[self.args.batchSize, self.args.uttContextSize, self.args.uttUnits], name='down')
            down_last = tf.slice(outputs_reverse_last_squeeze, begin=[self.args.uttContextSize, 0],
                                 size=[self.args.batchSize, self.args.uttUnits], name='down_last')

        with tf.name_scope('attention_up_down'):
            # shape: [self.args.batchSize]
            # `from` attend `to`
            attention_up2down = build_attention(to_context=down_context, from_last=up_last)

        with tf.name_scope('attention_down_up'):
            # shape: [self.args.batchSize]
            attention_down2up = build_attention(to_context=up_context, from_last=down_last)

        with tf.name_scope('concate'):

            # [0, self.args.batchSize]
            outputs_last_squeeze_0 = tf.slice(input_=outputs_last_squeeze, begin=[0, 0],
                                              size=[self.args.batchSize, self.args.uttUnits], name='slice_0')
            # [self.args.uttContextSize, self.uttContextSize + self.args.batchSize]
            outputs_last_squeeze_1 = tf.slice(input_=outputs_reverse_last_squeeze, begin=[self.args.uttContextSize, 0],
                                              size=[self.args.batchSize, self.args.uttUnits], name='slice_1')

            up_concats = tf.concat(values=[outputs_last_squeeze_0, attention_up2down], axis=1, name='up')
            down_concats = tf.concat(values=[outputs_last_squeeze_1, attention_down2up], axis=1, name='down')

            # shape: [self.args.batchSize, self.args.uttUnits*2]
            output_concats = tf.concat(values=[up_concats, down_concats], axis=1,
                                       name='outputs_concat')

        return output_concats


    def step(self, batch, test = False, lr=None):
        feed_dict = {}
        ops = None
        def pad(sequence, length):
            padding = []
            for i in range(self.args.maxLength):
                padding.append(0)
            while len(sequence) < self.args.batchSize + 2*(self.args.uttContextSize-1) + 1 :
                sequence.append(padding)
            while len(length) < self.args.batchSize + 2 * (self.args.uttContextSize - 1) + 1:
                length.append(2)
            return sequence, length

        if len(batch.sequence) < self.args.batchSize + 2*(self.args.uttContextSize-1) + 1:
            batch.sequence, batch.length = pad(batch.sequence, batch.length)
        if len(batch.length) < self.args.batchSize + 2*(self.args.uttContextSize-1) + 1:
            batch.sequence, batch.length = pad(batch.sequence, batch.length)

        assert len(batch.sequence) == len(batch.length)

        feed_dict[self.input_utterances] = batch.sequence
        feed_dict[self.labels] = batch.labels
        feed_dict[self.batchSize] = len(batch.labels)
        feed_dict[self.length] = batch.length
        if not test:
            feed_dict[self.dropOutRate] = self.args.dropOut
            ops = (self.optOp, self.loss, self.correct, self.predictions, self.vectors)
        else:
            feed_dict[self.dropOutRate] = 1.0
            ops = (self.correct, self.predictions)
        
        if lr != None:
            pass

        return ops, feed_dict

    def _variableSummaries(self, var):
        '''
        currently do not need any summaries in RNN implementation
        :param var:
        :return:
        '''
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
