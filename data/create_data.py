import os
import json
import numpy as np

transcript_directory = 'CNN_transcript'

class Show:
    def __init__(self):
        self.sents = []
        self.file_name = ''


class Sentence:
    '''
    label = 0: sentence is not boundary
    label = 1: sentence is a boundary
    '''
    def __init__(self):
        self.utterance = ''
        self.label = -1
        self.speaker = ''


def process(file_name):
    file = open(file_name, 'r')
    data = file.read()
    transcript = json.loads(data)

    # list of Sentence object
    sents = []

    for section in transcript['sections']:
        # each section is a dictionary
        for sentence in section['sentences']:
            # each sentence is also a dictionary
            sent = Sentence()
            sent.utterance = sentence['text'].lower().replace('\n', '').replace('\r', '')
            sent.speaker = section['speaker'].lower()
            sents.append(sent)

    for idx, sent in enumerate(sents):
        # the last sentence in a show is definitely a boundary
        if idx == len(sents) - 1:
            sent.label = 1
            break
        cur_speaker = sent.speaker
        next_speaker = sents[idx+1].speaker

        if cur_speaker != next_speaker:
            sent.label = 1
        else:
            sent.label = 0

    show = Show()

    show.file_name = file_name
    show.sents = sents

    file.close()

    return show


def count(shows):
    cnt_0 = 0
    cnt_1 = 0
    for show in shows:
        for sent in show.sents:
            if sent.label == 1:
                cnt_1 += 1
            else:
                cnt_0 += 1

    return cnt_1, cnt_0


def create(shows, directory):
    for idx, show in enumerate(shows):
        file = open(os.path.join(directory, directory+'_'+str(idx)), 'w')
        for sent in show.sents:
            file.write(sent.utterance + ' %$* ' + str(sent.label) + '\n')
        file.close()

def main():
    dirs = ['cnnt', 'ip', 'rs', 'smer', 'sotu']
    shows = []

    for dir_ in dirs:
        dir_ = os.path.join(transcript_directory, dir_)
        file_names = os.listdir(dir_)
        for file_name in file_names:
            file_name = os.path.join(dir_, file_name)
            show = process(file_name)
            shows.append(show)

    cnt_0 = 0
    cnt_1 = 0
    for show in shows:
        for sent in show.sents:
            if sent.label == 1:
                cnt_1 += 1
            else:
                cnt_0 += 1

    print('{} positive examples, {} total examples, {}'
          .format(cnt_1, cnt_0+cnt_1, cnt_1*1.0/(cnt_1+cnt_0)))
    np.random.shuffle(shows)

    numTrain = int(len(shows)*0.8)
    numVal = int(len(shows)*0.1)

    train = shows[:numTrain]
    val = shows[numTrain:numTrain+numVal]
    test = shows[numTrain+numVal:]

    pos, neg = count(train)
    print('Train: {} positive examples, {} total examples, {}'
          .format(pos, neg+pos, pos*1.0/(pos+neg)))

    pos, neg = count(val)
    print('Validation {} positive examples, {} total examples, {}'
          .format(pos, neg+pos, pos*1.0/(pos+neg)))

    pos, neg = count(test)
    print('Test {} positive examples, {} total examples, {}'
          .format(pos, neg+pos, pos*1.0/(pos+neg)))

    create(train, 'train')
    create(val, 'val')
    create(test, 'test')


if __name__ == '__main__':
    main()
