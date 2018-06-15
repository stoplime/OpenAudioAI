import nltk
import copy
import json
import os
from html.parser import HTMLParser
from tqdm import tqdm

class Sentence():
    def __init__(self, _sentId, _text):
        self.sentId = _sentId
        self.text = _text


class Section():
    def __init__(self):
        self.secId = 0
        self.speaker = 'UNK'
        self.sentences = []


class Transcript():
    def __init__(self):
        self.sectionCount = 0
        self.sentenceCount = 0
        self.docId = 'UNK'
        self.title = 'UNK'
        self.subTitle = 'UNK'
        self.time = 'UNK'
        self.noteByCnn = 'UNK'
        self.url = 'UNK'
        self.noteByCreator = 'Zhao Meng, Peking University, zhaomeng.pku@outlook.com. Sentence segmented by nltk.' \
                             ' Sections splited by BR'

        # expected to be list of speakers
        self.speakers = []

        # expected to be list of sections
        self.sections = []

        self.section2Id = {}
        self.sentence2Id = {}
        self.speaker2Id = {}


class InsidePoliticsParser(HTMLParser):

    # file must be an opened file
    def __init__(self, _file):
        HTMLParser.__init__(self)
        self.file = _file
        self.content = []
        self.record_cnnTransStoryHead = False
        self.record_cnnTransSubHead = False
        self.record_cnnBodyText = 0
        self.transcript = Transcript()

    # clean all the content in the parenthesis
    def clean(self, data):
        data = data.strip()

        #if data.find('been impressed in what') != -1:
         #   print("debug")

        if data.find('(') == -1 and data.find('[') == -1:
            return data

        clean_data = ''

        append = True
        for c in data:
            if c == '(':
                append = False
                continue
            if c == ')':
                append = True
                continue
            if append == True:
                clean_data += c

        clean_data2 = ''


        append = True
        for c in clean_data:
            if c == '[':
                append = False
                continue
            if c == ']':
                append = True
                continue
            if append == True:
                clean_data2 += c

        clean_data2 = clean_data2.strip()
        if clean_data2.find('(') != -1:
            print(clean_data2)

        return clean_data2

    def construct_content(self, data):
        data = self.clean(data)
        if data.find('[') != -1:
            print(data)
        if len(data.strip()) > 0:
            self.content.append(data.strip())

    def find_end(self, sec):
        for idx, c in enumerate(sec):
            if c.isupper() == False:
                return idx
        return 1

    def process_content(self):
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        speaker_id = 0
        sentence_id = 0
        speaker = 'UNK'

        for section_id, sec  in enumerate(self.content):
        # check if this is the start of one person's utterances
            section = Section()
            idx = 1
            # if we have a new speaker
            if sec[0].isupper():
                idx = sec.find(':')
                if idx > 1:
                    speaker = sec[0:idx]
                    sec = sec[idx + 1:].strip()
                    first_appear = speaker.find(',')
                    if first_appear != -1:
                        speaker = speaker[0:first_appear]
                        splits = speaker.split()
                        speaker = splits[-1]

                    if speaker not in self.transcript.speakers:
                        self.transcript.speaker2Id[speaker] = speaker_id
                        self.transcript.speakers.append(speaker)
                        speaker_id += 1


            section.speaker = speaker
            section.secId = section_id

            self.transcript.section2Id[section] = section_id

            sents = sent_detector.tokenize(sec.strip())

            for sent in sents:
                sentence = Sentence(sentence_id, sent)
                sentence_id += 1
                self.transcript.sentence2Id[sentence] = sentence_id
                section.sentences.append(sentence)
            self.transcript.sections.append(section)
        self.transcript.sectionCount = len(self.transcript.sections)
        sentence_id += 1
        self.transcript.sentenceCount = sentence_id



    # attrs are list of tuples [(attr_name, attr_value), ...]
    def handle_starttag(self, tag, attrs):
        if len(attrs) == 1 and tag == 'p' and attrs[0][0] == 'class':
                if attrs[0][1] == 'cnnTransStoryHead':
                    self.record_cnnTransStoryHead = True
                if attrs[0][1] == 'cnnTransSubHead':
                    self.record_cnnTransSubHead = True
                if attrs[0][1] == 'cnnBodyText':
                    self.record_cnnBodyText += 1
                    assert self.record_cnnBodyText <= 4
        pass

    def handle_endtag(self, tag):
        if tag == 'p' and self.record_cnnBodyText == 3:
            self.record_cnnBodyText = 4
            self.process_content()
        pass

    def handle_data(self, data):
        if self.record_cnnTransStoryHead == True:
            self.transcript.title = data
            self.record_cnnTransStoryHead = False

        if self.record_cnnTransSubHead == True:
            self.transcript.subTitle = data
            self.record_cnnTransSubHead = False

        # 1: time
        # 2: cnn note
        # 3: body
        if self.record_cnnBodyText == 1:
            if len(data.strip()) > 2:
                self.transcript.time = data
        elif self.record_cnnBodyText == 2:
            if len(data.strip()) > 2:
                self.transcript.noteByCnn = data
        elif self.record_cnnBodyText == 3:
            if len(data.strip()) > 2:
                self.construct_content(data)


def get_url(file_name):
    idx = file_name.find('TRANSCRIPT')
    url = 'transcripts.cnn.com/' + file_name[idx:]

    return url


def write2disk(out_file_name, transcript):
    out_file = open(out_file_name, 'w')

    trans = {}

    trans['sectionCount'] = transcript.sectionCount
    trans['sentenceCount'] = transcript.sentenceCount
    trans['docId'] = transcript.docId
    trans['title'] = transcript.title
    trans['subTitle'] = transcript.subTitle
    trans['time'] = transcript.time
    trans['noteByCnn'] = transcript.noteByCnn
    trans['url'] = transcript.url
    trans['noteByCreator'] = transcript.noteByCreator

    trans['speakers'] = transcript.speakers

    trans['sections'] = []

    for section in transcript.sections:
        sec = {}
        sec['secId'] = section.secId
        sec['speaker'] = section.speaker
        sec['sentences'] = []
        for sentence in section.sentences:
            sent = {}
            sent['sentId'] = sentence.sentId
            sent['text'] = sentence.text
            sec['sentences'].append(sent)
        trans['sections'].append(sec)

    out_file.write(json.dumps(trans))

    out_file.close()

def process_ip(in_file_name, out_file_name):
    in_file = open(in_file_name, 'r')

    data = in_file.read()

    url = get_url(in_file_name)

    parser = InsidePoliticsParser(in_file)
    parser.feed(data)

    parser.transcript.url = url
    parser.transcript.docId = out_file_name

    write2disk(out_file_name, parser.transcript)

    in_file.close()


def process_file_name(in_directory, out_directory):
    dates_dirs = os.listdir(in_directory)

    for dates_dir in tqdm(dates_dirs):
        num_dirs = os.listdir(in_directory + dates_dir)
        for num_dir in num_dirs:
            file_names = os.listdir(in_directory + dates_dir + '/' + num_dir)
            for file_name in file_names:
                full_name = in_directory + dates_dir + '/' + num_dir + '/' + file_name
                splits = full_name.split('/')
                out_file_name = out_directory + 'trancript.cnn.com.'+ splits[-4] + '.' + splits[-3] + '.' + splits[-2] + '.' + splits[-1] + '.json'
                process_ip(full_name, out_file_name)


def main():
    in_ip_directory = './data/insidepolitics/TRANSCRIPTS/'
    out_ip_directory = './ip/'

    in_rs_directory = './data/reliablesources/transcripts.cnn.com/TRANSCRIPTS/'
    out_rs_directory = './rs/'

    in_smer_directory = './data/smer/transcripts.cnn.com/TRANSCRIPTS/'
    out_smer_directory = './smer/'

    in_sotu_directory = './data/sotu/transcripts.cnn.com/TRANSCRIPTS/'
    out_sotu_directory = './sotu/'

    in_cnnt_directory = './data/cnnt/transcripts.cnn.com/TRANSCRIPTS/'
    out_cnnt_directory = './cnnt/'


    process_file_name(in_directory=in_ip_directory, out_directory=out_ip_directory)
    process_file_name(in_directory=in_rs_directory, out_directory=out_rs_directory)
    process_file_name(in_directory=in_smer_directory, out_directory=out_smer_directory)
    process_file_name(in_directory=in_sotu_directory, out_directory=out_sotu_directory)
    process_file_name(in_directory=in_cnnt_directory, out_directory=out_cnnt_directory)

if __name__ == '__main__':
    main()