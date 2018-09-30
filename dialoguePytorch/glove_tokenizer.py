from tqdm import tqdm
import mmap


class glove_tokenizer():
    def __init__(self, glove_path=None):
        if glove_path == None:
            self.glove_path = "/home/stoplime/workspace/audiobook/gloveData/glove.6B.200d.txt"
        else:
            self.glove_path = glove_path
        self.tokenizer = {}
        self.import_glove()
    
    def get_num_lines(self):
        print("loading glove data...")
        fp = open(self.glove_path, "r+")
        buf = mmap.mmap(fp.fileno(), 0)
        lines = 0
        while buf.readline():
            lines += 1
        return lines

    def import_glove(self):
        for line in tqdm(open(self.glove_path), total=self.get_num_lines()):
            splits = line.split(' ')
            word = splits[0].strip()
            string_vector = splits[1:]
            vector = []
            for vec in string_vector:
                try:
                    float_vec = float(vec)
                    vector.append(float_vec)
                except:
                    print("Error: glove vector import cant convert to float:", vec)
            self.tokenizer[word] = vector

    def tokenize(self, word):
        return self.tokenizer[word]

def main():
    glove = glove_tokenizer()
    print("the", glove.tokenize('the'))

if __name__ == '__main__':
    main()