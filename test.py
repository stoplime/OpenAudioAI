# from tensorflow.python.client import device_lib
# import tensorflow as tf

# def get_available_gpus():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos]# """ if x.device_type == 'GPU' """]

# print(get_available_gpus())
# print(tf.test.gpu_device_name())

import json
import os
import sys
import pprint

pp = pprint.PrettyPrinter(indent=4)

Join = os.path.join
PATH = os.path.abspath(os.path.dirname(__file__))

def main():
    file1 = Join(PATH, "data/CNN_transcript/cnnt/trancript.cnn.com.TRANSCRIPTS.1612.01.cnnt.01.html.json")
    with open(file1) as jsonFile:
        jsonData = json.load(jsonFile)

        for i, (key, value) in enumerate(jsonData.items()):
            print(key)
            if key == "sections":
                # pass
                pp.pprint(value)
        # pp.pprint(jsonData)

if __name__ == '__main__':
    main()