# Convert the Iq2 data from json to the data type we need
# sentence that ends with a pucntuation. %$* id
# Punctuations ".", "?", "!", "--"
# Remove some puctuations such as "..."
# Also remove empty sentences or sentences with just punctuation

# Run through all the dialogue
# Each file will contain at most one dialogue
# A single dialogue can split into multiple files if it gets too big

import json
import os
import copy

PATH = os.path.abspath(os.path.dirname(__file__))

filePath = os.path.join(PATH, "iq2_data_release.json")
dataName = "iq2"
outputFolders = {
    "train": os.path.join(PATH, dataName + "_train"),
    "val": os.path.join(PATH, dataName + "_val"),
    "test": os.path.join(PATH, dataName + "_test")
}

def make_train_val_test_split_folders():
    os.makedirs(outputFolders["train"], exist_ok=True)
    os.makedirs(outputFolders["val"], exist_ok=True)
    os.makedirs(outputFolders["test"], exist_ok=True)

def test():
    with open(filePath) as f:
        rawData = json.load(f)
    
    count = 0
    keys = []
    # Raw Data
    for key, val in rawData.items():
        # print(key)
        count += 1
        # Select PerformanceEnhancingDrugs Debate
        if key == "PerformanceEnhancingDrugs-011508":
            for _key, _val in val.items():
                # Select transcript
                if _key == "transcript":
                    for transcript in _val:
                        # Go through the list of transcripts
                        for __key, __val in transcript.items():
                            if __key not in keys:
                                keys.append(__key)
                            # Selects the speaker
                            if __key == "speaker":
                                print(__val)
                            # Selects the paragraphs
                            if __key == "paragraphs":
                                print(__val[0])
    print(count)
    print(keys)
    # print(rawData)

def even_spliter(total, inRange=[250, 500]):
    """ Splits the total into chunks where they are the size inRange
    ------
    total: The value that needs to be split
    ------
    inRange: List(min, max)
        min: The minimum value the split size needs to be
        max: The maximum split size before it needs to split more
            The algorithm is biased to the max split and will start to
            evenly split at the max first.
    """
    splitRange = copy.deepcopy(inRange)
    while splitRange[1] >= splitRange[0]:
        if (total + splitRange[1]-1) % splitRange[1] >= splitRange[0]-1:
            splitList = [splitRange[1] for i in range(total//splitRange[1])]
            if total % splitRange[1] != 0:
                splitList.append(total % splitRange[1])
            return splitList
        else:
            splitRange[1] -= 1
    return [total]

def main():
    # test()
    print(even_spliter(50))
    print(even_spliter(500))
    print(even_spliter(501))
    print(even_spliter(502))
    print(even_spliter(1000))
    print(even_spliter(1001))
    print(even_spliter(999))
    print(even_spliter(1250))

if __name__ == "__main__":
    main()