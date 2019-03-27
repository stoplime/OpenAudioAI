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
import tqdm

PATH = os.path.abspath(os.path.dirname(__file__))

val_split = 0.1
test_split = 0.1
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
                                print("\t",__val)
                                pass
                            # Selects the paragraphs
                            if __key == "paragraphs":
                                print()
                                if len(__val) > 0:
                                    for utterance in __val:
                                        for sentences in parse_utterance2sentences(utterance):
                                            # for sentence in sentences:
                                            print("------")
                                            print(sentences)
                                            print("------")
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

def parse_utterance2sentences(utterance):
    sentences = []
    sentence = ""
    for letter in utterance:
        end_sentence = False
        sentence += letter
        if len(sentence) > 1 and sentence[0] == " ":
            sentence = sentence[1:]
        if letter == "?" or letter == "!":
            end_sentence = True
        if letter == "…":
            end_sentence = True
            sentence = sentence[:-1]
        if len(sentence) > 2 and sentence[-3:] == "...":
            sentence = sentence.replace("...", ".")
            end_sentence = True
        if len(sentence) > 1 and sentence[-2:] == "--":
            end_sentence = True
        if len(sentence) > 2 and sentence[-2:] == ". " and sentence[-3].islower():
            sentence = sentence.replace(". ", ".")
            end_sentence = True

        if end_sentence and len(sentence) > 2:
            sentences.append(sentence.lower())
            sentence = ""
        elif end_sentence:
            sentence = ""
    if len(sentence) > 2:
        sentences.append(sentence.lower())
        sentence = ""
    return sentences

def data_iterator():
    with open(filePath) as f:
        rawData = json.load(f)

    # dialogue structure
    # List() - A dialogue per entry
        # List() - A single dialogue
            # Dictionary{} - The Utterance 
            #              - "speaker": "person name"
            #              - "sentences": List()
                # "sentences" List() - A sentence per entry 
    dialogues = []
    for key, val in tqdm.tqdm(rawData.items()):
        dialog = []
        for _key, _val in val.items():
            # Select transcript
            if _key == "transcript":
                for transcript in _val:
                    utterance = {}
                    utterance["sentences"] = []
                    # print(utterance["sentences"])
                    # Go through the list of transcripts
                    for __key, __val in transcript.items():
                        # Selects the speaker
                        if __key == "speaker":
                            utterance["speaker"] = __val
                            # print(__val)
                        # Selects the paragraphs
                        if __key == "paragraphs":
                            for ut in __val:
                                for sentences in parse_utterance2sentences(ut):
                                    # print(type(sentences))
                                    # print(utterance["sentences"])
                                    utterance["sentences"].append(sentences)
                                    # print(__val[0])
                    dialog.append(utterance)
        dialogues.append(dialog)
    return dialogues

def dialogues2listSentences(dialogues):
    speakers = {}
    sentences = []
    for dialog in dialogues:
        dialog_sentences = []
        for utterance in dialog:
            if utterance["speaker"] not in list(speakers.keys()):
                speakers[utterance["speaker"]] = len(speakers)
            for sentence in utterance["sentences"]:
                dialog_sentences.append(sentence + " %$* " + str(speakers[utterance["speaker"]]))
        sentences.append(dialog_sentences)
    # print(speakers)
    return sentences, speakers

def splits_sentence_dialogs2files(sentences):
    file_groups = []
    for dialog_sentences in sentences:
        dialog_group = []
        splits = even_spliter(len(dialog_sentences))
        split_sum = 0
        for split in splits:
            dialog_group.append(dialog_sentences[split_sum:split_sum+split])
            split_sum += split
        file_groups.append(dialog_group)
    return file_groups

def file_split(file_groups):
    groups = len(file_groups)
    # print(groups)
    # exit()

    val_split_value = int(groups * val_split)
    test_split_value = int(groups * test_split)

    train_count = 0
    val_count = 0
    test_count = 0

    make_train_val_test_split_folders()

    for i, group in tqdm.tqdm(enumerate(file_groups), total=groups):
        if i < test_split_value:
            for file_data in group:
                # write the file_data into a file
                with open(os.path.join(outputFolders["test"], "test_"+str(test_count)+".txt"), 'w') as f:
                    test_count += 1
                    for sentence in file_data:
                        f.write(sentence+"\n")
        elif i < test_split_value + val_split_value:
            for file_data in group:
                # write the file_data into a file
                with open(os.path.join(outputFolders["val"], "val_"+str(val_count)+".txt"), 'w') as f:
                    val_count += 1
                    for sentence in file_data:
                        f.write(sentence+"\n")
        else:
            for file_data in group:
                # write the file_data into a file
                with open(os.path.join(outputFolders["train"], "train_"+str(train_count)+".txt"), 'w') as f:
                    train_count += 1
                    for sentence in file_data:
                        f.write(sentence+"\n")
    print("done")

def main():
    # test()
    # pass
    # test_utterance = "Apparently the debate continues even as you cast your votes. And, and after—after we tabulate, I know your question, it’s a good one, and we will, we will pose it, I wanna thank the debaters and the audience, uh, for their participation. And we have to take care of some housekeeping here, the next Intelligence Squared debate will be on Tuesday, February 12th, here at Asia Society and Museum. The motion to be debated then is, “America should be the world’s policeman.” It will be moderated by “60 Minutes” correspondent Morley Safer. The panelists for the next debate are, for the motion, Senior Fellow for National Security Studies at the Council on Foreign Relations Max Boot, Professor and Director of the American Foreign Policy Program at Johns Hopkins University, Michael Mandelbaum, and author and director of the Centre for Social Cohesion, Douglas The Rosenkranz Foundation - Intelligence Squared US Debate “Performance Enhancing Drugs in Competitive Sports” Murray. Against the motion, president and founder of Eurasia Group Ian Bremer, President and CEO of the Henry L. Stimson Center, Ellen Laipson, and Matthew Parris, writer for the Times of London and broadcaster for the BBC. An edited version of tonight’s Intelligence Squared debate can be heard locally on WNYC-AM 820, on Sunday, January 27th, at 8 p.m. These debates are also heard on more than 90 NPR stations across the country, please check your local NPR member station listings for the dates and times of broadcast, outside New York City. Copies of Dick Pound’s books, Inside Dope: How Drugs Are the Biggest Threat to Sport, Why You Should Care and What Should Be Done About Them, and Inside the Olympics: A Behind the Scenes Look at the Politics, the Scandals, and the Glory of the Games, are on sale upstairs in the lobby, you can also purchase DVD’s from previous debates here tonight, or from the Intelligence Squared website. And now, very, very briefly, should there be two tiers of sports, Olympics or otherwise, one for those who use, one for those who don’t, one on this side, one on that side, 10 seconds. Should there be two tiers?"
    # sentences = parse_utterance2sentences(test_utterance)
    # print(len(sentences))
    # for sentence in sentences:
    #     print(sentence)
    #     print("------")
    data = data_iterator()
    sentences, speakers = dialogues2listSentences(data)
    file_groups = splits_sentence_dialogs2files(sentences)
    file_split(file_groups)

if __name__ == "__main__":
    main()