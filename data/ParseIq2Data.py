# Convert the Iq2 data from json to the data type we need
# sentence that ends with a pucntuation. %$* id
# Punctuations ".", "?", "!", "--"
# Remove some puctuations such as "..."
# Also remove empty sentences or sentences with just punctuation
import json
import os

PATH = os.path.abspath(os.path.dirname(__file__))

filePath = os.path.join(PATH, "iq2_data_release.json")

def main():
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

if __name__ == "__main__":
    main()