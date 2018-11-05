
import random
import json
import pprint

pp = pprint.PrettyPrinter(indent=4)

def create_random_sets(width, height):
    _set = []
    for i in range(width):
        _set.append([])
        for j in range(height):
            _set[i].append( random.randint(0, 10) )
    return _set

def generate_sets_to_json():
    test_set = []
    for i in range(10):
        test_set.append(([], create_random_sets(3, 3)))
    with open("sets.json", 'w') as file:
        json.dump(test_set, file, indent=4)

def load_sets_from_json():
    with open("sets.json", 'r') as file:
        sets = json.load(file)
    pp.pprint(sets)
    # sets.append(([], create_random_sets(4, 4)))
    # with open("sets.json", 'w') as file:
    #     json.dump(sets, file, indent=4)

def DataLabel2ListMatrix(data, labels):
    # convert the clusters into matricies
    total_labels = []
    for lable in labels:
        if label not in total_labels:
            total_labels.append(label)
    dataArray = [[0 for j in range(len(total_labels))] for i in range(len(total_labels))]

    for entry in data:
        entry_label = entry[0]
        for entry_set in entry[1]:
            if expression:
                pass
    return dataArray

def bestLabels(data, labels):
    ''' 
        Params
        ------
        data: kmeans clusters
        ------
        labels: List(tuple(sentence id, label))
    '''
    # List(custers)
    # clusters: List(tuple(sentence id, point))

    # pred_clusters = []
    # for i, cluster in enumerate(data):
    #     pred_clusters.append([])
    #     for point in cluster.points:
    #         pred_clusters[i].append(point)

    pred_clusters = []

    return 25

def main():
    load_sets_from_json()

if __name__ == '__main__':
    main()