
import random
import json
import pprint
import numpy as np
from tqdm import tqdm
import time

pp = pprint.PrettyPrinter(indent=4)

def create_random_sets(width, height, max_range=10):
    _set = []
    for i in range(width):
        _set.append([])
        for j in range(height):
            _set[i].append( random.randint(0, max_range) )
    return _set

def create_random_sets_total(width, height, total_number):
    _set = [[0 for j in range(height)] for i in range(width)]
    for n in range(total_number):
        i = random.randint(0, width-1)
        j = random.randint(0, height-1)
        # print("[{}, {}]".format(i, j))
        # print(len(_set))
        # print(len(_set[0]))
        _set[i][j] += 1
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
    return sets

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

def GetGlobalMaxes(data):
    ''' 
        Params
        ------
        data: list(list(uint))
            The rows and columns of the cluster label matrix
            i: cluster
            j: label
    '''
    maxes = []
    max_value = 0
    for i, data_cluster in data:
        for j, data_cell in data_cluster:
            if data_cell > max_value:
                max_value = data_cell
    
    for i, data_cluster in data:
        for j, data_cell in data_cluster:
            if data_cell == max_value:
                maxes.append((i, j, data_cell))
    return maxes

# def arreq_in_list(myarr, list_arrays):
#     return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)

def Backtracking(data_matrix, id_matrix, black_list=None):
    ''' 
        Params
        ------
        data_matrix: int numpy array (n, n)
    '''
    if black_list == None:
        black_list = []
    
    # if arreq_in_list(id_matrix, black_list):
    #     return 0

    row = data_matrix.shape[0]
    col = data_matrix.shape[1]

    total_max = -1
    cell_max = -1
    for i in range(row):
        for j in range(col):
            if cell_max < data_matrix[i, j]:
                cell_max = data_matrix[i, j]
    
    for i in range(row):
        for j in range(col):
            # print(id_matrix[i, j], black_list)
            if id_matrix[i, j] in black_list:
                continue
            
            selectedCell = data_matrix[i, j]
            if selectedCell < cell_max:
                continue
            
            row_slice = np.delete(data_matrix, i, axis=0)
            sub_data_matrix = np.delete(row_slice, j, axis=1)

            row_id_slice = np.delete(id_matrix, i, axis=0)
            sub_id_matrix = np.delete(row_id_slice, j, axis=1)

            calcMax = selectedCell + Backtracking(sub_data_matrix, sub_id_matrix, black_list)
            if total_max < calcMax:
                total_max = calcMax
            else:
                # black_list.append(id_matrix[i, j])
                # black_list.append(sub_id_matrix)
                pass
    
    if total_max == -1:
        total_max = 0
    
    return total_max

# def MaxCells(data):
    # numpy_data = np.array(data)
    # print(numpy_data)
    # max_value = Backtracking()
    # isolate the max rows and collumns

def test_size():
    size = 10
    custom_set = create_random_sets_total(size, size, 32)
    numpy_data = np.array(custom_set)
    print(numpy_data)
    start = time.time()
    print(Backtracking(numpy_data, np.arange((size**2)).reshape(size, size)))
    end = time.time()
    print("Time:", end - start)

def test_accuracy():
    datasets = load_sets_from_json()
    for data in datasets:
        print(data)
        numpy_data = np.array(data[1])
        size = numpy_data.shape[0]
        # print(numpy_data)
        max_value = Backtracking(numpy_data, np.arange((size**2)).reshape(size, size))
        print(max_value)

def get_standard_deviation(dataset):
    mean = sum(dataset)/len(dataset)
    dataSqrMeans = []
    for data in dataset:
        meanDiff = data - mean
        meanDiffSqr = meanDiff**2
        dataSqrMeans.append(meanDiffSqr)
    dataSqrMean = sum(dataSqrMeans)/len(dataSqrMeans)
    stdDeviation = dataSqrMean**(1/2.0)
    return stdDeviation

def test_guague():
    ''' Average set for batch and size
        Speakers    Average         Std Deviation
        --------    -------         ------------
        1           32              
        2           18.10           
        3           14.21           1.672452325
        4           12.66
        5           11.96
        6           11.69
        7           11.62
        8           11.68
        9           12.18
        10          12.36
    '''
    size = 3
    batch = 32
    epochs = 100000

    print("Settings:", "size:", size, "batch:", batch, "epochs:", epochs)

    dataset = []
    average = 0

    start = time.time()
    # t = tqdm.trange(range(epochs), desc='Bar desc', leave=True)
    for epoch in tqdm(range(epochs)):
        custom_set = create_random_sets_total(size, size, batch)
        numpy_data = np.array(custom_set)
        
        max_value = Backtracking(numpy_data, np.arange((size**2)).reshape(size, size))
        dataset.append(float(max_value))
        average = (average*epoch + float(max_value)) / (epoch + 1)
    end = time.time()
    # average /= epochs
    std_dev = get_standard_deviation(dataset)
    print("Average Score:", average)
    print("Total Time:", end - start)
    print("Average Time:", (end - start)/epochs)
    print("Standard Deviation", std_dev)
    save_to_file(str(std_dev) + ", \n")

def save_to_file(info):
    with open("ht.txt", "a") as _file:
        _file.write(info)



def main():
    test_guague()
    # test_size()
    # test_accuracy()
    # print(average())
    

if __name__ == '__main__':
    main()