
from model import ABHUE
from k_means import Kmeans
import preprocess
import torch
import os
import numpy as np

PATH = os.path.abspath(os.path.dirname(__file__))

data_dir = os.path.join(PATH, "..", "data", "test")
savePath = os.path.join(PATH, "saves", "model.pt")
# savePath = os.path.join(PATH, "backup_saves", "server-trained.pt")

window_size = 3
batch_size = 32
max_speakers = 10
dev = 0

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

def bestLabels(data, labels, size):
    ''' 
        Params
        ------
        data: kmeans clusters
        ------
        labels: List(tuple(sentence id, label))
    '''

    def GetLabel(labels, id):
        for label in labels:
            if label[0] == id:
                return label[1]
        return None
    list_of_labels = []
    labels_without_clusters = []
    # clusters: List(tuple(sentence id, point))
    pred_clusters = [[0 for j in range(size)] for i in range(len(data))]
    for i, cluster in enumerate(data):
        for j, point in enumerate(cluster.points):
            sentence_id = cluster.point_ids[j]
            label = GetLabel(labels, sentence_id)
            if label != None:
                if label not in list_of_labels:
                    list_of_labels.append(label)
                    labels_without_clusters.append(0)
                labels_without_clusters[list_of_labels.index(label)] += 1
                pred_clusters[i][list_of_labels.index(label)] += 1
    
    numpy_clusters = np.array(pred_clusters)
    print(numpy_clusters)
    print(np.array(labels_without_clusters))
    max_value = Backtracking(numpy_clusters, np.arange((size**2)).reshape(size, size))
    return max_value

def CheckNumOfSpeakers(labels):
    speakers = []
    for label in labels:
        if label[1] not in speakers:
            speakers.append(label[1])
    return len(speakers)

def main():
    preprocessor = preprocess.PreProcess(window_size)

    device = torch.device("cuda:"+str(dev) if torch.cuda.is_available() else "cpu")
    model = ABHUE()
    model = model.to(device)
    model.load_state_dict(torch.load(savePath))
    model.eval()

    batch_datas = []
    label_datas = []
    data_idx = 0
    batch_count = 0
    for data_file in os.listdir(data_dir):
        for data in preprocessor.parseData(os.path.join(data_dir, data_file)):
            if not preprocessor.create_sliding_window(data):
                continue
            
            data_input, data_label = preprocessor.tensorfy()

            output = model(data_input)

            batch_datas.append((data_idx, output.detach()))
            label_datas.append((data_idx, data_label))

            num_speakers = CheckNumOfSpeakers(label_datas)
            if data_idx+1 >= batch_size or num_speakers >= max_speakers:
                km = Kmeans(k=num_speakers, size=200)
                km.run( batch_datas )
                score = bestLabels(km.clusters, label_datas, num_speakers)
                # round(score / (data_idx + 1), 10)
                print('[{}] Inference Score: {} \t Batch Size: {} \t Speakers: {}'.format((batch_count + 1), score, data_idx+1, num_speakers)) # 
                batch_count += 1
                data_idx = 0
                batch_datas = []
                label_datas = []
            else:
                data_idx += 1
        print("")
        break

if __name__ == '__main__':
    main()