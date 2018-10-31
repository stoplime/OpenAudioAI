
from model import ABHUE
from k_means import Kmeans
import preprocess
import torch
import os

PATH = os.path.abspath(os.path.dirname(__file__))

data_dir = os.path.join(PATH, "..", "data", "test")
savePath = os.path.join(PATH, "saves", "model.pt")

window_size = 3
batch_size = 32

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
    pred_clusters = []
    for i, cluster in enumerate(data):
        pred_clusters.append([])
        for point in cluster.points:
            pred_clusters[i].append(point)
    return 25

def main():
    preprocessor = preprocess.PreProcess(window_size)
    km = Kmeans(k=2, size=200)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ABHUE()
    model = model.to(device)
    model.load_state_dict(torch.load(savePath))
    model.eval()

    batch_data = []
    label_data = []
    data_idx = 0
    batch_count = 0
    for data_file in os.listdir(data_dir):
        for data in preprocessor.parseData(os.path.join(data_dir, data_file)):
            if not preprocessor.create_sliding_window(data):
                continue
            
            data_input, data_label = preprocessor.tensorfy()

            output = model(data_input)

            batch_data.append((data_idx, output.detach()))
            label_data.append((data_idx, data_label))

            if data_idx >= batch_size:
                km.run( batch_data )
                score = bestLabels(km.clusters, label_data)
                print('[{}] Inference Score: {}'.format((batch_count + 1), round(score / (data_idx + 1), 10)))
                batch_count += 1
                data_idx = 0
            else:
                data_idx += 1
        print("")
        break

if __name__ == '__main__':
    main()