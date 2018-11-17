import matplotlib.pyplot as plt
import os

PATH = os.path.abspath(os.path.dirname(__file__))

log_file_name = "epoch1.txt"
log_file_path = os.path.join(PATH, "logs", log_file_name)

# infrence_ex = "[1] Inference Score: 21 	 Batch Size: 32 	 Speakers: 2"

def parse_log(log_line):
    if "epoch:" in log_line:
        split_line = str(log_line).split(" ")
        value = int(split_line[-1].rstrip())
        return ("e", value)
    elif "Training loss:" in log_line:
        split_line = str(log_line).split(" ")
        value = float(split_line[-1].rstrip())
        return ("t", value)
    elif "Inference Score:" in log_line:
        split_line = str(log_line).split(" ")
        value = int(split_line[3].rstrip())
        batch = int(split_line[7].rstrip())
        speakers = int(split_line[-1].rstrip())
        return ("v", value, batch, speakers)
    else:
        return ("u",)

def graph_plot(data):
    train_data = []
    print(data)
    # for epoch, data_per_epoch in sorted(data.items()):
        # print(epoch, data_per_epoch)
        # train_data.append(sum(data_per_epoch["t"]))
    for train_data_per_line in data[0]["t"]:
        train_data.append(train_data_per_line)
    # print(train_data)
    plt.plot(train_data)
    plt.show()

def main():
    aggregate_per_epoch = {}
    epoch = 0
    # print(parse_log(infrence_ex))
    log = open(log_file_path)
    for line in log:
        val = parse_log(line)
        if val[0] == "e":
            epoch = val[1]
            if epoch not in aggregate_per_epoch:
                aggregate_per_epoch[epoch] = {"t" : [], "v" : []}
        elif val[0] == "t":
            print(epoch, val[1])
            aggregate_per_epoch[epoch]["t"].append(val[1])
        elif val[0] == "v":
            aggregate_per_epoch[epoch]["v"].append((val[1], val[2], val[3]))
    graph_plot(aggregate_per_epoch)
        

if __name__ == "__main__":
    main()