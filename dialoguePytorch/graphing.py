import matplotlib.pyplot as plt
import os
import glob

PATH = os.path.abspath(os.path.dirname(__file__))

log_file_path = os.path.join(PATH, "logs")
# log_file_path = os.path.join(PATH, "..", "..", "serverData")

glob_name = "A27_training_log_2019_01_04*"

log_file_name = glob.glob(os.path.join(log_file_path, glob_name))[0]
print(log_file_name)
log_file_path = os.path.join(log_file_path, log_file_name)

avg_length = 100
avg = []

# infrence_ex = "[1] Inference Score: 21 	 Batch Size: 32 	 Speakers: 2"

def parse_log(log_line):
    if "epoch:" in log_line:
        split_line = str(log_line).split(" ")
        value = int(split_line[-1].rstrip())
        return ("e", value)
    elif "Training loss:" in log_line:
        split_line = str(log_line).split(" ")
        value = float(split_line[3][:-1].rstrip())
        memory = float(split_line[7][4:].rstrip())
        return ("t", value, memory)
    elif "Inference Score:" in log_line:
        split_line = str(log_line).split(" ")
        value = int(split_line[3].rstrip())
        batch = int(split_line[7].rstrip())
        speakers = int(split_line[10].rstrip())
        loss = float(split_line[14].rstrip())
        return ("v", value, batch, speakers, loss)
    else:
        return ("u",)

def averager(data_point):
    if len(avg) >= avg_length:
        avg.pop(0)
    avg.append(data_point)

def average_of_avg():
    return sum(avg)/len(avg)

def graph_plot(data):
    graph_data = []
    calibration = validation_calibration()
    # print(data)
    # Training loss
    for epoch, data_per_epoch in sorted(data.items()):
        for data_per_line in data_per_epoch["t"]:
            averager(data_per_line[0])
            graph_data.append(average_of_avg())

    # Val score
    # for epoch, data_per_epoch in sorted(data.items()):
    #     for data_per_line in data_per_epoch["v"]:
    #         averager(data_per_line[0])
    #         graph_data.append(average_of_avg())

    # Calibrated val score
    # for epoch, data_per_epoch in sorted(data.items()):
    #     for data_per_line in data_per_epoch["v"]:
    #         graph_data.append(
    #             data_per_line[0] - calibration[data_per_line[2]]
    #         )
    
    # Val loss
    # for epoch, data_per_epoch in sorted(data.items()):
    #     for data_per_line in data_per_epoch["v"]:
    #         averager(data_per_line[3])
    #         graph_data.append(average_of_avg())

    # print(graph_data)
    plt.plot(graph_data)
    plt.show()

def validation_calibration():
    calibration = {
        1 : 32,   
        2 : 18.10,
        3 : 14.21,
        4 : 12.66,
        5 : 11.96,
        6 : 11.69,
        7 : 11.62,
        8 : 11.68,
        9 : 12.18,
        10: 12.36
    }
    # print(calibration)
    return calibration

def main():
    aggregate_per_epoch = {}
    epoch = 0
    # print(parse_log(infrence_ex))
    log = open(log_file_path)
    for line in log:
        # if epoch == 3:
        #     break
        val = parse_log(line)
        if val[0] == "e":
            epoch = val[1]
            if epoch not in aggregate_per_epoch:
                aggregate_per_epoch[epoch] = {"t" : [], "v" : []}
        elif val[0] == "t":
            aggregate_per_epoch[epoch]["t"].append((val[1], val[2]))
        elif val[0] == "v":
            aggregate_per_epoch[epoch]["v"].append((val[1], val[2], val[3], val[4]))
    graph_plot(aggregate_per_epoch)
    print("number of epochs", len(aggregate_per_epoch))
        

if __name__ == "__main__":
    # validation_calibration()
    main()