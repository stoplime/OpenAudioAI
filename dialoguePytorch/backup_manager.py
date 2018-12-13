import os
from distutils.dir_util import copy_tree
import time

PATH = os.path.abspath(os.path.dirname(__file__))

save_folder = os.path.join(PATH, "saves")
log_folder = os.path.join(PATH, "logs")

backup_folder = os.path.join(PATH, "..", "backup")
backup_log_folder = os.path.join(PATH, "..", "backup", "logs")

def main():
    if not os.path.exists(os.path.dirname(save_folder)):
        os.makedirs(os.path.dirname(save_folder))
    if not os.path.exists(os.path.dirname(log_folder)):
        os.makedirs(os.path.dirname(log_folder))
    
    while True:
        time_stamp = str(time.strftime("%Y_%m_%d-%H_%M_%S"))

        backup_save_folder = os.path.join(backup_folder, "model_"+time_stamp, "saves")
        backup_log_folder = os.path.join(backup_folder, "model_"+time_stamp, "logs")

        if not os.path.exists(os.path.dirname(backup_save_folder)):
            os.makedirs(os.path.dirname(backup_save_folder))
        if not os.path.exists(os.path.dirname(backup_log_folder)):
            os.makedirs(os.path.dirname(backup_log_folder))

        copy_tree(save_folder, backup_save_folder)
        copy_tree(log_folder, backup_log_folder)

        print("Backed up @ {}".format(time_stamp))

        time.sleep(3600)
        # time.sleep(6)

if __name__ == '__main__':
    main()
