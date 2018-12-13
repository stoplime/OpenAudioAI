import os
import time
from model import ABHUE, GlobalModule
from losses import DistanceClusterLoss
import preprocess
import torch
import torch.optim as optim


class training_parameters(object):
    def __init__(self, verbose=0):
        self.verbose = verbose

        # Path Variables
        self.PATH = None
        self.train_data_dir = None
        self.val_data_dir = None

        self.time_stamp = None

        self.save_model_name = None
        self.save_model_global_name = None

        self.save_model_path = None
        self.save_model_global_path = None

        self.load_model_name = None
        self.load_model_global_name = None

        self.load_model_path = None
        self.load_model_global_path = None

        self.saves_files = None

        self.log_file_name = None
        self.log_file_path = None

        # Hyperparameters
        self.epochs = None
        self.batch_size = None
        self.max_speakers = None

        self.device_id = None
        self.model_id = None
        self.recurrent_model = None
        self.window_size = None
        self.dropout = None
        self.stack_size = None

        # Comon vars
        self.device = None
        self.preprocessor = None
        self.learning_rate = None

        # Changing vars
        self.data_file = None
        self.local_model = None
        self.global_model = None
        self.optimizer = None
        self.loss_function = None

    def Hyperparameter_Initialization(self, device_id=0, model_id=0, recurrent_model="lstm",
                                      window_size=3, dropout=0, stack_size=1):
        self.epochs = 30
        self.batch_size = 32
        self.max_speakers = 10

        self.device_id = device_id
        self.model_id = model_id
        self.recurrent_model = recurrent_model
        self.window_size = window_size
        self.dropout = dropout
        self.stack_size = stack_size

    def Path_Initialization(self):
        self.PATH = os.path.abspath(os.path.dirname(__file__))

        # Test Data
        self.train_data_dir = os.path.join(self.PATH, "..", "code", "data", "train")
        self.val_data_dir = os.path.join(self.PATH, "..", "code", "data", "val")

        # Actual Data
        # self.train_data_dir = os.path.join(PATH, "..", "data", "train")
        # self.val_data_dir = os.path.join(PATH, "..", "data", "val")

        if self.model_id == None:
            raise NotImplementedError(self.model_id)

        self.time_stamp = str(time.strftime("%Y_%m_%d-%H_%M_%S"))
        self.save_model_name = "A" + str(self.model_id) + "_model_" + self.time_stamp + ".pt"
        self.save_model_global_name = "A" + str(self.model_id) + "_model_global_" + self.time_stamp + ".pt"
        self.save_model_path = os.path.join(self.PATH, "saves", self.save_model_name)
        self.save_model_global_path = os.path.join(self.PATH, "saves", self.save_model_global_name)

        self.load_model_name = "A" + str(self.model_id) + "_model_2"
        self.load_model_global_name = "A" + str(self.model_id) + "_model_global_"
        self.saves_files = os.listdir(os.path.join(self.PATH, "saves"))
        
        if any(self.load_model_name in x for x in self.saves_files) and \
           any(self.load_model_global_name in x for x in self.saves_files):
            for _file in self.saves_files:
                if self.load_model_name in _file:
                    self.load_model_path = os.path.join(self.PATH, "saves", _file)
                elif self.load_model_global_name in _file:
                    self.load_model_global_path = os.path.join(self.PATH, "saves", _file)
        
        self.log_file_name = "A" + str(self.model_id) + "_training_log_" + self.time_stamp + ".log"
        self.log_file_path = os.path.join(self.PATH, "logs", self.log_file_name)

        # create file directories
        if not os.path.exists(os.path.dirname(self.save_model_path)):
            os.makedirs(os.path.dirname(self.save_model_path))
        if not os.path.exists(os.path.dirname(self.log_file_path)):
            os.makedirs(os.path.dirname(self.log_file_path))

    def Preprocessing_Initialization(self):
        self.device = torch.device("cuda:" + str(self.device_id) if torch.cuda.is_available() else "cpu")
        self.preprocessor = preprocess.PreProcess(self.window_size, dev=self.device)
        self.learning_rate = 0.001

    def Model_Initialization(self):
        # Create Models
        self.local_model = ABHUE(recurrent_model=self.recurrent_model, dropout=self.dropout, 
                                 stack_size=self.stack_size, dev=self.device)
        self.global_model = GlobalModule(recurrent_model=self.recurrent_model, dropout=self.dropout, 
                                         stack_size=self.stack_size, dev=self.device)
        self.optimizer = optim.Adam(list(self.local_model.parameters()) + list(self.global_model.parameters()),
                                    lr=self.learning_rate)
        self.loss_function = DistanceClusterLoss(self.batch_size, dev=self.device)
        
        # Load models
        if self.load_model_path != None and self.load_model_global_path != None:
            self.local_model.load_state_dict(torch.load(self.load_model_path))
            self.global_model.load_state_dict(torch.load(self.load_model_global_path))
        
        # Set load model equal to the save
        self.load_model_path = self.save_model_path
        self.load_model_global_path = self.save_model_global_path

        # Set model to train mode
        self.local_model.train()
        self.global_model.train()

    def Reset_Model(self):
        try:
            del self.local_model
            del self.global_model
            del self.optimizer
            del self.loss_function
        except Exception as e:
            print("Reset_Model", e)
        