from scipy.stats import norm
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import logging
from pathlib import Path
from datetime import datetime
from model.model import Med2021, Med2020
import sys

########################################################################################################################
# FOLDERS AND MODEL SELECTION
########################################################################################################################

ROOT_PATH_DATASET ='C:/Users/Pluto/Documents/PHD/Ebrake/e-brake/Data/Dataset'
DATASET_NAME = 'c50e1006'


models_path = 'C:/Users/Pluto/Documents/PHD/Ebrake/e-brake/PytorchEnv/saved/models'
# model_name = 'Feb21_18-35-29_a713059c@c50e1006'  # MODEL TORCH WITH DO
model_name = 'Feb21_15-08-41_c2bc468a@c50e1006'  # MODEL TORCH WITHOUT DO
model_chkpt = 'Checkpoint-epoch90.pth'
model_path = ''.join([models_path, '/', model_name, '/' + model_chkpt])
model_input_dim = 100
model_output_dim = 1

# Creo istanze dei modelli

# model_NN = Med2021(input_size=model_input_dim, output_size=model_output_dim)
model_NN = Med2020(input_size=model_input_dim, output_size=model_output_dim)

########################################################################################################################
# DATASET CLASS DEFINITION
########################################################################################################################

class DatasetBurckhardt(Dataset):
    """
     Burckard dataset type
     """

    def __init__(self, root_path):
        self.data = np.genfromtxt(root_path, delimiter=',')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        x = self.data[index, 0:len(self.data[index])-1]
        y = np.asarray([self.data[index, -1]])

        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


########################################################################################################################
# DATASETS CREATION: TRAINING, VALIDATION, TEST
########################################################################################################################
train_dataset = DatasetBurckhardt(root_path=os.path.join(ROOT_PATH_DATASET, DATASET_NAME, DATASET_NAME+'_used.csv'))

valid_dataset = DatasetBurckhardt(root_path=os.path.join(ROOT_PATH_DATASET, DATASET_NAME, DATASET_NAME+'_val.csv'))

test_dataset = DatasetBurckhardt(root_path=os.path.join(ROOT_PATH_DATASET, DATASET_NAME, DATASET_NAME+'_hid.csv'))

ref_dataset = DatasetBurckhardt(root_path=os.path.join(ROOT_PATH_DATASET, DATASET_NAME, DATASET_NAME+'_ref.csv'))

########################################################################################################################
# DATA LOADERS
########################################################################################################################

batch_size_tr = 2048
batch_size_val = 128
batch_size_test = batch_size_ref = 1


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size_tr, shuffle=True)

valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size_val, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=True)

ref_loader = torch.utils.data.DataLoader(dataset=ref_dataset, batch_size=batch_size_ref, shuffle=True)

list_of_dataset = [(train_loader, 'train_loader'), (valid_loader, 'valid_loader'),
                   (test_loader, 'test_loader'), (ref_loader, 'ref_loader')]

########################################################################################################################
# NEURAL NETWORK DEFINITION
########################################################################################################################


# Carico i pesi nei modelli
model_NN.load_state_dict(torch.load(model_path))

# Controllo la GPU e alloco i modelli in essa (se disponibile)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_NN = model_NN.to(device)


########################################################################################################################
# LOGGING AND PATHS DEFINITION
########################################################################################################################


# Path Settings
source_path = Path(__file__).resolve().parent
source_path = str(source_path).replace("\\",'/')
log_path = ''.join([source_path, '/saved', '/evaluations'])

# Check main folders
if not os.path.exists(log_path):
    try:
        print('Folder {} not found, creating one....'.format(log_path))
        os.makedirs(log_path)
    except (FileExistsError, FileNotFoundError):
        print(f'An error occurred while creating "{log_path}" directory, exiting')
        sys.exit(1)

# Setting folder and log file
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
log_filename = ''.join([log_path, '/', current_time+'_results.log'])


try:
    print('Folder {} not found, creating one....'.format(log_filename))
    f = open(log_filename, "x")

except (FileExistsError, FileNotFoundError):
    print(f'An error occurred while creating "{log_filename}" directory, exiting')
    sys.exit()

# Logging settings

format_logging = '%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s'
logging.basicConfig(filename=log_filename, filemode='a',format=format_logging, level=logging.INFO, datefmt="%H:%M:%S")
logging.info(''.join(['\n\t model: ', model_name, ' \n\tdataset: ', DATASET_NAME]))
logger = logging.getLogger('Logger_pythorch_evals')


########################################################################################################################
# EVALUATION
########################################################################################################################
criterion = nn.MSELoss()
with torch.no_grad():

    for dataset in list_of_dataset:
        dataset_loader = dataset[0]
        dataset_name = dataset[1]
        loss_logs_v = []
        metrics_logs_v = []
        for i, (samples, labels) in enumerate(valid_loader):
            samples_T = samples.to(device)
            labels_T = labels.to(device)

            # Forward pass
            outputs = model_NN(samples_T)
            loss = criterion(outputs, labels_T)

            loss_logs_v.append(loss.item())
            metrics_logs_v.append(np.sqrt(loss.item()))

        mse_loss = np.mean(loss_logs_v)
        rmse_loss = np.mean(metrics_logs_v)

        logging.info(''.join(['\n\n------------------------------------------------------------------',
                              '\n\tdataset_loader: '+str(dataset_name),
                              '\n\tmse_loss:' + str(mse_loss),
                              '\n\trmse_loss:' + str(rmse_loss)
                              ]))
f.flush()
f.close()