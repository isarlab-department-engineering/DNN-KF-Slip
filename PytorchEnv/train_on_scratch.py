'''
Train on scratch MLP with pytorch
'''
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tensorboardX import SummaryWriter
import logging
from pathlib import Path
import socket
from datetime import datetime
from model.model import Med2021, Med2020
import sys
import uuid



########################################################################################################################
# CUDA CONFIG AND SEED CONFIGS
########################################################################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 1456454
torch.manual_seed(SEED)
np.random.seed(SEED)

id_model =uuid.uuid1().__str__()[:8]

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

ROOT_PATH_DATASET ='C:/Users/Pluto/Documents/PHD/Ebrake/e-brake/Data/Dataset'
DATASET_NAME = 'c50e1006'

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



########################################################################################################################
# LOGGING AND PATHS DEFINITION
########################################################################################################################
# Logging settings
format_logging = "%(asctime)s: %(message)s"
logging.basicConfig(format=format_logging, level=logging.INFO, datefmt="%H:%M:%S")
logger = logging.getLogger('Logger_pythorch')

# Path Settings
source_path = Path(__file__).resolve().parent
log_path = os.path.join(source_path, 'saved/log')
model_path = os.path.join(source_path, 'saved/models')


# Check main folders
if not os.path.exists(log_path) or not os.path.exists(model_path):
    raise FileNotFoundError(f'An error occurred while creating the following directories '
                            f'\n ---> "{log_path}"\n ---> "{model_path}')

# Setting folder of this run for tensorboard
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
log_dir = os.path.join(
    log_path, current_time + '_' + id_model + '@' + DATASET_NAME + '')

# Creating folder of this run for model savings
model_dir = os.path.join(
    model_path, current_time + '_' + id_model + '@' + DATASET_NAME + '')
try:
    os.makedirs(model_dir)
except (FileExistsError, FileNotFoundError):
    print(f'An error occurred while creating "{model_dir}" directory, exiting')
    sys.exit()

writer = SummaryWriter(log_dir=log_dir)



########################################################################################################################
# TRAINING PARAMETERS AND TRAINING
########################################################################################################################

# Load NN
input_dim = 100
output_dim = 1
model = Med2021(input_size=input_dim, output_size=output_dim)
model.to(device)
learning_rate = 0.001
num_epochs = 100
save_epoch_interval = 10
alpha_decay = 0.0001
# loss, optimizer
criterion = nn.MSELoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True,
   #                         weight_decay=learning_rate*alpha_decay)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=learning_rate*alpha_decay )

# training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    loss_logs_epoch = []
    metrics_logs_epoch = []
    for i, (samples, labels) in enumerate(train_loader):
        samples_T = samples.to(device)
        labels_T = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()


        # forward + backward + optimize
        outputs = model(samples_T)
        loss = criterion(outputs, labels_T)
        loss.backward()
        optimizer.step()

        # Logging
        loss_logs = loss.item()
        metrics_logs = np.sqrt(loss_logs)
        logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss_logs:.5f},'
                     f' Metrics: {metrics_logs:.5f}')
        loss_logs_epoch.append(loss_logs)
        metrics_logs_epoch.append(metrics_logs)

    loss_per_epoch = np.mean(loss_logs_epoch)
    metrics_per_epoch = np.mean(metrics_logs_epoch)
    writer.add_scalar("Loss/train_mse", loss_per_epoch, epoch)
    writer.add_scalar("Metric/train_rmse", metrics_per_epoch, epoch)
    logger.info(f'EPOCH SUMMARY:[{epoch + 1}], Loss Train: {loss_per_epoch:.5f},'
                 f' Metrics Train: {metrics_per_epoch:.5f}')


    # After each epoch validate
    with torch.no_grad():
        loss_logs_v = []
        metrics_logs_v = []
        for i, (samples, labels) in enumerate(valid_loader):
            samples_T = samples.to(device)
            labels_T = labels.to(device)

            # Forward pass
            outputs = model(samples_T)
            loss = criterion(outputs, labels_T)

            loss_logs_v.append(loss.item())
            metrics_logs_v.append(np.sqrt(loss.item()))

        loss_per_epoch_v = np.mean(loss_logs_v)
        metrics_per_epoch_v = np.mean(metrics_logs_v)
        writer.add_scalar("Loss/valid_mse", np.mean(loss_per_epoch_v), epoch)
        writer.add_scalar("Metric/valid_rmse", np.mean(metrics_per_epoch_v), epoch)

        logger.info(f'VALIDATION: Epoch [{epoch + 1}], Loss Val: {loss_per_epoch:.5f},'
                    f' Metrics Val: {metrics_per_epoch:.5f}\n\n')
        # test values
        sample_ref, label_ref = next(iter(ref_loader))
        sample_ref = sample_ref.to(device)
        label_hat = model(sample_ref)

        logger.info(f'EXAMPLE OF PREDICTION: Y=[{label_ref.cpu().numpy().flatten()[0]:.5f}], '
                    f'Y_HAT={label_hat.cpu().numpy().flatten()[0]:.5f} \n\n')

    if epoch % save_epoch_interval == 0:
        model_name = 'Checkpoint-epoch'+ str(epoch) + '.pth'
        logger.info(f'\n\nSAVING model {model_name} into {model_path} \n\n')
        torch.save(model.state_dict(), os.path.join(model_dir, model_name))

    model_name = 'Final_model.pth'
    logger.info(f'\n\nSAVING model {model_name} into {model_path} \n\n')
    torch.save(model, os.path.join(model_dir, model_name))

    writer.flush()

writer.close()
