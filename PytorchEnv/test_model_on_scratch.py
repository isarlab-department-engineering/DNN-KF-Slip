import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pickle
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from model.model import Med2021, Med2021

#PATH MODEL
path = 'C:/Users/Pluto/Documents/PHD/Ebrake/e-brake/PytorchEnv/saved/models/Feb21_18-35-29_a713059c@c50e1006/Checkpoint-epoch90.pth'



########################################################################################################################
# DATASETS LOADER: TRAINING, VALIDATION, TEST
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

ROOT_PATH_DATASET = 'C:/Users/Pluto/Documents/PHD/Ebrake/e-brake/Data/Dataset/'
DATASET_NAME = 'c50e1006'
DATASET_PART = '_ref'

ref_dataset = DatasetBurckhardt(root_path=os.path.join(ROOT_PATH_DATASET, DATASET_NAME,
                                                       DATASET_NAME + DATASET_PART+'.csv'))


ref_loader = torch.utils.data.DataLoader(dataset=ref_dataset, batch_size=1, shuffle=False)


########################################################################################################################
# LOADING MODEL
########################################################################################################################
input_dim = 100
output_dim = 1
model = Med2021(input_size=input_dim, output_size=output_dim)
model.load_state_dict(torch.load(path))


# prepare model for testing
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

best_slip = []
best_slip_hat = []


with torch.no_grad():

    for i, (samples, labels) in enumerate(ref_loader):
        samples_T = samples.to(device)
        output_torch = model(samples_T)
        output = np.float64(output_torch.cpu().numpy())
        best_slip.append(np.float64(labels.cpu().numpy()))
        best_slip_hat.append(output)

########################################################################################################################
# SHOW PREDICTIONS
########################################################################################################################
color_hat = (0.94, 0.61, 0.08)
color_GT = (0.2, 0.54, 1.0)


plt.figure()
plt.title(f'Best slip prediction for the dataset {DATASET_NAME + DATASET_PART+".csv"}')
plt.grid()
plt.xlabel('Samples')
plt.ylabel('Best slip')
h1 = plt.plot(best_slip, '.', color=color_GT, label='Best Slip GT')
h2 = plt.plot(best_slip_hat, '.', color=color_hat, label='Best Slip Hat')
plt.legend()
plt.show()