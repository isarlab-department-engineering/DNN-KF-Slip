import torch
import numpy as np
import socket, pickle
from scipy.stats import norm
from model.model import Med2020


# With DROPout 0.2

# TEST MODELLO CON PYTORCH DELLA RETE ORIGINARIA FATTA CON SCIKIT LEARN

path = 'C:/Users/Pluto/Documents/PHD/Ebrake/e-brake/PytorchEnv/saved/models/Feb21_15-08-41_c2bc468a@c50e1006/Checkpoint-epoch50.pth'
########################################################################################################################
# NEURAL NETWORK DEFINITION
########################################################################################################################

# build model architecture
input_dim = 100
output_dim = 1

model = Med2020(input_size=input_dim, output_size=output_dim)

model.load_state_dict(torch.load(path))

# prepare model for testing
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


# Create a TCP/IP socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM
                  )
# Bind the socket to the port
HOST = 'localhost'
VPN_HOST ='10.8.0.8'
PORT = 50007
s.bind((HOST, PORT))

# Listen for incoming connections
s.listen(1)

model.eval()

count = 0

while 1:
    conn, addr = s.accept()
    print('Connected by', addr)
    data = conn.recv(4096)
    if data:
        with torch.no_grad():
            input_array  = np.asarray(pickle.loads(data))
            print(f'Input Values:[{input_array}]')
            tensor_input = torch.from_numpy(input_array).float()
            tensor_input_cuda = tensor_input.to(device)
            output_torch = model(tensor_input_cuda)
            prediction = output_torch.cpu()
            output_numpy = np.float64(prediction.numpy())
            print(f'Predicted Value:[{output_numpy:.5f}]')
            data_string=pickle.dumps(np.float64((output_numpy, 0.0)))
            conn.send(data_string)
            count +=1;
            print(f'ount sample: [{count}]')
            # conn.close()
            # s.close()
