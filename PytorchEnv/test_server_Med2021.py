import torch
import numpy as np
import socket, pickle
from scipy.stats import norm
from model.model import Med2021



# With DROPout 0.2

# TEST MODELLO CON PYTORCH DELLA RETE ORIGINARIA FATTA CON SCIKIT LEARN

path = 'C:/Users/Pluto/Documents/PHD/Ebrake/e-brake/PytorchEnv/saved/models/Feb21_18-35-29_a713059c@c50e1006/Checkpoint-epoch90.pth'


########################################################################################################################
# NEURAL NETWORK DEFINITION
########################################################################################################################

# build model architecture
input_dim = 100
output_dim = 1

model = Med2021(input_size=input_dim, output_size=output_dim)

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


confidence_calc_samples = 100

while 1:
    conn, addr = s.accept()
    print('Connected by', addr)
    data = conn.recv(4096)
    if data:
        with torch.no_grad():
            input_array = np.asarray(pickle.loads(data))
            tensor_input = torch.from_numpy(input_array).float()
            attempts_tensor = torch.zeros(confidence_calc_samples)
            for i in range(len(attempts_tensor)):
                tensor_input_cuda = tensor_input.to(device)
                attempts_tensor[i]=model(tensor_input_cuda)
            output_torch = attempts_tensor.cpu()
            output_numpy = np.asarray(output_torch.numpy())
            mean, std = norm.fit(output_numpy)
            print('Mean value: %.6f, Standard deviation: %.6f' % (mean, std))
            data_string=pickle.dumps(np.float64((mean, std)))
            conn.send(data_string)
            # conn.close()
            # s.close()
