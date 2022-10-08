import torch
import numpy as np
import socket, pickle
from scipy.stats import norm
from model.model import Med2020, Med2021


########################################################################################################################
# NEURAL NETWORK DEFINITION
########################################################################################################################

# CARICO I MODELLI ADDESTRATI PER LA SIMULAZIONE
path_noDropout = 'C:/Users/Pluto/Documents/PHD/Ebrake/e-brake/PytorchEnv/saved/models/Feb21_15-08-41_c2bc468a' \
                 '@c50e1006/Checkpoint-epoch90.pth '
path_dropout = 'C:/Users/Pluto/Documents/PHD/Ebrake/e-brake/PytorchEnv/saved/models/Feb21_18-35-29_a713059c@c50e1006' \
               '/Checkpoint-epoch90.pth '

# Paramtri IN/OUT DELLE RETI : UGUALI CON FINESTRA DA 50 SU SLIP + MU -> BEST SLIP (100 FEATURES 1 OUTPUT)
input_dim = 100
output_dim = 1

# Creo istanze dei modelli
model_noDO = Med2020(input_size=input_dim, output_size=output_dim)
model_DO = Med2021(input_size=input_dim, output_size=output_dim)

# Carico i pesi nei modelli
model_noDO.load_state_dict(torch.load(path_noDropout))
model_DO.load_state_dict(torch.load(path_dropout))

# Controllo la GPU e alloco i modelli in essa (se disponibile)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_noDO = model_noDO.to(device)
model_noDO.eval()

model_DO = model_DO.to(device)

# Create a TCP/IP socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM
                  )
# Bind the socket to the port
HOST = 'localhost'
VPN_HOST ='10.8.0.8'
PORT = 50008
s.bind((HOST, PORT))
# Listen for incoming connections
s.listen(1)


count = 0
confidence_calc_samples = 100

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
            prediction_noDo = np.float64((model_noDO(tensor_input_cuda)).cpu())
            attempts_tensor = np.zeros(confidence_calc_samples)
            for i in range(len(attempts_tensor)):
                attempts_tensor[i] = np.float64((model_DO(tensor_input_cuda)).cpu())
            prediction_Do, prediction_Do_devStd  = norm.fit(attempts_tensor)
            print('Predicted Value:\nNoDropout:{best_slip_ND}\nDropout:\n\tMean:{best_slip_D}, \n\tDevStd:{devstd_D}'
                  .format(best_slip_ND=prediction_noDo, best_slip_D=prediction_Do, devstd_D=prediction_Do_devStd))
            data_string = pickle.dumps(np.float64(np.array([prediction_noDo, prediction_Do, prediction_Do_devStd])))
            conn.send(data_string)
            count += 1
            print(f'ount sample: [{count}]')
            # conn.close()
            # s.close()

#
# while 1:
#     conn, addr = s.accept()
#     print('Connected by', addr)
#     data = conn.recv(4096)
#     if data:
#         with torch.no_grad():
#             input_array  = np.asarray(pickle.loads(data))
#             print(f'Input Values:[{input_array}]')
#             tensor_input = torch.from_numpy(input_array).float()
#             tensor_input_cuda = tensor_input.to(device)
#             prediction_noDo = np.float64((model_noDO(tensor_input_cuda)).cpu())
#             prediction_Do = np.float64((model_NN(tensor_input_cuda)).cpu())
#             print('Predicted Value:\nNoDropout:{best_slip_ND}\nDropout:\n\tMean:{best_slip_D}, \n\tDevStd:{devstd_D}'
#                   .format(best_slip_ND=prediction_noDo, best_slip_D=prediction_Do, devstd_D=0.0))
#             data_string = pickle.dumps(np.float64(np.array([prediction_noDo, prediction_Do, 0.0])))
#             conn.send(data_string)
#             count += 1
#             print(f'ount sample: [{count}]')
#             # conn.close()
#             # s.close()
#

