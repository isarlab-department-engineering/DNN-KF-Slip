import numpy as np
import socket, pickle
import array

def predict(input_arr:np.array) -> np.array:

    HOST = 'localhost'
    PORT = 50008
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    # arr = ([1,2,3,4,5,6,1,2,3,4,5,6])
    data_string = pickle.dumps(input_arr)
    s.send(data_string)

    data = s.recv(4096)
    [pred_noDrop, pred_Drop, dev_std_Drop] = pickle.loads(data)
    s.close()
    outputs = array.array('d', [pred_noDrop, pred_Drop, dev_std_Drop])
    return outputs
