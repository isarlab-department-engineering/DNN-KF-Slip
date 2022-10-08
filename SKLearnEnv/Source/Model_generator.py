import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from joblib import dump
from tqdm import tqdm
from SKLearnEnv.Models.DataTools.DatasetTools import CsvManipulator
import datetime
import os
import uuid
from SKLearnEnv.Models import Configuration as Cfg
import SKLearnEnv.Models.DataTools.FileTools as Ft
import logging


###########################################################################
#                                                                         #
#                            INIT ACTIONS                                 #
#                                                                         #
###########################################################################

# Basic setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Dataset_Loading')

# Loading parser
csv_parser = Cfg.MlParser().parser
args = csv_parser.parse_args()

###########################################################################
#                                                                         #
#                          Loading DATASET                                #
#                                                                         #
###########################################################################

# Dataset load
dataset_id = args.DatasetID
dataset_full_path = os.path.join(Cfg.DATASET_DIR, dataset_id)


logger.info('Dataset Path: %s', dataset_full_path)
logger.info('Dataset ID: %s', dataset_id)


# Loading dataset into variables
path_csv_used = os.path.join(dataset_full_path, ''.join([dataset_id, '_used.csv']))
path_csv_val = os.path.join(dataset_full_path, ''.join([dataset_id, '_val.csv']))
path_csv_hid = os.path.join(dataset_full_path, ''.join([dataset_id, '_hid.csv']))
path_csv_ref = os.path.join(dataset_full_path, ''.join([dataset_id, '_ref.csv']))

dataset = np.genfromtxt(path_csv_used, delimiter=',')
dataset_val = np.genfromtxt(path_csv_val, delimiter=',')
dataset_hil = np.genfromtxt(path_csv_hid, delimiter=',')
dataset_ref = np.genfromtxt(path_csv_ref, delimiter=',')

# Preparing datasets for training and evaluations

last_column_y_index = dataset.shape[1]-1

# Used Dataset
x_tr_us, y_tr_us = CsvManipulator.split_xy_supervised(dataset, last_column_y_index)

# Validation Dataset
x_val, y_val = CsvManipulator.split_xy_supervised(dataset_val, last_column_y_index)

# Hidden Dataset
x_hid, y_hid = CsvManipulator.split_xy_supervised(dataset_hil, last_column_y_index)

# reference Dataset
x_ref, y_ref = CsvManipulator.split_xy_supervised(dataset_ref, last_column_y_index)



###########################################################################
#                                                                         #
#                            MLP REGRESSOR                                #
#                                                                         #
###########################################################################
logger = logging.getLogger('MLP Regressor')

# MLP models
param_grid_MLP = [
	{'hidden_layer_sizes': [(250, 250)]}
]
logger.info('List of parameters: %s', param_grid_MLP)


# training
for param in tqdm(param_grid_MLP[0].__getitem__('hidden_layer_sizes'), desc='\nModel generation'):
	# Generating the ID of the MLP
	current_time = datetime.datetime.now()
	id_mlp = uuid.uuid1().__str__()[:8]
	# MLP
	hidden_layers = param
	perceptron_regressor = MLPRegressor(hidden_layers, activation='relu',
										solver='sgd', alpha=0.0001, batch_size='auto',
										learning_rate='constant', learning_rate_init=0.001,
										power_t=0.5, max_iter=100, shuffle=True,
										random_state=None, tol=0.0000001, verbose=True,
										warm_start=False, momentum=0.9,
										nesterovs_momentum=True, early_stopping=False,
										validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
										epsilon=1e-08)

	# Training
	perceptron_regressor.fit(x_tr_us, y_tr_us)

	# Scoring
	y_hat_tr = perceptron_regressor.predict(x_tr_us)
	y_hat_vl = perceptron_regressor.predict(x_val)
	y_hat_hidden = perceptron_regressor.predict(x_hid)
	y_hat_reference = perceptron_regressor.predict(x_ref)

	mse_tr = metrics.mean_squared_error(y_tr_us, y_hat_tr)
	mse_val = metrics.mean_squared_error(y_val, y_hat_vl)
	mse_hid = metrics.mean_squared_error(y_hid, y_hat_hidden)
	mse_ref = metrics.mean_squared_error(y_ref, y_hat_reference)

	r2_tr = metrics.r2_score(y_tr_us, y_hat_tr)
	r2_val = metrics.r2_score(y_val, y_hat_vl)
	r2_hid = metrics.r2_score(y_hid, y_hat_hidden)
	r2_ref = metrics.r2_score(y_ref, y_hat_reference)

	rmse_tr = np.sqrt(mse_tr)
	rmse_val = np.sqrt(mse_val)
	rmse_hid = np.sqrt(mse_hid)
	rmse_ref = np.sqrt(mse_ref)

	# Computational complexity
	coeff_MLP = perceptron_regressor.coefs_
	tot_number_param = x_tr_us.shape[1]
	tot_activations = 0
	for i in range(0, len(coeff_MLP)):
		tot_number_param *= len(coeff_MLP[i][0])
		tot_activations += len(coeff_MLP[i][0])
	flops = (tot_number_param * 2) + tot_activations

	# Information of MLP model
	result = {
		'Id mlp': id_mlp,
		'nhidden layers': 	perceptron_regressor.get_params().__getitem__('hidden_layer_sizes'),
		'FLOPS': flops,
		'MSE training':		mse_tr	,
		'MSE validation': 	mse_val	,
		'MSE Hidden': 		mse_hid	,
		'MSE_Reference': 		mse_ref	,
		'RMSE training': 		rmse_tr	,
		'RMSE validation': 	rmse_val,
		'RMSE Hidden': 		rmse_hid,
		'rMSE_Reference':		rmse_ref
	}

	# Display information
	for key in result:
		logger.info(''.join([str(key), ' -> ', str(result[key])]))

	# Logging And storing
	working_directory = str(Ft.Folder(Cfg.MLM_PATH, id_mlp).get_path())
	txt_writer = Ft.FileTxt(path=working_directory, filename=id_mlp)
	txt_writer.append_row_to_txt('\n----------------------------------------------------------------------------')
	txt_writer.append_row_to_txt('\nMODEL-ID : %s' % id_mlp)
	txt_writer.append_row_to_txt('\nDATE OF CREATION: %s' % current_time)
	txt_writer.append_row_to_txt('\nDATASET-ID USED: %s' % dataset_id)
	txt_writer.append_row_to_txt('\n----------------------------------------------------------------------------')
	txt_writer.append_row_to_txt('\n MODEL %s SUMMARY:' % id_mlp)
	txt_writer.append_row_to_txt('\n----------------------------------------------------------------------------')
	txt_writer.append_row_to_txt('\n'.join(['\t', str(perceptron_regressor)]))
	txt_writer.append_row_to_txt('\n----------------------------------------------------------------------------')
	txt_writer.append_row_to_txt('\n MODEL %s INFORMATIONS AND RESULTS:' % id_mlp)
	txt_writer.append_row_to_txt('\n----------------------------------------------------------------------------\n')
	for key in result:
		txt_writer.append_row_to_txt(''.join(['\t', str(key), ' -> ', str(result[key]), '\n']))
	txt_writer.append_row_to_txt('----------------------------------------------------------------------------\n\n\n')

	# Model dumping
	model_filename = os.path.join(working_directory, ''.join([id_mlp, '_MLP_', dataset_id, '.joblib']))
	dump(perceptron_regressor, model_filename)

	# weights dumping
	weights_filename = os.path.join(working_directory, ''.join(['Weights_',id_mlp, '_MLP_', dataset_id, '.joblib']))
	dump(coeff_MLP, weights_filename)

	# Saving images
	plt.figure()
	plt.plot(y_tr_us, '.')
	plt.plot(y_hat_tr, '.')
	plt.grid()
	plt.xlabel('Samples')
	plt.ylabel('Best slip value')
	plt.title(id_mlp + ': Training dataset')
	plt.legend(['True', 'Predicted'])
	plt.savefig(os.path.join(working_directory, ''.join([id_mlp, '_training', '.png'])), format='png')

	plt.figure()
	plt.plot(y_val, '.')
	plt.plot(y_hat_vl, '.')
	plt.grid()
	plt.xlabel('Samples')
	plt.ylabel('Best slip value')
	plt.title(id_mlp + ': Validation dataset')
	plt.legend(['True', 'Predicted'])
	plt.savefig(os.path.join(working_directory, ''.join([id_mlp, '_validation', '.png'])), format='png')

	plt.figure()
	plt.plot(y_hid, '.')
	plt.plot(y_hat_hidden, '.')
	plt.grid()
	plt.xlabel('Samples')
	plt.ylabel('Best slip value')
	plt.title(id_mlp + ':  Hidden dataset')
	plt.legend(['True', 'Predicted'])
	plt.savefig(os.path.join(working_directory, ''.join([id_mlp, '_hidden', '.png'])), format='png')

	plt.figure()
	plt.plot(y_ref, '.')
	plt.plot(y_hat_reference, '.')
	plt.grid()
	plt.xlabel('Samples')
	plt.ylabel('Best slip value')
	plt.title(id_mlp + ':  Reference dataset')
	plt.legend(['True', 'Predicted'])
	plt.savefig(os.path.join(working_directory, ''.join([id_mlp, '_reference', '.png'])), format='png')

