import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from joblib import dump, load
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
logger = logging.getLogger('INIT')
logger.info('Init. activities...')
current_time = datetime.datetime.now()

# Loading parser
csv_parser = Cfg.EvalParser().parser
args = csv_parser.parse_args()

# Dataset used
dataset_id = args.DatasetID

# List of models
list_models = args.ModelsID

# Reference float point operations
max_flops_hardware_ref = args.DevMaxFlops
frequency_Hz_Op = args.DevHz

# Identifier  and destination folder for the evaluation
id_evaluation = uuid.uuid1().__str__()[:8]

# Creating Folder
working_directory = Ft.Folder(Cfg.EVAL_PATH, id_evaluation).get_path()

# Creating report File
txt_writer = Ft.FileTxt(path=working_directory, filename=id_evaluation)
txt_writer.append_row_to_txt('\n----------------------------------------------------------------------------')
txt_writer.append_row_to_txt('\nEVALUATION-ID : %s' % id_evaluation)
txt_writer.append_row_to_txt('\nDATE OF CREATION: %s' % current_time)
txt_writer.append_row_to_txt('\n----------------------------------------------------------------------------')
txt_writer.append_row_to_txt('\nDATASET-ID USED FOR EVALUARION: %s' % dataset_id)
txt_writer.append_row_to_txt('\n----------------------------------------------------------------------------')
txt_writer.append_row_to_txt('\nMODELS EVALUATED:')
for model_name in list_models:
	txt_writer.append_row_to_txt(''.join(['\t', '->', model_name, '\n']))
txt_writer.append_row_to_txt('----------------------------------------------------------------------------\n\n')

logger.info('Done!')


###########################################################################
#                                                                         #
#                          Loading DATASET                                #
#                                                                         #
###########################################################################

# Basic setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DATASET')
logger.info('Loading dataset %s' % dataset_id)


# Loading dataset into variables
dataset_full_path = os.path.join(Cfg.DATASET_DIR, dataset_id)
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
#                     MLP REGRESSORS EVALUATION                           #
#                                                                         #
###########################################################################
logger = logging.getLogger('MLP_LOADER')
logger.info('Loading mlp models... %s' % dataset_id)

# create dictionary
MLP_list = []
for model_name in list_models:
	current_working_path = os.path.join(Cfg.MLM_PATH, model_name)
	if os.path.isdir(current_working_path):
		found_model = False
		for filename in os.listdir(current_working_path):
			if filename.endswith(".joblib") and not filename.startswith("Weights"):
				found_model = True
				MLP_model = load(os.path.join(current_working_path, filename))
				MLP = {
					'Id mlp': filename[0:8],
					'Model': MLP_model,
				}
				MLP_list.append(MLP)
		if not found_model:
			logger.error('Unable to find a valid .joblib model related to %s' % model_name)
	else:
		logger.error('Unable to find a valid path %s' % current_working_path)

# Scoring
results = []
for model in tqdm(MLP_list, desc='\nModels evaluations', leave=True):
	id_mlp = model.__getitem__('Id mlp')
	logger.info('evaluating : %s', id_mlp)
	perceptron_regressor = model.__getitem__('Model')


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

	# computaional complexity
	coeff_MLP = perceptron_regressor.coefs_
	tot_number_param = x_tr_us.shape[1]
	tot_activations = 0
	for i in range(0, len(coeff_MLP)):
		tot_number_param *= len(coeff_MLP[i][0])
		tot_activations += len(coeff_MLP[i][0])

	flops = (tot_number_param*2) + tot_activations

	# Information of MLP model
	result = {
		'Id mlp': id_mlp,
		'hidden layers': perceptron_regressor.get_params().__getitem__('hidden_layer_sizes'),
		'FLOPS': flops,
		'MSE training': mse_tr,
		'MSE validation': mse_val,
		'MSE Hidden': mse_hid,
		'MSE_Reference': mse_ref,
		'RMSE training': rmse_tr,
		'RMSE validation': rmse_val,
		'RMSE Hidden': rmse_hid,
		'RMSE_Reference': rmse_ref
	}

	# Add this result to the list of the results
	results.append(result)

	# Logging the information about this model
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

	filename = ''.join([working_directory, '/', id_mlp, '_MLP_', dataset_id, '.joblib'])
	dump(perceptron_regressor, filename)

	logger.info('Done!')


###########################################################################
#                                                                         #
#                     RESULTS PROCESSING                                  #
#                                                                         #
###########################################################################

id_mlp_list = []
hidden_layers_list = []
mse_tr_list = []
mse_val_list = []
mse_hid_list = []
mse_ref_list = []
rmse_tr_list = []
rmse_val_list = []
rmse_hid_list = []
rmse_ref_list = []
floating_op_list = []

for result in results:
	id_mlp_list.append(result.__getitem__('Id mlp'))
	hidden_layers_list.append(str(result.__getitem__('hidden layers')))
	floating_op_list.append(result.__getitem__('FLOPS')*frequency_Hz_Op)
	mse_tr_list.append(result.__getitem__('MSE training'))
	mse_val_list.append(result.__getitem__('MSE validation'))
	mse_hid_list.append(result.__getitem__('MSE Hidden'))
	mse_ref_list.append(result.__getitem__('MSE_Reference'))
	rmse_tr_list.append(result.__getitem__('RMSE training'))
	rmse_val_list.append(result.__getitem__('RMSE validation'))
	rmse_hid_list.append(result.__getitem__('RMSE Hidden'))
	rmse_ref_list.append(result.__getitem__('RMSE_Reference'))

# sorting results
index_sort = np.argsort(floating_op_list, ).astype(int)[::-1]
id_mlp_list= np.asarray(id_mlp_list)[index_sort]
hidden_layers_list = np.asarray(hidden_layers_list)[index_sort]
floating_op_list = np.asarray(floating_op_list)[index_sort]
mse_tr_list = np.asarray(mse_tr_list)[index_sort]
mse_val_list = np.asarray(mse_val_list)[index_sort]
mse_hid_list = np.asarray(mse_hid_list)[index_sort]
mse_ref_list = np.asarray(mse_ref_list)[index_sort]
rmse_tr_list = np.asarray(rmse_tr_list)[index_sort]
rmse_val_list = np.asarray(rmse_val_list)[index_sort]
rmse_hid_list = np.asarray(rmse_hid_list)[index_sort]
rmse_ref_list = np.asarray(rmse_ref_list)[index_sort]


###########################################################################
#                                                                         #
#                    			FIGURES                                   #
#                                                                         #
###########################################################################

plt.figure(1)
plt.plot(hidden_layers_list, mse_tr_list, '.')
plt.plot(hidden_layers_list, mse_val_list, '.')
plt.plot(hidden_layers_list, mse_hid_list, '.')
plt.plot(hidden_layers_list, mse_ref_list, '.')
plt.grid()
plt.xlabel('hidden layers')
plt.ylabel('MSE scores')
plt.title('Mean Squared Error values')
plt.legend(['Training', 'Validation', 'Test', 'Burc. Reference'])
plt.xticks(rotation=90)
plt.show()
plt.savefig(os.path.join(working_directory, ''.join([id_evaluation, '_MSE_values', '.png'])), format='png')

plt.figure(2)
plt.subplot(2, 2, 1)
plt.plot(hidden_layers_list, mse_tr_list, '.')
plt.grid()
plt.xlabel('hidden layers')
plt.ylabel('MSE score')
plt.title('Training')
plt.legend(['Evaluated Models'])
plt.xticks(rotation=90)
plt.subplot(2, 2, 2)
plt.plot(hidden_layers_list, mse_val_list, '.')
plt.grid()
plt.xlabel('hidden layers')
plt.ylabel('MSE score')
plt.title('Validation')
plt.legend(['Evaluated Models'])
plt.xticks(rotation=90)
plt.subplot(2, 2, 3)
plt.plot(hidden_layers_list, mse_hid_list, '.')
plt.grid()
plt.xlabel('hidden layers')
plt.ylabel('MSE score')
plt.title('Test')
plt.legend(['Evaluated Models'])
plt.xticks(rotation=90)
plt.subplot(2, 2, 4)
plt.plot(hidden_layers_list, mse_ref_list, '.')
plt.grid()
plt.xlabel('hidden layers')
plt.ylabel('MSE score')
plt.title('Burc. Reference')
plt.legend(['Evaluated Models', 'Reference Model'])
plt.xticks(rotation=90)
plt.savefig(os.path.join(working_directory, ''.join([id_evaluation, '_MSE_values_comparison', '.png'])), format='png')


plt.figure(3)
plt.plot(hidden_layers_list, rmse_tr_list, '.')
plt.plot(hidden_layers_list, rmse_val_list, '.')
plt.plot(hidden_layers_list, rmse_hid_list, '.')
plt.plot(hidden_layers_list, rmse_ref_list, '.')
plt.grid()
plt.xlabel('hidden layers')
plt.ylabel('RMSE scores')
plt.title('Root Mean Squared Error values')
plt.legend(['Training', 'Validation', 'Test', 'Burc. Reference'])
plt.xticks(rotation=90)
plt.show()

plt.savefig(os.path.join(working_directory, ''.join([id_evaluation, '_RMSE_values', '.png'])), format='png')


plt.figure(4)
plt.subplot(2, 2, 1)
plt.plot(hidden_layers_list, rmse_tr_list, '.')
plt.grid()
plt.xlabel('hidden layers')
plt.ylabel('RMSE score')
plt.title('Training')
plt.legend(['Evaluated Models'])
plt.xticks(rotation=90)
plt.subplot(2, 2, 2)
plt.plot(hidden_layers_list, rmse_val_list, '.')
plt.grid()
plt.xlabel('hidden layers')
plt.ylabel('RMSE score')
plt.title('Validation')
plt.legend(['Evaluated Models'])
plt.xticks(rotation=90)
plt.subplot(2, 2, 3)
plt.plot(hidden_layers_list, rmse_hid_list, '.')
plt.grid()
plt.xlabel('hidden layers')
plt.ylabel('RMSE score')
plt.title('Test')
plt.legend(['Evaluated Models'])
plt.xticks(rotation=90)
plt.subplot(2, 2, 4)
plt.plot(hidden_layers_list, rmse_ref_list, '.')
plt.grid()
plt.xlabel('hidden layers')
plt.ylabel('RMSE score')
plt.title('Burc. Reference')
plt.legend(['Evaluated Models'])
plt.xticks(rotation=90)
plt.savefig(os.path.join(working_directory, ''.join([id_evaluation, '_RMSE_values_comparison', '.png'])), format='png')
plt.figure()
plt.plot(hidden_layers_list, floating_op_list, '.')
plt.plot(hidden_layers_list, np.ones((len(hidden_layers_list)))*max_flops_hardware_ref, '.')
plt.grid()
plt.xlabel('hidden layers')
plt.ylabel('FLOPs')
plt.title('Floating Point Operations Required')
plt.legend(['Needed', 'Maximum permitted'])
plt.xticks(rotation=90)
plt.show()
plt.savefig(os.path.join(working_directory, ''.join([id_evaluation, '_Flops', '.png'])), format='png')



