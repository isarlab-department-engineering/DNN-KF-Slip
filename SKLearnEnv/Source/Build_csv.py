
#!/usr/bin/python
# -*- coding: utf-8 -*-
from SKLearnEnv.Models import Configuration as Cfg
from tqdm import tqdm
import SKLearnEnv.Models.DataTools.DatasetTools as Ds
import datetime
from SKLearnEnv.Models.Roads import RoadModels as Rm, RoadModifiers as Rmm
from SKLearnEnv.Models.DataTools import FileTools as Ft
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import numpy as np
import copy

###########################################################################
#                                                                         #
#          CONFIGURATIONS FOR ROAD SCENARIOS AND DATASETS                 #
#                                                                         #
###########################################################################


# loading minimal parser for the purpose of this script: testing the generalization capabilities ---> USED
csv_parser = Cfg.CsvParser().parser
args = csv_parser.parse_args()

startslip = args.startslip
stopslip = args.stopslip
n_points = args.npointslip
beta_space_step = args.betaspace
dest_path = args.path
# id_filename = uuid.uuid1().__str__()[:8]
id_filename ='f98fa0b8'
noise_val = args.noise
n_curves_validation = args.ncurvvalid
n_curves_hidden = args.ncurvhid
current_time = datetime.datetime.now()
windows_dim = args.windows
include_nonoise_train = args.includenonoise
desampling_used_curves= args.subsampling
referenceIncluded = args.includeReference
shuffle_window = args.shuffleWindow


#
#
working_directory = Ft.Folder(path=dest_path, foldername=id_filename).get_path()

###########################################################################
#                                                                         #
#            BETA CONFIGURATION FOR ROAD SCENARIOS                        #
#                                                                         #
###########################################################################

# BURCKHARDT COEFF. GENERATION

# Reference Beta
betas_BU_dry_ref = (1.28, 23.9, 0.52)
betas_BU_cobb_ref = (1.3713, 6.4565, 0.6691)
betas_BU_wet_ref = (0.857, 33.822, 0.347)
betas_BU_snow_ref = (0.19, 94.93, 0.06)

# # Beta range min to max - with cobb
# b1 = (0.19, 2)
# b2 = (94.93, 6.0)
# b3 = (0.06, 0.7)

# Beta range min to max - with cobb
b1 = (0.01 , 2)
b2 = (100.00, 6.0)
b3 = (0.01, 0.7)



# Generation of Betas from the min-max interval and splitting
beta_burc_all, beta_burc_used, beta_burc_validation, beta_burc_hidden = \
    Ds.CoeffGenerator.burchkardt_three_sets_linspace(b1, b2, b3, beta_space_step,
                                                     n_curves_hidden,
                                                     n_curves_validation)


# Generation of Betas from the min-max interval and splitting - 3x3 CUBE (B1,B2,B3)
beta_burc_all_hyp, beta_burc_used_hyp, beta_burc_validation_hyp, beta_burc_hidden_hyp = \
    Ds.CoeffGenerator.burchkardt_three_sets_latin_hyper(b1, b2, b3, beta_space_step,
                                                        n_curves_hidden,
                                                        n_curves_validation)


# PACEJKA COEFF. GENERATION

betas_PAC_dry_ref = (10, 1.9, 1, 0.97)
betas_PAC_wet_ref = (12, 2.3, 0.82, 1)
betas_PAC_ice_ref = (5, 2, 0.3, 1)

# # set all betas from min to max - no cobb - pacecjka
B = (5, 12)
C = (2.4, 1.9)
D = (0.3, 1)
E = (1, 0.97)

# Generation of Betas from the min-max interval and splitting
beta_pacej_all, beta_pacej_used, beta_pacej_validation, beta_pacej_hidden = \
    Ds.CoeffGenerator.pacejka_three_sets_linspace(B, C, D, E, beta_space_step,
                                                  n_curves_hidden,
                                                  n_curves_validation)


# Generation of Betas from the min-max interval and splitting  - 4x4 CUBE (B,C,D,E)
beta_pacej_all_hyp, beta_pacej_used_hyp, beta_pacej_validation_hyp, beta_pacej_hidden_hyp = \
    Ds.CoeffGenerator.pacejka_three_sets_latin_hyper(B, C, D, E, beta_space_step,
                                                     n_curves_hidden,
                                                     n_curves_validation)


# Add original reference curves to the used betas, Note only temp.
if referenceIncluded:
    beta_burc_used = beta_burc_used + (betas_BU_dry_ref,)
    beta_burc_used = beta_burc_used + (betas_BU_cobb_ref,)
    beta_burc_used = beta_burc_used + (betas_BU_wet_ref,)
    beta_burc_used = beta_burc_used + (betas_BU_snow_ref,)




###########################################################################
#                                                                         #
#                            ROAD SCENARIOS                               #
#                                                                         #
###########################################################################

# list of all scenarios and reference roads
roads_list_complete_mixed = []
roads_list_used_mixed = []
roads_list_validation_mixed = []
roads_list_hidden_mixed = []
roads_list_reference = []


# ALL ROADS
roads_list_complete_mixed.append(Rm.Burckhardt(list_betas=beta_burc_all, n_points_slip=n_points,
                                               start_slip=startslip, stop_slip=stopslip,
                                               description='BUR_DIAG_ALL'))
#
# roads_list_complete_mixed.append(Rm.Burckhardt(list_betas=beta_burc_all_hyp, n_points_slip=n_points,
#                                                start_slip=startslip, stop_slip=stopslip,
#                                                description='BUR_LATIN_ALL'))

# roads_list_complete_mixed.append(Rm.Pacejka(list_betas=beta_pacej_all, n_points_slip=n_points,
#                                             start_slip=startslip, stop_slip=stopslip,
#                                             description='Pacejka diagonal, all betas'))

# roads_list_complete_mixed.append(Rm.Pacejka(list_betas=beta_pacej_all_hyp, n_points_slip=n_points,
#                                             start_slip=startslip, stop_slip=stopslip,
#                                             description='Pacejka latin-hyper, all betas'))


# USED ROADS

if desampling_used_curves:
    # Calc subsampling factor
    factor_subsampling = np.int(np.floor(n_points/windows_dim))
    # Create multiple version of the same used road with different slip points: i*window_size, i=[1:factor_subsampling]
    for i in tqdm(range(1, factor_subsampling+1), desc='Subsampling slip points'):
        slip_point_subs = i*windows_dim

        roads_list_used_mixed.append(Rm.Burckhardt(list_betas=beta_burc_used, n_points_slip=slip_point_subs,
                                                   start_slip=startslip, stop_slip=stopslip,
                                                   description='BUR_DIAG_USED'))
        #
        # roads_list_used_mixed.append(Rm.Burckhardt(list_betas=beta_burc_used_hyp, n_points_slip=n_points,
        #                                            start_slip=startslip, stop_slip=stopslip,
        #                                            description='BUR_LATIN_USED'))

        # roads_list_used_mixed.append(Rm.Pacejka(list_betas=beta_pacej_used, n_points_slip=n_points,
        #                                         start_slip=startslip, stop_slip=stopslip,
        #                                         description='roads_scenario_used'))

        # roads_list_used_mixed.append(Rm.Pacejka(list_betas=beta_pacej_used_hyp, n_points_slip=n_points,
        #                                         start_slip=startslip, stop_slip=stopslip,
        #                                         description='roads_scenario_used'))

    # #  FOCUS ZOOM SOLO PER PAPER MED, MODIFICA TEMPORANEA ---------------------------------------------------
    # for i in tqdm(range(1, factor_subsampling + 1), desc='Subsampling slip points'):
    #     slip_point_subs = i * windows_dim
    #
    #     roads_list_used_mixed.append(Rm.Burckhardt(list_betas=beta_burc_used, n_points_slip=slip_point_subs,
    #                                                start_slip=startslip, stop_slip=0.08,
    #                                                description='BUR_DIAG_USED'))
    # # Then add the curves with the original number of slip points
    # roads_list_used_mixed.append(Rm.Burckhardt(list_betas=beta_burc_used, n_points_slip=n_points,
    #                                            start_slip=startslip, stop_slip=0.08,
    #                                            description='BUR_DIAG_USED'))
    # # Then add the curves with the original number of slip points
    # # -------------------------------------------------------------------------------------------------------

    roads_list_used_mixed.append(Rm.Burckhardt(list_betas=beta_burc_used, n_points_slip=n_points,
                                               start_slip=startslip, stop_slip=stopslip,
                                               description='BUR_DIAG_USED'))
    #
    # roads_list_used_mixed.append(Rm.Burckhardt(list_betas=beta_burc_used_hyp, n_points_slip=n_points,
    #                                            start_slip=startslip, stop_slip=stopslip,
    #                                            description='BUR_LATIN_USED'))

    # roads_list_used_mixed.append(Rm.Pacejka(list_betas=beta_pacej_used, n_points_slip=n_points,
    #                                         start_slip=startslip, stop_slip=stopslip,
    #                                         description='roads_scenario_used'))

    # roads_list_used_mixed.append(Rm.Pacejka(list_betas=beta_pacej_used_hyp, n_points_slip=n_points,
    #                                         start_slip=startslip, stop_slip=stopslip,
    #                                         description='roads_scenario_used'))

else:
    roads_list_used_mixed.append(Rm.Burckhardt(list_betas=beta_burc_used, n_points_slip=n_points,
                                               start_slip=startslip, stop_slip=stopslip,
                                               description='BUR_DIAG_USED'))
    #
    # roads_list_used_mixed.append(Rm.Burckhardt(list_betas=beta_burc_used_hyp, n_points_slip=n_points,
    #                                            start_slip=startslip, stop_slip=stopslip,
    #                                            description='BUR_LATIN_USED'))

    # roads_list_used_mixed.append(Rm.Pacejka(list_betas=beta_pacej_used, n_points_slip=n_points,
    #                                         start_slip=startslip, stop_slip=stopslip,
    #                                         description='roads_scenario_used'))


    # roads_list_used_mixed.append(Rm.Pacejka(list_betas=beta_pacej_used_hyp, n_points_slip=n_points,
    #                                         start_slip=startslip, stop_slip=stopslip,
    #                                         description='roads_scenario_used'))


# VALIDATION ROADS
roads_list_validation_mixed.append(Rm.Burckhardt(list_betas=beta_burc_validation,
                                                 n_points_slip=n_points, start_slip=startslip, stop_slip=stopslip,
                                                 description='BUR_DIAG_VALID'))
#
# roads_list_validation_mixed.append(Rm.Burckhardt(list_betas=beta_burc_validation_hyp,
#                                                  n_points_slip=n_points, start_slip=startslip, stop_slip=stopslip,
#                                                  description='BUR_LATIN_VALID'))

# roads_list_validation_mixed.append(Rm.Pacejka(list_betas=beta_pacej_validation,
#                                               n_points_slip=n_points, start_slip=startslip, stop_slip=stopslip,
#                                               description='roads_scenario_validation'))
#
# roads_list_validation_mixed.append(Rm.Pacejka(list_betas=beta_pacej_hidden_hyp,
#                                               n_points_slip=n_points, start_slip=startslip, stop_slip=stopslip,
#                                               description='roads_scenario_hidden'))


# HIDDEN ROADS
roads_list_hidden_mixed.append(Rm.Burckhardt(list_betas=beta_burc_hidden,
                                             n_points_slip=n_points, start_slip=startslip, stop_slip=stopslip,
                                             description='BUR_DIAG_HIDD'))
#
# roads_list_hidden_mixed.append(Rm.Burckhardt(list_betas=beta_burc_hidden_hyp,
#                                              n_points_slip=n_points, start_slip=startslip, stop_slip=stopslip,
#                                              description='BUR_LATIN_HIDD'))

# roads_list_hidden_mixed.append(Rm.Pacejka(list_betas=beta_pacej_hidden,
#                                           n_points_slip=n_points, start_slip=startslip, stop_slip=stopslip,
#                                           description='roads_scenario_hidden'))
#
# roads_list_hidden_mixed.append(Rm.Pacejka(list_betas=beta_pacej_validation_hyp,
#                                           n_points_slip=n_points, start_slip=startslip, stop_slip=stopslip,
#                                           description='roads_scenario_validation'))

# DEFAULT ROADS
# DRY
roads_list_reference.append(Rm.Burckhardt([betas_BU_dry_ref], n_points_slip=n_points,
                                         start_slip=startslip, stop_slip=stopslip,
                                         description='BUR_DRY'))

# road_list_reference.append(Rm.Pacejka([betas_PAC_dry_ref], n_points_slip=n_points,
#                                       start_slip=startslip, stop_slip=stopslip,
#                                       description='PAJ_DRY'))

# COBBLESTONE  - only Burckhardt
roads_list_reference.append(Rm.Burckhardt([betas_BU_cobb_ref], n_points_slip=n_points,
                                         start_slip=startslip, stop_slip=stopslip,
                                         description='BUR_COBB'))

# WET
roads_list_reference.append(Rm.Burckhardt([betas_BU_wet_ref], n_points_slip=n_points,
                                         start_slip=startslip, stop_slip=stopslip,
                                         description='BUR_WET'))

# road_list_reference.append(Rm.Pacejka([(betas_PAC_wet_ref], n_points_slip=n_points,
#                                       start_slip=startslip, stop_slip=stopslip,
#                                       description='PAJ_WET'))

# SNOW/ICE
roads_list_reference.append(Rm.Burckhardt([betas_BU_snow_ref], n_points_slip=n_points,
                                         start_slip=startslip, stop_slip=stopslip,
                                         description='BUR_SNOW'))
#
# road_list_reference.append(Rm.Pacejka([betas_PAC_ice_ref], n_points_slip=n_points,
#                                       start_slip=startslip, stop_slip=stopslip,
#                                       description='PAJ_ICE'))


###########################################################################
#                                                                         #
#                           NOISE GENERATION                              #
#                                                                         #
###########################################################################

if include_nonoise_train:

    # Create a copy of train roads without noise to add later to the noised version.
    roads_list_used_mixed_nonoise = []
    for i in range(len(roads_list_used_mixed)):
        roads_list_used_mixed_nonoise.append(copy.deepcopy(roads_list_used_mixed[i]))
    # Adding noise to the models
    noiser = Rmm.WhiteNoiseAdder(noise_val)
    noiser.alter_model(roads_list_used_mixed)
    noiser.alter_model(roads_list_validation_mixed)
    noiser.alter_model(roads_list_hidden_mixed)
    noiser.alter_model(roads_list_reference)

    # Add the non-noised version of training sample
    for i in range(len(roads_list_used_mixed_nonoise)):
        roads_list_used_mixed.append(copy.deepcopy(roads_list_used_mixed_nonoise[i]))

else:
    # Adding noise to the models
    noiser = Rmm.WhiteNoiseAdder(noise_val)
    noiser.alter_model(roads_list_used_mixed)
    noiser.alter_model(roads_list_validation_mixed)
    noiser.alter_model(roads_list_hidden_mixed)
    noiser.alter_model(roads_list_reference)




###########################################################################
#                                                                         #
#                           CSV GENERATION                                #
#                                                                         #
###########################################################################


# USED CSV
used_dataset_builder_strategy = Ds.SlidingWindowBestSlip(working_directory=working_directory,
                                                         filename=''.join([id_filename, '_used']),
                                                         roads_list=roads_list_used_mixed,
                                                         windows_size=windows_dim, shuffle_sample=shuffle_window)
csv_maker = Ds.CsvRoadBuilder(used_dataset_builder_strategy)
csv_maker.build_csv()

# VALIDATION CSV
validation_dataset_builder_strategy = Ds.SlidingWindowBestSlip(working_directory=working_directory,
                                                               filename=''.join([id_filename, '_val']),
                                                               roads_list=roads_list_validation_mixed,
                                                               windows_size=windows_dim, shuffle_sample=shuffle_window)
csv_maker = Ds.CsvRoadBuilder(validation_dataset_builder_strategy)
csv_maker.build_csv()

# HIDDEN CSV
hidden_dataset_builder_strategy = Ds.SlidingWindowBestSlip(working_directory=working_directory,
                                                           filename=''.join([id_filename, '_hid']),
                                                           roads_list=roads_list_hidden_mixed,
                                                           windows_size=windows_dim, shuffle_sample=shuffle_window)
csv_maker = Ds.CsvRoadBuilder(hidden_dataset_builder_strategy)
csv_maker.build_csv()

# REFERENCE_CSV
reference_dataset_builder_strategy = Ds.SlidingWindowBestSlip(working_directory=working_directory,
                                                           filename=''.join([id_filename, '_ref']),
                                                           roads_list=roads_list_reference,
                                                           windows_size=windows_dim, shuffle_sample=False)
csv_maker = Ds.CsvRoadBuilder(reference_dataset_builder_strategy)
csv_maker.build_csv()



###########################################################################
#                                                                         #
#                           SAVING  INFORMATIONS                          #
#                                                                         #
###########################################################################

txt_writer = Ft.FileTxt(path=working_directory, filename=id_filename)
txt_writer.append_row_to_txt('-------------DATE OF CREATION:-------------\n %s\n' % current_time)

txt_writer.append_row_to_txt('\n-------------COMMON CONFIGURATIONS:-------------\n\n')
configurations = list(args.__dict__.items())
for ix in range(0, len(configurations)):
    configuration = configurations[ix]
    txt_writer.append_row_to_txt(''.join([str(configuration[0]), ' : ', str(configuration[1]), '\n']))

txt_writer.append_row_to_txt('\n-------------CUSTOM CONFIGURATIONS:--------------\n')
txt_writer.append_row_to_txt('\nBETAS(Min,Max)\n')
txt_writer.append_row_to_txt(''.join(['b1:', str(b1), '\n', 'b2:', str(b2), '\n', 'b3:', str(b3), '\n']))
txt_writer.append_row_to_txt('Generation of Beta Method->\n')
txt_writer.append_row_to_txt(str(Ds.CoeffGenerator.burchkardt_three_sets_linspace.__doc__)+'\n\n')
txt_writer.append_row_to_txt(str(Ds.CoeffGenerator.burchkardt_three_sets_latin_hyper.__doc__)+'\n')


txt_writer.append_row_to_txt('\n\n--------------ROAD SCENARIOS-------------\n')
txt_writer.append_row_to_txt('\n--> USED:\n')
txt_writer.append_row_to_txt('\n\t N° scenarios:' + str(len(roads_list_used_mixed)))
for road in roads_list_used_mixed:
    txt_writer.append_row_to_txt('\n\t\t Coeff. Model: ' + str(road.model_name))
    txt_writer.append_row_to_txt('\n\t\t   Annotation: ' + str(road.model_annotation))
    txt_writer.append_row_to_txt('\n\t\t     # Curves: ' + str(road.get_n_curves()))

txt_writer.append_row_to_txt('\n--> VALIDATION:\n')
txt_writer.append_row_to_txt('\n\t N° scenarios:' + str(len(roads_list_validation_mixed)))
for road in roads_list_validation_mixed:
    txt_writer.append_row_to_txt('\n\t\t Coeff. Model: ' + str(road.model_name))
    txt_writer.append_row_to_txt('\n\t\t   Annotation: ' + str(road.model_annotation))
    txt_writer.append_row_to_txt('\n\t\t     # Curves: ' + str(road.get_n_curves()))

txt_writer.append_row_to_txt('\n--> HIDDEN:\n')
txt_writer.append_row_to_txt('\n\t N° scenarios:' + str(len(roads_list_hidden_mixed)))
for road in roads_list_hidden_mixed:
    txt_writer.append_row_to_txt('\n\t\t Coeff. Model: ' + str(road.model_name))
    txt_writer.append_row_to_txt('\n\t\t   Annotation: ' + str(road.model_annotation))
    txt_writer.append_row_to_txt('\n\t\t     # Curves: ' + str(road.get_n_curves()))

txt_writer.append_row_to_txt('\n\n----CSV BUILD CONFIGURATIONS:----\n')

txt_writer.append_row_to_txt('Builder type->\n')
txt_writer.append_row_to_txt(str(Ds.SlidingWindowBestSlip.__doc__) + '\n')

txt_writer.append_row_to_txt('Csv: ' + str(used_dataset_builder_strategy.csv_file.get_filename()) + '-->\n')
txt_writer.append_row_to_txt('\t\t\tRows: ' + str(used_dataset_builder_strategy.csv_file.get_number_of_rows()) + '\n')
txt_writer.append_row_to_txt(
    '\t\t\tColumns: ' + str(used_dataset_builder_strategy.csv_file.get_number_of_columns()) + '\n')

txt_writer.append_row_to_txt('Csv: ' + str(validation_dataset_builder_strategy.csv_file.get_filename()) + '-->\n')
txt_writer.append_row_to_txt(
    '\t\t\tRows: ' + str(validation_dataset_builder_strategy.csv_file.get_number_of_rows()) + '\n')
txt_writer.append_row_to_txt(
    '\t\t\tColumns: ' + str(validation_dataset_builder_strategy.csv_file.get_number_of_columns()) + '\n')

txt_writer.append_row_to_txt('Csv: ' + str(hidden_dataset_builder_strategy.csv_file.get_filename()) + '-->\n')
txt_writer.append_row_to_txt('\t\t\tRows: ' + str(hidden_dataset_builder_strategy.csv_file.get_number_of_rows()) + '\n')
txt_writer.append_row_to_txt(
    '\t\t\tColumns: ' + str(hidden_dataset_builder_strategy.csv_file.get_number_of_columns()) + '\n')



###########################################################################
#                                                                         #
#                           DISPLAY IMAGES                                #
#                                                                         #
###########################################################################

# All Roads Generated
plt.figure(1)
plt.title(''.join(['Generated roads scenarios: ALL -  ID : ', id_filename]))
h1 = [plt.plot(road.get_lambda_dummy(), road.get_mu_roads(), color=(0.2, 0.54, 1.0), label=road.model_annotation) for
        road in roads_list_used_mixed]
h2 = [plt.plot(road.get_lambda_dummy(), road.get_mu_roads(), color=(0.94, 0.61, 0.08), label=road.model_annotation) for
        road in roads_list_validation_mixed]
h3 = [plt.plot(road.get_lambda_dummy(), road.get_mu_roads(), color=(0.45, 0.67, 0.5), label=road.model_annotation) for
        road in roads_list_hidden_mixed]
plt.xlabel('$\lambda$')
plt.ylabel('$\mu$')
used_patch = mpatches.Patch(color=(0.2, 0.54, 1.0), label='Training Roads')
val_patch = mpatches.Patch(color=(0.94, 0.61, 0.08), label='Validation Roads')
hidd_patch = mpatches.Patch(color=(0.45, 0.67, 0.5), label='Hidden Roads')
plt.legend(handles=[used_patch, val_patch, hidd_patch])
plt.grid()
plt.show()
plt.savefig(os.path.join(working_directory, ''.join([id_filename, '_', 'ROADS_ALL', '.png'])), format='png')

# Roads used for training
plt.figure(2)
title = ''.join(['Generated roads scenarios: USED -  ID : ', id_filename])
plt.title(title)
h = plt.gca().get_children()
h[0] = [plt.plot(road.get_lambda_dummy(), road.get_mu_roads(), color=(0.2, 0.54, 1.0), label=road.model_annotation) for
        road in roads_list_used_mixed]
plt.xlabel('$\lambda$')
plt.ylabel('$\mu$')
used_patch = mpatches.Patch(color=(0.2, 0.54, 1.0), label='Used Roads')
plt.legend(handles=[used_patch])
plt.grid()
plt.savefig(os.path.join(working_directory, ''.join([id_filename, '_', 'ROADS_USED', '.png'])), format='png')

# Roads used for validation
plt.figure(3)
title = ''.join(['Generated roads scenarios: VALIDATION -  ID : ', id_filename])
plt.title(title)
h = plt.gca().get_children()
h[0] = [plt.plot(road.get_lambda_dummy(), road.get_mu_roads(), color=(0.94, 0.61, 0.08), label=road.model_annotation)
        for road in roads_list_validation_mixed]
plt.xlabel('$\lambda$')
plt.ylabel('$\mu$')
val_patch = mpatches.Patch(color=(0.94, 0.61, 0.08), label='Validation Roads')
plt.legend(handles=[val_patch])
plt.grid()
plt.savefig(os.path.join(working_directory, ''.join([id_filename, '_', 'ROADS_VALIDATION', '.png'])), format='png')

# Roads used for testing
plt.figure(4)
title = ''.join(['Generated roads scenarios: HIDDEN -  ID : ', id_filename])
plt.title(title)
h = plt.gca().get_children()
h[0] = [plt.plot(road.get_lambda_dummy(), road.get_mu_roads(), color=(0.45, 0.67, 0.5), label=road.model_annotation) for
        road in roads_list_hidden_mixed]
plt.xlabel('$\lambda$')
plt.ylabel('$\mu$')
hidd_patch = mpatches.Patch(color=(0.45, 0.67, 0.5), label='Hidden Roads')
plt.legend(handles=[hidd_patch])
plt.grid()
plt.savefig(os.path.join(working_directory, ''.join([id_filename, '_', 'ROADS_HIDDEN', '.png'])), format='png')

# Roads used as reference and used for ultimate test
plt.figure(5)
title = ''.join(['Generated roads scenarios: REFERENCE -  ID : ', id_filename])
plt.title(title)
h = plt.gca().get_children()
h[0] = [plt.plot(road.get_lambda_dummy(), road.get_mu_roads(), color=(0.85, 0.36, 0.30), label=road.model_annotation)
        for road in roads_list_reference]
plt.xlabel('$\lambda$')
plt.ylabel('$\mu$')
ref_patch = mpatches.Patch(color=(0.85, 0.36, 0.30), label='Reference Roads')
plt.legend(handles=[ref_patch])
plt.grid()
plt.savefig(os.path.join(working_directory, ''.join([id_filename, '_', 'ROADS_REFERENCE', '.png'])), format='png')


# Betas space representation for Burckhardt's model
matrix_betas_diag = np.asarray(beta_burc_all)
b1 = matrix_betas_diag[:, 0]
b2 = matrix_betas_diag[:, 1]
b3 = matrix_betas_diag[:, 2]

matrix_betas_hyp = np.asarray(beta_burc_all_hyp)
b1_h = matrix_betas_hyp[:, 0]
b2_h = matrix_betas_hyp[:, 1]
b3_h = matrix_betas_hyp[:, 2]

matrix_betas_reference = np.asarray([betas_BU_dry_ref, betas_BU_cobb_ref, betas_BU_wet_ref, betas_BU_snow_ref])
b1_r = matrix_betas_reference[:, 0]
b2_r = matrix_betas_reference[:, 1]
b3_r = matrix_betas_reference[:, 2]

fig_orig = plt.figure(40)
ax1 = fig_orig.add_subplot(1, 1, 1, projection='3d')
ax1.view_init(azim=-161)
ax1.view_init(elev=15)
ax1.scatter3D(b1,
              b2,
              b3, 'blue', label='Diagonal', alpha = 1)

ax1.scatter3D(b1_h,
              b2_h,
              b3_h, 'red', label='Latin Hypercube', alpha = 1)

ax1.scatter3D(b1_r,
              b2_r,
              b3_r, 'green', label='Burchkardt', alpha = 1)

ax1.set_xlabel(r'$\beta$1', fontsize=12)
ax1.set_ylabel(r'$\beta$2', fontsize=12)
ax1.set_zlabel(r'$\beta$3', fontsize=12)
ax1.title.set_text(r'Burckhardt ($\beta$1,$\beta$2,$\beta$3) space rapresentation')
ax1.legend()
plt.savefig(os.path.join(working_directory, ''.join([id_filename, '_', 'BURCKHARDT_SPACE', '.png'])), format='png')