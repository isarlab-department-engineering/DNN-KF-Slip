import argparse
import os

# FOLDER'S PATHS
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT_DIR, 'Data', '../../Data/Dataset')
MLM_PATH = os.path.join(ROOT_DIR, 'Data', '../../Data/MLModels')
MLM_ARCHIVE_PATH = os.path.join(ROOT_DIR, 'Data', 'Archive', '../../Data/MLModels')
EVAL_PATH = os.path.join(ROOT_DIR, 'Data', 'Evaluations')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Parser for csv Building
class CsvParser:

    def __init__(self):

        self.parser = argparse.ArgumentParser(
            description="parser for the Csv Builder")

        self.parser.add_argument('--path', type=str, required=False, default=DATASET_DIR,
                                 metavar='PATH',
                                 help='Save path for the .csv file')

        self.parser.add_argument('--betaspace', type=int, required=True,
                                 default=3, metavar='ROAD',
                                 help='Number of betas generated (Burckardt or Pacejka coeff)')

        self.parser.add_argument('--npointslip', type=int, required=True,
                                 default=100, metavar='ROAD',
                                 help='the step size of the slip: used to generate the slips values')

        self.parser.add_argument('--startslip', type=float, required=False,
                                 default=0.0, metavar='ROAD',
                                 help='The start value of the slip value for the road')

        self.parser.add_argument('--stopslip', type=float, required=False,
                                 default=1.0, metavar='ROAD',
                                 help='The start value of the slip value for the road')

        self.parser.add_argument('--subsampling', type=str2bool, required=False,
                                 default=False, metavar='ROAD',
                                 help='Activate subsampling slip points')

        self.parser.add_argument('--windows', type=int, required=True,
                                 metavar='ROAD',
                                 help='the sliding windows used to create the features')

        self.parser.add_argument('--focusmin', type=float, required=False,
                                 default=None, metavar='ROAD',
                                 help='The min value of focus range')

        self.parser.add_argument('--focusmax', type=float, required=False,
                                 default=None, metavar='ROAD',
                                 help='The max value of fucus range')

        self.parser.add_argument('--noise', type=float, required=True, metavar='ROADMOD',
                                 help='Amplitude of AWG applied to all roads')

        self.parser.add_argument('--ncurvvalid', type=int, required=True,
                                metavar='ROAD',
                                help='number of beta used for the validation file.csv')

        self.parser.add_argument('--ncurvhid', type=int, required=True,
                                metavar='ROAD',
                                help='number of betas used for the hidden file.csv')

        self.parser.add_argument('--includenonoise', type=str2bool, required=False, default=False,
                                 metavar='ROAD',
                                 help='Include the original curves in the training dataset')
        self.parser.add_argument('--includeReference', type=str2bool, required=False, default=False,
                                 metavar='ROAD',
                                 help='Include the reference curves in the training dataset')

        self.parser.add_argument('--shuffleWindow', type=str2bool, required=True,
                                 metavar='ROAD',
                                 help='Shuffle the elements retrieved by the window during the csv building phase. '
                                      'Note: not the rows but the elements of the row')



    def __get__(self, instance, owner):

        return self.parser


# Parser for MLP generation
class MlParser:

    def __init__(self):

        self.parser = argparse.ArgumentParser(
            description="parser for the Machine Learning algorithms")

        self.parser.add_argument('--DatasetID', type=str, required=True,
                                 metavar='ID',
                                 help='Id of the datasets to be chosen')

    def __get__(self, instance, owner):
        return self.parser


# Parser for MLP evaluation
class EvalParser:

    def __init__(self):

        self.parser = argparse.ArgumentParser(
            description="parser for the Evaluation script")

        self.parser.add_argument('--DatasetID', type=str, required=True,
                                 metavar='ID',
                                 help='Id of the dataset to be chosen for the evaluation')

        self.parser.add_argument('--ModelsID', '--list', required=True,
                                 metavar='ID', default=[],  action='append',
                                 help='Ids of the models compared', type=str)

        self.parser.add_argument('--DevMaxFlops', type=int, required=False,
                                 metavar='PARAM', default=5 * 10 ** 6,
                                 help='Max Flops of the device')

        self.parser.add_argument('--DevHz', type=int, required=False,
                                 metavar='PARAM', default=200,
                                 help='Sample Frequency used')

    def __get__(self, instance, owner):
        return self.parser