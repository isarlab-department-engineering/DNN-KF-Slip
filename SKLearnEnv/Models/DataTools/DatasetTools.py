from tqdm import tqdm
from sympy import *
import numpy as np
import abc
from SKLearnEnv.Models.DataTools import FileTools
from sklearn import model_selection
from pyDOE import lhs
from scipy.interpolate import interp1d
import random


class CsvManipulator:

    @staticmethod
    def split_dataset_supervised(x_data: np.ndarray, y_data: np.ndarray,
                                 ts_size: float, seed: int) -> (np.ndarray, np.ndarray, np.ndarray,np.ndarray):

        x_tr, x_te, y_tr, y_te = model_selection.train_test_split(x_data, y_data, test_size=ts_size, random_state=seed)

        return x_tr, x_te, y_tr, y_te

    @staticmethod
    def split_xy_supervised(dataset: np.ndarray, index_colum_y:np.array) -> (np.ndarray, np.ndarray):

        y_data = dataset[:, index_colum_y].copy()
        x_data = np.delete(dataset, index_colum_y, axis=1).copy()

        return x_data, y_data


###########################################################################
#                                                                         #
#       WARNING: STRATEGY (OR SOMETHING LIKE IT) DESIGN PATTERN           #
#       definition of algorithms for the generation of csv datasets       #
#                                                                         #
###########################################################################


class CsvRoadBuilder:

    """
    Define the interface of interest to clients.
    Maintain a reference to a Strategy object.
    """

    def __init__(self, strategy):
        self._strategy = strategy

    def build_csv(self):
        self._strategy.algorithm_interface()


class Method(metaclass=abc.ABCMeta):
    """
       Declare an interface common to all supported algorithms. Context
       uses this interface to call the algorithm defined by a
       ConcreteStrategy.
       """
    def __init__(self, working_directory: str, dest_filename: str):

        self.csv_file = FileTools.CsvFile(path=working_directory, filename=dest_filename)

    @abc.abstractmethod
    def algorithm_interface(self):
        pass


class SlidingWindowBestSlip(Method):

    """
        SlidingWindowBestSlip: Starting from a list of road models (curve families) iteratively for each model,
        each curve is sampled through a sliding window of fixed size (size passed as function argument).
        The elements present in the window at each step are shuffled.
        Each step corresponds to one line of the csv, and the size of the window determines the number
        of features: Len(win)+bestSlip(curve sampled at step n-th): from each the model, from each curve
        is extracted the value of best slip and linked to the values in the window at step n.
       """

    def __init__(self, working_directory: str, filename: str, roads_list: list, windows_size: int, shuffle_sample = False):
        self.windows = windows_size
        self.list_of_roads_mod = roads_list
        self.shuffle = shuffle_sample
        super(SlidingWindowBestSlip, self).__init__(working_directory, filename)

    def algorithm_interface(self):

        n_columns_input = self.windows * 2
        n_columns_output = 1
        column_total = n_columns_input + n_columns_output

        for ix_road in tqdm(range(0, len(self.list_of_roads_mod)), desc='Processing models'):
            model = self.list_of_roads_mod[ix_road]
            road_models = model.get_mu_roads()
            best_slips = model.get_best_slips()
            road_model_samp = model.get_mu_roads()
            slip_values_samp = model.get_lambda_dummy()

            # scorro ciascuna colonna
            for ix_column in tqdm(range(0, road_models.shape[1]), desc='roads of model:'):
                column_ix = road_model_samp[:, ix_column]
                best_slip_ix = best_slips[ix_column]
                # muovo la finestra per creare il dataset
                for ix_slid in range(0, len(column_ix) + 1 - self.windows):
                    slip_windowed = slip_values_samp[ix_slid:ix_slid + self.windows]
                    mu_windowed = column_ix[ix_slid:ix_slid + self.windows]
                    # each row of matrix contains  two columns: slip_x,mu_x
                    slip_mu_matrix = np.array((slip_windowed, mu_windowed)).T
                    if self.shuffle:
                        np.random.shuffle(slip_mu_matrix)
                    slip_mu_row = slip_mu_matrix.flatten('C')
                    # SHUFFLE
                    row = np.ones(column_total, dtype=object)
                    row[:] = np.nan
                    row[0:len(slip_mu_row)] = slip_mu_row
                    row[-1] = best_slip_ix.copy()
                    self.csv_file.append_row_to_csv(row)


###########################################################################
#                                                                         #
#      methods for the generation of coefficients used by road models     #
#                                                                         #
###########################################################################

class CoeffGenerator:

    """
      methods for the generation of coefficients used by road models
       """

    @staticmethod
    def burchkardt_three_sets_linspace(b1: type(()), b2: type(()), b3: type(()), linspace: int, n_curves_hidden: int,
                                       n_curves_validation: int) -> (list, list, list, list):
        """
         burchkardt_three_sets_linspace:
         starting from 3 tuples corresponding to the values (min, max) assumed by b1, b2, b3
         and from the number of desired points, will be generated three vectors containing the values
         included between the minimum and the maximum of beta. These 3 vectors are partitioned
         in other 3 sets of beta for the generation of 3 different scenarios: 'used,validation,hidden'.

           """

        b1_all = np.linspace(start=b1[0], stop=b1[1], num=linspace, endpoint=True)
        b2_all = np.linspace(start=b2[0], stop=b2[1], num=linspace, endpoint=True)
        b3_all = np.linspace(start=b3[0], stop=b3[1], num=linspace, endpoint=True)

        b1_index = list(np.arange(start=0, stop=len(b1_all)))
        b2_index = list(np.arange(start=0, stop=len(b2_all)))
        b3_index = list(np.arange(start=0, stop=len(b3_all)))

        n_elements_hidden = n_curves_hidden
        n_elements_valid = n_curves_validation

        # note: first and last element will be never removed from used betas-> LINSPACE EQ -2
        step_skip_drop = int((linspace) // (n_elements_hidden + n_elements_valid))


        b1_index_excluded = list(b1_index[0:len(b1_index):step_skip_drop])
        b2_index_excluded = list(b2_index[0:len(b2_index):step_skip_drop])
        b3_index_excluded = list(b3_index[0:len(b3_index):step_skip_drop])

        while len(b1_index_excluded) > (n_elements_hidden + n_elements_valid):
            b1_index_excluded.pop(int(len(b1_index_excluded)//2))
            b2_index_excluded.pop(int(len(b2_index_excluded) // 2))
            b3_index_excluded.pop(int(len(b3_index_excluded) // 2))

        b1_index_used = [e for e in b1_index if e not in b1_index_excluded]
        b2_index_used = [e for e in b2_index if e not in b2_index_excluded]
        b3_index_used = [e for e in b3_index if e not in b3_index_excluded]

        step_skip_hidden = int(len(b1_index_excluded) // n_elements_hidden)

        b1_index_hidden = list(b1_index_excluded[0:len(b1_index_excluded)-1:step_skip_hidden])
        b2_index_hidden = list(b2_index_excluded[0:len(b2_index_excluded)-1:step_skip_hidden])
        b3_index_hidden = list(b3_index_excluded[0:len(b3_index_excluded)-1:step_skip_hidden])

        b1_index_validation = [e for e in b1_index_excluded if e not in b1_index_hidden]
        b2_index_validation = [e for e in b2_index_excluded if e not in b2_index_hidden]
        b3_index_validation = [e for e in b3_index_excluded if e not in b3_index_hidden]

        b1_used = b1_all[b1_index_used]
        b2_used = b2_all[b2_index_used]
        b3_used = b3_all[b3_index_used]

        b1_validation = b1_all[b1_index_validation]
        b2_validation = b2_all[b2_index_validation]
        b3_validation = b3_all[b3_index_validation]

        b1_hidden = b1_all[b1_index_hidden]
        b2_hidden = b2_all[b2_index_hidden]
        b3_hidden = b3_all[b3_index_hidden]

        # Refactor v.1.0 model accept only list of tuples....
        b_all = tuple(map(tuple, np.asarray([b1_all, b2_all, b3_all]).transpose()))
        b_used = tuple(map(tuple, np.asarray([b1_used, b2_used, b3_used]).transpose()))
        b_validation = tuple(map(tuple, np.asarray([b1_validation, b2_validation, b3_validation]).transpose()))
        b_hidden = tuple(map(tuple, np.asarray([b1_hidden, b2_hidden, b3_hidden]).transpose()))

        return b_all, b_used, b_validation, b_hidden

    @staticmethod
    def burchkardt_three_sets_latin_hyper(b1: type(()), b2: type(()), b3: type(()), linspace: int,
                                          n_curves_hidden: int,
                                          n_curves_validation: int, seed: int = 1234) -> (list, list, list, list):
        """
               burchkardt_three_sets_linspace_latin_hyper:
               starting from 3 tuples corresponding to the values (min,  max) assumed by b1, b2, b3
               and from the number of points desired,

                 """

        lhd_used = lhs(3, samples=linspace)

        map_b1 = interp1d([0.0, 1.0], [b1[0], b1[1]])
        map_b2 = interp1d([0.0, 1.0], [b2[0], b2[1]])
        map_b3 = interp1d([0.0, 1.0], [b3[0], b3[1]])

        b1_all = b1_used = map_b1(lhd_used[:, 0])
        b2_all = b2_used = map_b2(lhd_used[:, 1])
        b3_all = b3_used = map_b3(lhd_used[:, 2])

        index_used = np.arange(start=0, stop=len(b1_all))

        # VALIDATION BETAS
        random.seed = seed
        index_val = np.asarray(random.sample(list(index_used), n_curves_validation))


        b1_validation = b1_used[index_val]
        b2_validation = b2_used[index_val]
        b3_validation = b3_used[index_val]


        b1_used = np.delete(b1_used, index_val)
        b2_used = np.delete(b2_used, index_val)
        b3_used = np.delete(b3_used, index_val)
        index_used = np.arange(start=0, stop=len(b1_used))

        # HIDDEN BETAS
        random.seed = seed
        index_hidden = np.asarray(random.sample(list(index_used), n_curves_hidden))

        b1_hidden = b1_used[index_hidden]
        b2_hidden = b2_used[index_hidden]
        b3_hidden = b3_used[index_hidden]

        b1_used = np.delete(b1_used, index_hidden)
        b2_used = np.delete(b2_used, index_hidden)
        b3_used = np.delete(b3_used, index_hidden)


        # Refactor v.1.0 model accept only list of tuples....
        b_all = tuple(map(tuple, np.asarray([b1_all, b2_all, b3_all]).transpose()))
        b_used = tuple(map(tuple, np.asarray([b1_used, b2_used, b3_used]).transpose()))
        b_validation = tuple(map(tuple, np.asarray([b1_validation, b2_validation, b3_validation]).transpose()))
        b_hidden = tuple(map(tuple, np.asarray([b1_hidden, b2_hidden, b3_hidden]).transpose()))

        return b_all, b_used, b_validation, b_hidden

    @staticmethod
    def pacejka_three_sets_linspace(b1: type(()), b2: type(()), b3: type(()), b4: type(()), linspace: int,
                                    n_curves_hidden: int, n_curves_validation: int) -> (list, list, list, list):
        """
         burchkardt_three_sets_linspace:
         starting from tuples corresponding to the values (min, max) assumed by b1, b2, b3, b4
         and from the number of desired points, will be generated three vectors containing the values
         included between the minimum and the maximum of beta. These 3 vectors are partitioned
         in other 3 sets of beta for the generation of 3 different scenarios: 'used,validation,hidden'.

           """

        b1_all = np.linspace(start=b1[0], stop=b1[1], num=linspace, endpoint=True)
        b2_all = np.linspace(start=b2[0], stop=b2[1], num=linspace, endpoint=True)
        b3_all = np.linspace(start=b3[0], stop=b3[1], num=linspace, endpoint=True)
        b4_all = np.linspace(start=b4[0], stop=b4[1], num=linspace, endpoint=True)

        b1_index = list(np.arange(start=0, stop=len(b1_all)))
        b2_index = list(np.arange(start=0, stop=len(b2_all)))
        b3_index = list(np.arange(start=0, stop=len(b3_all)))
        b4_index = list(np.arange(start=0, stop=len(b4_all)))

        n_elements_hidden = n_curves_hidden
        n_elements_valid = n_curves_validation

        # note: first and last element will be never removed from used betas-> LINSPACE EQ -2
        step_skip_drop = int((linspace) // (n_elements_hidden + n_elements_valid))

        b1_index_excluded = list(b1_index[0:len(b1_index):step_skip_drop])
        b2_index_excluded = list(b2_index[0:len(b2_index):step_skip_drop])
        b3_index_excluded = list(b3_index[0:len(b3_index):step_skip_drop])
        b4_index_excluded = list(b4_index[0:len(b4_index):step_skip_drop])

        while len(b1_index_excluded) > (n_elements_hidden + n_elements_valid):
            b1_index_excluded.pop(int(len(b1_index_excluded) // 2))
            b2_index_excluded.pop(int(len(b2_index_excluded) // 2))
            b3_index_excluded.pop(int(len(b3_index_excluded) // 2))
            b4_index_excluded.pop(int(len(b4_index_excluded) // 2))

        b1_index_used = [e for e in b1_index if e not in b1_index_excluded]
        b2_index_used = [e for e in b2_index if e not in b2_index_excluded]
        b3_index_used = [e for e in b3_index if e not in b3_index_excluded]
        b4_index_used = [e for e in b4_index if e not in b4_index_excluded]

        step_skip_hidden = int(len(b1_index_excluded) // n_elements_hidden)

        b1_index_hidden = list(b1_index_excluded[0:len(b1_index_excluded) - 1:step_skip_hidden])
        b2_index_hidden = list(b2_index_excluded[0:len(b2_index_excluded) - 1:step_skip_hidden])
        b3_index_hidden = list(b3_index_excluded[0:len(b3_index_excluded) - 1:step_skip_hidden])
        b4_index_hidden = list(b4_index_excluded[0:len(b4_index_excluded) - 1:step_skip_hidden])

        b1_index_validation = [e for e in b1_index_excluded if e not in b1_index_hidden]
        b2_index_validation = [e for e in b2_index_excluded if e not in b2_index_hidden]
        b3_index_validation = [e for e in b3_index_excluded if e not in b3_index_hidden]
        b4_index_validation = [e for e in b4_index_excluded if e not in b4_index_hidden]

        b1_used = b1_all[b1_index_used]
        b2_used = b2_all[b2_index_used]
        b3_used = b3_all[b3_index_used]
        b4_used = b4_all[b4_index_used]

        b1_validation = b1_all[b1_index_validation]
        b2_validation = b2_all[b2_index_validation]
        b3_validation = b3_all[b3_index_validation]
        b4_validation = b4_all[b4_index_validation]

        b1_hidden = b1_all[b1_index_hidden]
        b2_hidden = b2_all[b2_index_hidden]
        b3_hidden = b3_all[b3_index_hidden]
        b4_hidden = b4_all[b4_index_hidden]

        # Refactor v.1.0 model accept only list of tuples....
        b_all = tuple(map(tuple, np.asarray([b1_all, b2_all, b3_all, b4_all]).transpose()))
        b_used = tuple(map(tuple, np.asarray([b1_used, b2_used, b3_used, b4_used]).transpose()))
        b_validation = tuple(
            map(tuple, np.asarray([b1_validation, b2_validation, b3_validation, b4_validation]).transpose()))
        b_hidden = tuple(map(tuple, np.asarray([b1_hidden, b2_hidden, b3_hidden, b4_hidden]).transpose()))

        return b_all, b_used, b_validation, b_hidden

    @staticmethod
    def pacejka_three_sets_latin_hyper(b1: type(()), b2: type(()), b3: type(()), b4: type(()), linspace: int,
                                       n_curves_hidden: int,
                                       n_curves_validation: int, seed: int = 1234) -> (list, list, list, list):

        """
        burchkardt_three_sets_linspace_latin_hyper:
        starting from 3 tuples corresponding to the values (min,  max) assumed by b1, b2, b3
        and from the number of points desired,

        """
        lhd_used = lhs(4, samples=linspace)

        map_b1 = interp1d([0.0, 1.0], [b1[0], b1[1]])
        map_b2 = interp1d([0.0, 1.0], [b2[0], b2[1]])
        map_b3 = interp1d([0.0, 1.0], [b3[0], b3[1]])
        map_b4 = interp1d([0.0, 1.0], [b4[0], b4[1]])

        b1_all = b1_used = map_b1(lhd_used[:, 0])
        b2_all = b2_used = map_b2(lhd_used[:, 1])
        b3_all = b3_used = map_b3(lhd_used[:, 2])
        b4_all = b4_used = map_b4(lhd_used[:, 3])

        index_used = np.arange(start=0, stop=len(b1_all))

        # VALIDATION BETAS
        random.seed = seed
        index_val = np.asarray(random.sample(list(index_used), n_curves_validation))

        b1_validation = b1_used[index_val]
        b2_validation = b2_used[index_val]
        b3_validation = b3_used[index_val]
        b4_validation = b4_used[index_val]

        b1_used = np.delete(b1_used, index_val)
        b2_used = np.delete(b2_used, index_val)
        b3_used = np.delete(b3_used, index_val)
        b4_used = np.delete(b4_used, index_val)
        index_used = np.arange(start=0, stop=len(b1_used))

        # HIDDEN BETAS
        random.seed = seed
        index_hidden = np.asarray(random.sample(list(index_used), n_curves_hidden))

        b1_hidden = b1_used[index_hidden]
        b2_hidden = b2_used[index_hidden]
        b3_hidden = b3_used[index_hidden]
        b4_hidden = b4_used[index_hidden]

        b1_used = np.delete(b1_used, index_hidden)
        b2_used = np.delete(b2_used, index_hidden)
        b3_used = np.delete(b3_used, index_hidden)
        b4_used = np.delete(b4_used, index_hidden)

        # Refactor v.1.0 model accept only list of tuples....
        b_all = tuple(map(tuple, np.asarray([b1_all, b2_all, b3_all, b4_all]).transpose()))
        b_used = tuple(map(tuple, np.asarray([b1_used, b2_used, b3_used, b4_used]).transpose()))
        b_validation = tuple(
            map(tuple, np.asarray([b1_validation, b2_validation, b3_validation, b4_validation]).transpose()))
        b_hidden = tuple(map(tuple, np.asarray([b1_hidden, b2_hidden, b3_hidden, b4_hidden]).transpose()))

        return b_all, b_used, b_validation, b_hidden
