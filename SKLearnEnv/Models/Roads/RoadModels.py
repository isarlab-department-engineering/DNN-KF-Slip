#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
|---------------------------------------------------------------------------|
|Class to generate road models, burckhardt or pacejka,                      |
|Define the skeleton of an algorithm in an operation, deferring some        |
|steps to subclasses.  certain  steps of an algorithm without changing      |
|the algorithm's structure. Francesco Crocetti @isarlab PG	                |
|                                                                           |
|---------------------------------------------------------------------------|
"""
from abc import abstractmethod, ABCMeta
import numpy as np
from sympy import *
import math


class RoadModel(metaclass=ABCMeta):
    """
    Define abstract primitive operations that concrete subclasses define
    to implement steps of an algorithm.
    Template Method lets subclasses redefine and  represents
    the mathematical models that will be implemented to generate t
    he friction curves (surface models) as a function of the slip coefficient.
    """

    @abstractmethod
    def __init__(self, n_points_slip: int, start_slip: float, stop_slip: float):
        self.n_points_slip = n_points_slip
        self.start_slip = start_slip
        self.stop_slip = stop_slip
        self._model_name = None
        self._model_annotations = None

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, value):
        self._model_name = value

    @property
    def model_annotation(self):
        return self._model_annotations

    @model_annotation.setter
    def model_annotation(self, value):
        self._model_annotations = value

    @abstractmethod
    def _get_lambda_dummy(self) -> np.array:
        raise NotImplementedError()

    @abstractmethod
    def _get_mu_roads(self) -> np.array:
        raise NotImplementedError()

    @abstractmethod
    def _get_best_slips(self) -> np.array:
        raise NotImplementedError()

    @abstractmethod
    def _get_betas(self) -> np.array:
        raise NotImplementedError()

    @abstractmethod
    def set_mu_roads(self, new_roads: np.ndarray):
        raise NotImplementedError()

    @abstractmethod
    def _get_n_roads(self) -> int:
        raise NotImplementedError()

    def get_lambda_dummy(self) -> np.array:
        return self._get_lambda_dummy()

    def get_mu_roads(self) -> np.array:
        return self._get_mu_roads()

    def get_best_slips(self) -> np.array:
        return self._get_best_slips()

    def get_n_point_slip(self) -> float:
        return self.n_points_slip

    def get_n_curves(self) -> int:
        return self._get_n_roads()

    def get_betas_used(self) -> np.ndarray:
        return self._get_betas()

    def get_start_slip(self) -> float:
        return self.start_slip

    def get_stop_slip(self) -> float:
        return self.stop_slip


class Burckhardt(RoadModel):
    """
        Description of Burckhardt:
        starting from the list of tuples (b1,b2,b3) burchkard roads model will be generated
        Note: the values for the common surfaces are listed below
        # vr_dry_bu = [1.28 23.9 0.52]
        # vr_cobble_bu = [1.3713 6.4565 0.6691]
        # vr_wet_bu = [0.857 33.822 0.347]
        # vr_snow_bu = [0.19 94.93 0.06]
            """

    def __init__(self, list_betas: list, n_points_slip: int, start_slip: float, stop_slip: float,
                 description: str, positive: bool = True, ):

        super(Burckhardt, self).__init__(n_points_slip, start_slip, stop_slip)

        self.model_name = "Burckhardt"
        self.model_annotation = description

        # Creazione vettore slip
        self.lambda_dummy = np.linspace(start=self.start_slip,
                                            stop=self.stop_slip, num=n_points_slip, endpoint=True)

        # Creazione della matrice contenente tutti i vettori Bn (1->3) utilizzati per generare le funzioni
        self.bn_burckhardt = np.array(np.zeros([len(list_betas), 3]))

        self.best_slips_values = np.ones(len(self.bn_burckhardt), dtype=float) * 0.2
        self.mu_burchardt_range = np.ones((len(self.lambda_dummy), len(self.bn_burckhardt)))
        for i in range(0, len(list_betas)):
            beta1 = self.bn_burckhardt[i, 0] = list_betas[i][0]
            beta2 = self.bn_burckhardt[i, 1] = list_betas[i][1]
            beta3 = self.bn_burckhardt[i, 2] = list_betas[i][2]
            self.mu_burchardt_range[:, i] = beta1 * (
                    1 - np.exp(-self.lambda_dummy * beta2)) - self.lambda_dummy * beta3
            self.best_slips_values[i] = (-1 / beta2) * math.log(beta3 / (beta1 * beta2))

        # check if some curves are minor than 0
        if positive:
            index = 0
            while index < self.mu_burchardt_range.shape[1]:
                if np.min(self.mu_burchardt_range[:, index]) < 0:
                    self.mu_burchardt_range = np.delete(self.mu_burchardt_range, index, axis=1)
                    self.best_slips_values = np.delete(self.best_slips_values, index)
                    self.bn_burckhardt = np.delete(self.bn_burckhardt, index, axis=0)
                    index = 0
                else:
                    index += 1

    def _get_lambda_dummy(self) -> np.array:
        return self.lambda_dummy

    def _get_mu_roads(self) -> np.array:
        return self.mu_burchardt_range

    def _get_best_slips(self) -> np.array:
        return self.best_slips_values

    def _get_betas(self) -> np.ndarray:
        return self.bn_burckhardt

    def _get_n_roads(self) -> int:
        return self.mu_burchardt_range.shape[1]

    def set_mu_roads(self, new_roads: np.ndarray):
        self.mu_burchardt_range = new_roads


class Pacejka(RoadModel):
    """
        Description of PacejkaLinear:
        starting from the beta vectors B,C,D,E  the curves are obtained
        through the exploration along the diagonal of the Hypercube space (B,C,D,E).
        Note: the main values for the common surfaces are listed below.
        # vr_dry_pa = [10   1.9  1      0.97]
        # vr_wet_pa = [12   2.3  0.82   1]
        # vr_snow_pa =[5    2    0.3    1]
       """

    def __init__(self, list_betas: list, n_points_slip: int, start_slip: float, stop_slip: float, description: str,
                 positive: bool = True):

        super(Pacejka, self).__init__(n_points_slip, start_slip, stop_slip)

        self.model_name = "Pacejka"
        self.model_annotation = description

        # Creazione vettore slip
        self.lambda_dummy = np.linspace(start=self.start_slip, stop=self.stop_slip, num=n_points_slip, endpoint=True)

        # Matrice contenente tutti i vettori B,C,D,E utilizzati per generare le funzioni
        self.bn_pacejka = np.array(np.zeros([len(list_betas), 4]))

        self.best_slips_values = np.ones(len(self.bn_pacejka), dtype=float) * 0.2
        self.mu_pacejka_range = np.ones((len(self.lambda_dummy), len(self.bn_pacejka)))
        for i in range(0, len(self.bn_pacejka)):

            b_i = self.bn_pacejka[i, 0] = list_betas[i][0]
            c_j = self.bn_pacejka[i, 1] = list_betas[i][1]
            d_k = self.bn_pacejka[i, 2] = list_betas[i][2]
            e_l = self.bn_pacejka[i, 3] = list_betas[i][3]
            self.mu_pacejka_range[:, i] = d_k * np.sin(c_j * np.arctan(b_i * self.lambda_dummy - e_l * (
                    b_i * self.lambda_dummy - np.arctan(b_i * self.lambda_dummy))))
            x = Symbol('x', real=True, positive=True)
            f = d_k * sin(c_j * atan(b_i * x - d_k * (b_i * x - atan(b_i * x))))
            df = diff(f, x)
            best_slip_val = nsolve(df, x, 0, verify=True, prec=3)
            # verify=false can help when the results are near to the roots and the verification fail
            self.best_slips_values[i] = best_slip_val

            # check if some curves are minor than 0
            if positive:
                index = 0
                while index < self.mu_pacejka_range.shape[1]:
                    if np.min(self.mu_pacejka_range[:, index]) < 0:
                        self.mu_pacejka_range = np.delete(self.mu_pacejka_range, index, axis=1)
                        self.best_slips_values = np.delete(self.best_slips_values, index)
                        self.bn_pacejka = np.delete(self.bn_pacejka, index, axis=0)
                        index = 0
                    else:
                        index += 1

    def _get_lambda_dummy(self) -> np.array:
        return self.lambda_dummy

    def _get_mu_roads(self) -> np.array:
        return self.mu_pacejka_range

    def _get_best_slips(self) -> np.array:
        return self.best_slips_values

    def _get_betas(self) -> np.ndarray:
        return self.bn_pacejka

    def _get_n_roads(self) -> int:
        return self.mu_pacejka_range.shape[1]

    def set_mu_roads(self, new_roads: np.ndarray):
        self.mu_pacejka_range = new_roads
