import abc
import numpy as np
from SKLearnEnv.Models.Roads import RoadModels as Rm
from tqdm import tqdm
import SKLearnEnv.Models.Exceptions as Ex


class RoadModifier(metaclass=abc.ABCMeta):
    """
    Define abstract primitive operations that concrete subclasses define
    to implement steps of an algorithm.
    Implement a template method defining the skeleton of an algorithm.
    The template method calls primitive operations as well as operations
    defined in AbstractClass or those of other objects.
    """

    @abc.abstractmethod
    def alter_model(self, road_model: Rm.RoadModel):
        pass


class WhiteNoiseAdder(RoadModifier):
    """
        Define abstract primitive operations that concrete subclasses define
        to implement steps of an algorithm.
        Implement a template method defining the skeleton of an algorithm.
        The template method calls primitive operations as well as operations
        defined in AbstractClass or those of other objects.
        """

    def __init__(self, noise_ampl: float):
        self.snr_db = noise_ampl

    def alter_model(self, road_model: list):
        for model in road_model:
            if not isinstance(model, Rm.RoadModel):
                raise Ex.RoadModel('Not a valid Road Model Elements!')
            else:
                model.set_mu_roads(
                    self._add_white_noise_snr(model.get_mu_roads(), self.snr_db, descr=model.model_annotation))

    @staticmethod
    def _add_white_noise_snr(roads: np.ndarray, noise_amplitude: float,
                             descr: str = '') -> np.ndarray:

        for ix_column in tqdm(range(0, roads.shape[1]), desc=''.join(['Noising: ', descr])):
            signal = roads[:, ix_column]
            mean_noise = 0
            white_noise_additive = np.random.normal(mean_noise, noise_amplitude, len(signal))
            # Noise up the original signal
            roads[:, ix_column] = signal.__add__(white_noise_additive)

        return roads
