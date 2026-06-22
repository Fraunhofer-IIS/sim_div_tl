import sys, os
from .TrainerEarlyStopping import TrainerEarlyStopping
from . import BaseModel

sys.path.append(os.path.abspath("."))
from gluonts.mx.model.tft import TemporalFusionTransformerEstimator
from gluonts.dataset.common import FileDataset
from gluonts.model.predictor import Predictor
from typing import List, Dict, Any
import numpy as np
from gluonts.model.forecast import SampleForecast, QuantileForecast
from typing import Iterable, Union, Tuple
import pandas as pd
from gluonts.model.forecast import Forecast
from gluonts.dataset.common import Dataset
from pathlib import Path
from gluonts.core.serde import dump_json
from gluonts.core import fqname_for
import gluonts
import json


class TFT(BaseModel):
    """
    TFT using the estimator from gluonts.
    """

    def __init__(
        self,
        context_length,
        forecast_horizon,
        epochs,
        freq,
        num_batches_per_epoch=100,
        max_num_decays: int = 10,
    ):
        self.name = type(self).__name__
        super().__init__(self.name, forecast_horizon)

        self.forecast_horizon = forecast_horizon
        self.epochs = epochs
        self.freq = freq
        self.num_batches_per_epoch = num_batches_per_epoch

        self.estimator = TemporalFusionTransformerEstimator(
            freq=self.freq,
            prediction_length=self.forecast_horizon,
            context_length=context_length,
            trainer=TrainerEarlyStopping(
                add_default_callbacks=True,
                callbacks=None,
                clip_gradient=10.0,
                ctx=None,
                epochs=self.epochs,
                hybridize=True,
                init="xavier",
                learning_rate=0.001,
                num_batches_per_epoch=self.num_batches_per_epoch,  # Not all batches are looked at in each epoch
                weight_decay=1e-08,
                max_num_decays=max_num_decays,
            ),
        )

        self.predictor = None

    def train(
        self, dataset: FileDataset, save: False, path: str
    ) -> gluonts.mx.model.predictor.RepresentableBlockPredictor:

        self.predictor = self.estimator.train(dataset)

        def serialize(predictor, path: Path) -> None:
            # serialize Predictor type
            with (path / "type.txt").open("w") as fp:
                fp.write(fqname_for(self.__class__))
            with (path / "version.json").open("w") as fp:
                json.dump({"gluonts": gluonts.__version__}, fp)
            composite_path = path / f"predictor"
            os.makedirs(str(composite_path), exist_ok=True)  # exist_ok=True
            predictor.serialize(composite_path)

        if save:
            path = Path(path)
            os.makedirs(path, exist_ok=True)
            serialize(
                self.predictor,
                path=path,
            )

    def predict(
        self,
        dataset: FileDataset,
        num_samples: int,
    ) -> Tuple[Iterable[Forecast], Iterable[Union[pd.DataFrame, pd.Series]]]:
        """
        Return contains the forecasts and the target data.
        It is important that the samples in the Forecast have the correct shapes for:
            - Univariate forecast: (num_samples, prediction_length)
            - Multivariate forecast: (num_samples, prediction_length, target_dim)

        Parameters
        ----------
        dataset: FileDataset
            #TODO has to be dataset.test, so has to look like what?
        """

        forecasts = self.predictor.predict(dataset)
        q_fcsts: List[QuantileForecast] = list(forecasts)

        result: List[SampleForecast] = []

        for fc in q_fcsts:
            # 0.5-Quantil (Median) aus dem QuantileForecast holen
            median = fc.quantile(0.5)  # shape: (prediction_length[, target_dim])
            samples = np.expand_dims(
                median, 0
            )  # -> (1, prediction_length[, target_dim])

            # SampleForecast mit nur einem Sample erzeugen
            sf = SampleForecast(
                samples=samples,
                start_date=fc.start_date,
                item_id=fc.item_id,
                info=fc.info,
            )
            result.append(sf)

        return result

    def finetune_model(self, target_train: Dataset, path):
        """
        Fine tune the model on the given dataset.

        """
        predictor_path = Path(path) / "predictor"
        predictor = Predictor.deserialize(predictor_path)

        if self.epochs > 0:
            self.predictor = self.estimator.train_from(
                predictor=predictor,
                training_data=target_train,
            )
        else:
            self.predictor = predictor

        return self.predictor
