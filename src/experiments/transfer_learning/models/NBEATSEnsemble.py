import sys, os
from .TrainerEarlyStopping import TrainerEarlyStopping
from . import BaseModel

sys.path.append(os.path.abspath("."))
from gluonts.mx.model.n_beats import NBEATSEnsembleEstimator
from gluonts.mx.model.n_beats._ensemble import NBEATSEnsemblePredictor
from gluonts.dataset.common import FileDataset
from typing import Iterable, Union, Tuple
import pandas as pd
from gluonts.model.forecast import Forecast
from gluonts.dataset.common import Dataset
from pathlib import Path
from gluonts.core.serde import dump_json
from gluonts.core import fqname_for
import gluonts
import json


class NBEATSEnsemble(BaseModel):
    """
    NBEATSEnsemble using the estimator from gluonts.
    """

    def __init__(
        self,
        meta_context_length,
        forecast_horizon,
        meta_bagging_size,
        epochs,
        freq,
        widths,
        meta_loss_function=["sMAPE", "MASE", "MAPE"],
        num_batches_per_epoch=100,
        max_num_decays: int = 10,
        num_stacks: int = 30,
        num_blocks: Iterable[int] = [1],
        num_block_layers: Iterable[int] = [4],
        expansion_coefficient_lengths: Iterable[int] = [32],
        sharing: Iterable[bool] = [False],
        stack_types: Iterable[str] = ["G"],
    ):
        self.name = type(self).__name__  # 'NBeats_gluonts'
        super().__init__(self.name, forecast_horizon)

        self.forecast_horizon = forecast_horizon
        self.epochs = epochs
        self.freq = freq
        self.meta_loss_function = list(meta_loss_function)
        self.num_batches_per_epoch = num_batches_per_epoch
        self.num_stacks = num_stacks
        self.widths = widths
        self.num_blocks = num_blocks
        self.num_block_layers = num_block_layers
        self.expansion_coefficient_lengths = expansion_coefficient_lengths
        self.sharing = sharing
        self.stack_types = stack_types

        self.estimator = NBEATSEnsembleEstimator(
            freq=self.freq,
            prediction_length=self.forecast_horizon,
            meta_context_length=list(meta_context_length),
            meta_loss_function=self.meta_loss_function,
            meta_bagging_size=meta_bagging_size,
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
            # recommended values for stack type "generic"
            num_stacks=self.num_stacks,
            widths=[self.widths],
            num_blocks=[self.num_blocks],
            num_block_layers=[self.num_block_layers],
            expansion_coefficient_lengths=[self.expansion_coefficient_lengths],
            sharing=[self.sharing],
            stack_types=[self.stack_types],
        )

    def train(
        self, dataset: FileDataset, save: False, path: str
    ) -> NBEATSEnsemblePredictor:

        self.predictor = self.estimator.train(dataset)

        def serialize(predictors, prediction_length, path: Path) -> None:
            # serialize Predictor type
            with (path / "type.txt").open("w") as fp:
                fp.write(fqname_for(self.__class__))
            with (path / "version.json").open("w") as fp:
                json.dump({"gluonts": gluonts.__version__}, fp)
            # basically save each predictor in its own sub-folder
            num_digits = len(str(len(predictors)))
            for index, predictor in enumerate(predictors):
                composite_path = path / f"predictor_{str(index).zfill(num_digits)}"
                os.makedirs(str(composite_path), exist_ok=True)  # exist_ok=True
                predictor.serialize(composite_path)

            # serialize all remaining constructor parameters
            with (path / "parameters.json").open("w") as fp:
                parameters = dict(
                    prediction_length=prediction_length,
                    aggregation_method="median",
                    num_predictors=len(predictors),
                )
                print(dump_json(parameters), file=fp)

        if save:
            path = Path(path)
            os.makedirs(path, exist_ok=True)
            serialize(
                self.predictor.predictors,
                prediction_length=self.forecast_horizon,
                path=path,
            )

    def predict(
        self,
        dataset: FileDataset,
        num_samples: int,
        mode: str = "none",
        aggregation_method: str = "none",
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
        # NBEATSEnsemble doesn't give distribution so we need forecasts of individual models
        self.predictor.set_aggregation_method(aggregation_method)

        forecasts = self.predictor.predict(dataset, num_samples=num_samples)

        fcst_list = list(forecasts)
        return fcst_list

    def finetune_model(self, target_train: Dataset, path):
        """
        Fine tune the model on the given dataset.

        """
        predictor = NBEATSEnsemblePredictor.deserialize(Path(path))

        if self.epochs > 0:
            self.predictor = self.estimator.train_from(
                predictor=predictor, training_data=target_train
            )
        else:
            self.predictor = predictor

        return self.predictor
