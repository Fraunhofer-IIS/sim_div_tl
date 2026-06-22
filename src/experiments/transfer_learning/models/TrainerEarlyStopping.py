from typing import List, Optional, Union
import mxnet as mx
from gluonts.mx.trainer.callback import Callback, CallbackList
from gluonts.mx.trainer.model_averaging import SelectNBestMean, ModelAveraging
from gluonts.mx.trainer.learning_rate_scheduler import LearningRateReduction
from gluonts.mx.trainer.learning_rate_scheduler import (
    MetricAttentiveScheduler,
    Patience,
    Objective,
)
from gluonts.mx.trainer._base import Trainer
from typing_extensions import Literal


class LearningRateReduction_early_stopping(LearningRateReduction):
    def __init__(
        self,
        objective: Literal["min", "max"],
        patience: int,
        base_lr: float = 0.01,
        decay_factor: float = 0.5,
        min_lr: float = 0.0,
        max_num_decays: int = 5,
    ) -> None:

        super().__init__(objective=objective, patience=patience)

        assert (
            0 < decay_factor < 1
        ), "The value of `decay_factor` should be in the (0, 1) range"
        assert patience >= 0, "The value of `patience` should be >= 0"
        assert (
            0 <= min_lr <= base_lr
        ), "The value of `min_lr` should be >= 0 and <= base_lr"

        self.lr_scheduler = MetricAttentiveScheduler(
            patience=Patience(patience, Objective.from_str(objective)),
            learning_rate=base_lr,
            decay_factor=decay_factor,
            min_learning_rate=min_lr,
            max_num_decays=max_num_decays,
        )


class TrainerEarlyStopping(Trainer):
    def __init__(
        self,
        ctx: Optional[mx.Context] = None,
        epochs: int = 100,
        num_batches_per_epoch: int = 50,
        learning_rate: float = 1e-3,
        clip_gradient: float = 10.0,
        weight_decay: float = 1e-8,
        max_num_decays: int = 10,
        init: Union[str, mx.initializer.Initializer] = "xavier",
        hybridize: bool = True,
        callbacks: Optional[List[Callback]] = None,
        add_default_callbacks: bool = True,
    ) -> None:

        super().__init__(
            ctx=ctx,
            epochs=epochs,
            num_batches_per_epoch=num_batches_per_epoch,
            learning_rate=learning_rate,
            clip_gradient=clip_gradient,
            weight_decay=weight_decay,
            init=init,
            hybridize=hybridize,
            callbacks=callbacks,
            add_default_callbacks=add_default_callbacks,
        )

        self.max_num_decays = max_num_decays

        # Make sure callbacks is list -- they are assigned to `self.callbacks`
        # below
        callbacks = callbacks or []

        # The following is done for backwards compatibility. For future
        # versions, add the default callbacks as default arg
        if add_default_callbacks:
            if not any(isinstance(callback, ModelAveraging) for callback in callbacks):
                callbacks.append(
                    ModelAveraging(avg_strategy=SelectNBestMean(num_models=1))
                )

            if not any(
                isinstance(callback, LearningRateReduction) for callback in callbacks
            ):
                callbacks.append(
                    LearningRateReduction_early_stopping(
                        base_lr=learning_rate,
                        patience=10,
                        objective="min",
                        max_num_decays=max_num_decays,
                    )
                )

        self.callbacks = CallbackList(callbacks)
