import sys

from pytorch_lightning.callbacks import ProgressBar
from tqdm import tqdm


# needed to fix tqdm bug
class LitProgressBar(ProgressBar):

    def init_train_tqdm(self):
        bar = tqdm(
            desc='Training',
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            ncols=200,
            file=sys.stdout,
            smoothing=0,
        )
        return bar

    def init_validation_tqdm(self):
        bar = tqdm(
            desc='Validating',
            initial=self.val_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            ncols=200,
            file=sys.stdout,
            smoothing=0,
        )
        return bar
