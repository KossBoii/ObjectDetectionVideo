import os
import pytorch_lightning
import pytorchvideo.data
import torch.utils.data
import pytorchvideo as pv
import torch 
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip
)

class DatasetModule(pytorch_lightning.LightningDataModule):
    def __init__(self, dataset_path, clip_duration=2, batch_size=8, num_workers=8):
        super().__init__()

        # Dataset configuration
        _DATA_PATH = dataset_path
        _CLIP_DURATION = clip_duration  # Duration of sampled clip for each video
        _BATCH_SIZE = batch_size
        _NUM_WORKERS = num_workers  # Number of parallel processes fetching data

    def train_dataloader(self):
        """
        Create the train partition from the list of video labels in {self._DATA_PATH}/train
        """

        # transformation
        train_transform = Compose(
            [
                ApplyTransformToKey(
                    key='video',
                    transform=Compose(
                        [
                            UniformTemporalSubsample(8),
                            Lambda(lambda x: x / 255.0),
                            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                            RandomShortSideScale(min_size=256, max_size=320),
                            RandomCrop(244),
                            RandomHorizontalFlip(p=0.5),
                        ]
                    )
                )
            ]
        )

        train_dataset = pv.data.Kinetics(
            data_path=os.path.join(self._DATA_PATH, "train"),
            clip_sampler=pv.data.make_clip_sampler("random", self._CLIP_DURATION),
            decode_audio=False,
            transform=train_transform
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
        )

    def val_dataloader(self):
        """
        Create the validation partition from the list of video labels in {self._DATA_PATH}/val
        """
        val_dataset = pv.data.Kinetics(
            data_path=os.path.join(self._DATA_PATH, "val"),
            clip_sampler=pv.data.make_clip_sampler("uniform", self._CLIP_DURATION),
            decode_audio=False,
        )
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
        )

