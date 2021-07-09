from videoclassification import VideoClassificationLightningModule
from dataset import DatasetModule

def train():
    classification_module = VideoClassificationLightningModule()
    data_module = DatasetModule()
    trainer = pytorch_lightning.Trainer()
    trainer.fit(classification_module, data_module)