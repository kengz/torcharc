import os

import lightning as L
import torch
import yaml
from lightning.pytorch.loggers import TensorBoardLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

import torcharc


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 32, data_dir: str = "./data/"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.batch_size = batch_size

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)


class MNISTClassifier(L.LightningModule):
    def __init__(self, model_spec_path: str):
        super().__init__()
        spec = yaml.safe_load(open(model_spec_path))
        self.model = torcharc.build(spec)
        self.lr = 1e-3
        self.accuracy = Accuracy(task="multiclass", num_classes=10, top_k=1)
        # set this for log_graph and reporting params
        self.example_input_array = torch.rand(1, 1, 28, 28)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        acc = self.accuracy(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        acc = self.accuracy(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc}


if __name__ == "__main__":
    dm = MNISTDataModule()
    arc = os.getenv("ARC", "conv2d")  # conv2d, mlp, rnn
    model = MNISTClassifier(f"torcharc/example/spec/mnist/{arc}.yaml")
    # launch tensorboard with `tensorboard --logdir ./tb_logs`
    logger = TensorBoardLogger("tb_logs", name=arc, log_graph=True)
    trainer = L.Trainer(max_epochs=1, logger=logger)
    trainer.fit(model, dm)
