from abc import abstractmethod
import torch
import torch.nn as nn
from tqdm import tqdm

from moviad.models.training_args import TrainingArgs

class VADModel(nn.Module):

    def train_epoch(
        self, epoch, train_dataloader, training_args: TrainingArgs
    ):
        avg_batch_loss = 0

        # train the model
        for batch in tqdm(train_dataloader):
            avg_batch_loss += self.train_step(batch, training_args)

        avg_batch_loss /= len(train_dataloader)
        return avg_batch_loss

    @torch.no_grad()
    def predict(self, dataloader, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw anomaly maps and scores of every image of the dataloader, on the CPU, in dataloader order.

        A batch is the images alone (a train split), (image, label, mask, path) (a test split) or
        (image, label, mask, path, foreground); the foreground masks, if present, go to the model as
        self(images, foreground=foreground).

        Returns:
            (N, 1, H, W) anomaly maps, not normalized, and (N,) anomaly scores.
        """
        self.eval()
        maps, scores = [], []
        for batch in tqdm(dataloader, desc="Predict"):
            if isinstance(batch, torch.Tensor):  # a train split: the images alone
                batch = (batch,)
            kwargs = {"foreground": batch[4].to(device)} if len(batch) > 4 else {}
            batch_maps, batch_scores = self(batch[0].to(device), **kwargs)
            maps.append(batch_maps.cpu())
            scores.append(batch_scores.cpu())
        return torch.cat(maps), torch.cat(scores)

    @abstractmethod
    def train_step(self, batch: torch.Tensor, training_args: TrainingArgs): ...

    @abstractmethod
    def train_chunk(self, train_dataloader: torch.utils.data.DataLoader, training_args: TrainingArgs): 
        pass

    def reset_model(self):
        pass

    def save(self, save_path: str):
        """Save the model state dictionary to ``save_path``."""
        torch.save(self.state_dict(), save_path)

    def save_model(self, save_path: str):
        """Compatibility alias for :meth:`save`."""
        self.save(save_path)

    def get_model_size(self):
        pass
