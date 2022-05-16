from torch.utils.data import DataLoader, Dataset
import torch


class ImageDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])


if __name__ == "__main__":
    from roadsimulator.models.utils import get_datasets
    train_X, train_Y, val_X, val_Y, _, _ = get_datasets('my_dataset', n_images=100)
    sd = ImageDataset(train_X, train_Y)

    #print(sd[0][0].size())
