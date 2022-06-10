from ImageDataset import ImageDataset
from road_simulator.roadsimulator.models.utils import get_datasets
from model.VroomGG16 import VroomGG16
import torch

train_X, train_Y, val_X, val_Y, _, _ = get_datasets('dataset/dataset_propre/2022_03_18_18_04', n_images=30)#30871)#+22649)


if __name__ == "__main__":
    device = "mps" if torch.has_mps else "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu"
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VroomGG16(train_loader=ImageDataset(train_X, train_Y),
                      valid_loader=ImageDataset(val_X, val_Y),
                      input_dim=40,
                      hidden_dim=16,
                      output_dim=2,
                      n_layers=1,
                      drop_prob=0,
                      device=device)

    model.fit(model.n_epochs)
