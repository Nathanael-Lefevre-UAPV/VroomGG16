from ImageDataset import ImageDataset
from road_simulator.roadsimulator.models.utils import get_datasets
from model.VroomGG16 import VroomGG16

train_X, train_Y, val_X, val_Y, _, _ = get_datasets(['newAngle_propre/2022_03_18_22_14','dataset_propre/2022_03_18_18_04'], n_images=30871+22649)


if __name__ == "__main__":
    #print(len(train_X))
    #print(val_X.shape)
    #input()
    model = VroomGG16(train_loader=ImageDataset(train_X, train_Y),
                      valid_loader=ImageDataset(val_X, val_Y),
                      input_dim=40,
                      hidden_dim=16,
                      output_dim=2,
                      n_layers=1,
                      drop_prob=0)

    # train(n_epochs)
    # model.test()
    # for epoch in range(1, model.n_epochs + 1):
    model.fit(model.n_epochs)
