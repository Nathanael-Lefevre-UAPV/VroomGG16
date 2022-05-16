import os
import torch
import numpy as np

from tqdm import tqdm
from imageio import imread

from VroomGG16 import VroomGG16
from roadsimulator.models.utils import get_datasets


def get_images(paths, n_images=1000):

    if isinstance(paths, str):
        paths = [paths]

    images = []
    labels = []

    n = 0
    for path in paths:
        if n > n_images: break
        print(path)
        for image_file in tqdm(os.listdir(path)):
            if n > n_images: break
            if '.jpg' not in image_file: continue
            try:
                img = imread(os.path.join(path, image_file))
                itc = image_file[:-4].split('_')
                lbl = [float(itc[3]), float(itc[5])]
                if img is not None:
                    images.append(img[:, :])
                    labels.append(lbl)
                    n += 1
            except Exception as e:
                pass

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

if __name__ == "__main__":
    best_model = VroomGG16(None, None,
                          input_dim=40,
                          hidden_dim=16,
                          output_dim=2,
                          n_layers=1,
                          drop_prob=0)
    best_model.load_state_dict(torch.load(os.path.dirname(__file__) + "/models/" + "Model.torch"))
    best_model.eval()

    image, label = get_images("/Users/nathanael.l/SynchroDir/Etudes/Cours et TP/M1 Informatique CMI (2021-2022)/Projet master/tensorflow_first_test/model", n_images=1)
    print("image", image)
    print("label", label)
    print("_" * 100)

    print("inf", best_model(torch.tensor(image).float()))

    print("hi")

