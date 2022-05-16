import tensorflow as tf
import tensorflow_datasets as tfds
import time

import os
from tensorflow import keras

print("Tensorflow version", tf.version.VERSION)


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


def train_model():

    time_start = time.time()

    '''
    try:
        # Disable all GPUS
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
    #'''


    print("TensorFlow version:", tf.__version__)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    tf.config.list_physical_devices('GPU')

    '''
    configuration = tf.compat.v1.ConfigProto(device_count={"GPU": 0})
    session = tf.compat.v1.Session(config=configuration)
    #'''


    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    batch_size = 512
    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                               activation='relu'),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3),
                               activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        #   tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        #   tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy'],
    )
    model.fit(
        ds_train,
        epochs=12,
        validation_data=ds_test,
    )

    os.makedirs("./checkpoints", exist_ok=True)

    checkpoint = tf.train.Checkpoint(model)
    checkpoint.save('./checkpoints/my_checkpoint')

    time_end = time.time()
    time_elapsed = time_end - time_start

    print(f'Temps d\'exécution : {time_elapsed:.4}s')
    print("batch_size:", batch_size)



def load_and_test_model():
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    batch_size = 64


    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)


    # Create a new model instance
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                               activation='relu'),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3),
                               activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        #   tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        #   tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy'],
    )

    # Restore the weights
    checkpoint = tf.train.Checkpoint(model)
    checkpoint.restore('./checkpoints/my_checkpoint-1')

    model(ds_train[0])



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import tensorflow as tf
    import PIL as pl
    import scipy as sc

    print(sc.__version__)
    print("eee")


    print(pl.__version__)
    print("eee")
    print(tf.__version__)
    #train_model()
    #load_and_test_model()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
