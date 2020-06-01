import os
import tensorflow as tf

from lib.data import shuffle_split_dataset
from lib.plot import plot_tf_metrics


def train_evaluate_model(
    model,
    encoded_dataset,
    checkpoint_name=None,
    batch_size=100,
    do_test=True,
    tensorboard=False,
    visualise_metrics=[],
    shuffle_dataset=True,
):

    (training_set, validation_set, test_set) = shuffle_split_dataset(
        encoded_dataset, do_shuffle=shuffle_dataset
    )
    print(f"test set shape: {test_set[0].shape}")

    if checkpoint_name:
        checkpoint_dir = f"./output/training_checkpoints/{checkpoint_name}"
        checkpoint_filepath = os.path.join(checkpoint_dir, "ckpt_{epoch}")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath, save_weights_only=True
        )
    else:
        checkpoint_callback = []

    if tensorboard:
        tensorboard_callback = [tf.keras.callbacks.TensorBoard(log_dir="./output")]
    else:
        tensorboard_callback = []

    callbacks = [*checkpoint_callback, *tensorboard_callback]

    training_history = model.fit(
        training_set[0],
        training_set[1],
        epochs=10,
        batch_size=batch_size,
        verbose=1,
        validation_data=(validation_set[0], validation_set[1]),
        callbacks=callbacks,
    )

    plot_tf_metrics(training_history, "loss")

    for m in visualise_metrics:
        plot_tf_metrics(training_history, m)

    if do_test:
        print(
            model.evaluate(test_set[0], test_set[1], verbose=1, batch_size=batch_size)
        )
