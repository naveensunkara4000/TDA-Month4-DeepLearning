import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
from tensorflow.keras.utils import to_categorical
import joblib

BASEDIR = os.path.dirname(os.path.abspath(__file__))
MODELDIR = os.path.join(BASEDIR, "../models")
OUTDIR = os.path.join(BASEDIR, "../outputs", "week12")

os.makedirs(MODELDIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

def load_data():
    print("Trying to load CIFAR-10 dataset...")
    try:
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

        y_train_cat = to_categorical(y_train, 10)
        y_test_cat = to_categorical(y_test, 10)

        print("CIFAR-10 loaded successfully.")
        print("Train shape:", x_train.shape, "Test shape:", x_test.shape)
        return (x_train, y_train_cat), (x_test, y_test_cat), y_train, y_test

    except Exception as e:
        print("Failed to load CIFAR-10 due to:", e)
        print("Using synthetic image data as fallback.")

        num_classes = 10
        num_train = 800
        num_test = 200

        x_train = np.random.rand(num_train, 32, 32, 3).astype("float32")
        x_test = np.random.rand(num_test, 32, 32, 3).astype("float32")

        y_train = np.random.randint(0, num_classes, size=(num_train,))
        y_test = np.random.randint(0, num_classes, size=(num_test,))

        y_train_cat = to_categorical(y_train, num_train if False else num_classes)
        y_test_cat = to_categorical(y_test, num_train if False else num_classes)

        print("Synthetic dataset created.")
        print("Train shape:", x_train.shape, "Test shape:", x_test.shape)

        # Reshape y_train, y_test to 1D
        y_train = np.array(y_train).reshape(-1)
        y_test = np.array(y_test).reshape(-1)
        return (x_train, y_train_cat), (x_test, y_test_cat), y_train, y_test

def build_model(input_shape=(32, 32, 3), num_classes=10):
    print("Building CNN model...")
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model

def plot_history(history):
    # Accuracy
    plt.figure()
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.title("CNN Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTDIR, "accuracy.png"))
    plt.close()

    # Loss
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("CNN Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTDIR, "loss.png"))
    plt.close()


def plot_sample_predictions(model, x_test, y_test_raw, class_names):
    # Take first 16 test samples
    x_samples = x_test[:16]

    # Flatten labels to 1D: supports shapes (N,), (N,1), (1,N)
    labels_flat = np.array(y_test_raw).flatten()[:len(x_samples)]

    # Get predictions
    preds = model.predict(x_samples)
    pred_classes = np.argmax(preds, axis=1)

    plt.figure(figsize=(10, 10))
    for i in range(len(x_samples)):
        plt.subplot(9, 9, i + 1)
        plt.imshow(x_samples[i])

        true_idx = int(labels_flat[i])
        pred_idx = int(pred_classes[i])

        # Safety: if index outside range, show as number
        true_label = class_names[true_idx] if 0 <= true_idx < len(class_names) else str(true_idx)
        pred_label = class_names[pred_idx] if 0 <= pred_idx < len(class_names) else str(pred_idx)

        plt.title(f"T: {true_label}\nP: {pred_label}", fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "sample_predictions.png"))
    plt.close()


def main():
    (x_train, y_train_cat), (x_test, y_test_cat), y_train_raw, y_test_raw = load_data()

    model = build_model(input_shape=x_train.shape[1:], num_classes=10)

    print("Training CNN...")
    history = model.fit(
        x_train,
        y_train_cat,
        epochs=10,
        batch_size=64,
        validation_split=0.1,
        verbose=1,
    )

    print("Evaluating CNN on test data...")
    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}, loss: {test_loss:.4f}")

    # Save model
    model_path = os.path.join(MODELDIR, "cnn_cifar10.h5")
    model.save(model_path)
    print("Model saved at:", model_path)

    # Save history
    joblib.dump(history.history, os.path.join(OUTDIR, "history.pkl"))

    # Save metrics
    with open(os.path.join(OUTDIR, "evaluation.txt"), "w") as f:
        f.write(f"Test accuracy: {test_acc:.4f}\n")
        f.write(f"Test loss: {test_loss:.4f}\n")

    # Plots
    plot_history(history)

    # Class names for CIFAR-10
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    plot_sample_predictions(model, x_test, y_test_raw, class_names)

    print("Week 12 complete! Check outputs in:", OUTDIR)


if __name__ == "__main__":
    main()
