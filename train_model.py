import json
import os
import random
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121, EfficientNetB0, ResNet50
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR.parent / "dataset"
MODEL_DIR = BASE_DIR / "model"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
COMPARISON_PATH = MODEL_DIR / "model_comparison.json"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
STAGE1_EPOCHS = int(os.getenv("STAGE1_EPOCHS", "5"))
STAGE2_EPOCHS = int(os.getenv("STAGE2_EPOCHS", "5"))

MODEL_SPECS = {
    "EfficientNetB0": {
        "builder": EfficientNetB0,
        "preprocess": efficientnet_preprocess,
        "filename": "efficientnetb0.keras",
    },
    "DenseNet121": {
        "builder": DenseNet121,
        "preprocess": densenet_preprocess,
        "filename": "densenet121.keras",
    },
    "ResNet50": {
        "builder": ResNet50,
        "preprocess": resnet_preprocess,
        "filename": "resnet50.keras",
    },
}


def _image_files(folder):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    return sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in extensions)


def prepare_dataset_if_needed():
    train_dir = DATASET_DIR / "train"
    validation_dir = DATASET_DIR / "validation"
    test_dir = DATASET_DIR / "test"
    if all(path.exists() for path in (train_dir, validation_dir, test_dir)):
        return train_dir, validation_dir, test_dir

    class_dirs = sorted(path for path in DATASET_DIR.iterdir() if path.is_dir())
    if not class_dirs:
        raise FileNotFoundError(f"No class folders found under {DATASET_DIR}.")
    for split_dir in (train_dir, validation_dir, test_dir):
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    for class_dir in class_dirs:
        images = _image_files(class_dir)
        if len(images) < 3:
            raise ValueError(f"Class '{class_dir.name}' needs at least three images.")
        rng.shuffle(images)
        train_end = max(1, int(len(images) * 0.70))
        validation_end = min(len(images) - 1, train_end + max(1, int(len(images) * 0.15)))
        split_files = {
            "train": images[:train_end],
            "validation": images[train_end:validation_end],
            "test": images[validation_end:],
        }
        for split_name, files in split_files.items():
            target_root = {"train": train_dir, "validation": validation_dir, "test": test_dir}[split_name]
            target_dir = target_root / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            for index, source in enumerate(files):
                shutil.copy2(source, target_dir / f"{index:06d}_{source.name}")
    return train_dir, validation_dir, test_dir


def raw_datasets(train_dir, validation_dir, test_dir):
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        label_mode="categorical", shuffle=True, seed=SEED,
    )
    class_names = list(train_dataset.class_names)
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        validation_dir, class_names=class_names, image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE, label_mode="categorical", shuffle=False,
    )
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir, class_names=class_names, image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE, label_mode="categorical", shuffle=False,
    )
    return train_dataset, validation_dataset, test_dataset, class_names


def model_datasets(train_dataset, validation_dataset, test_dataset, preprocess):
    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.03),
            layers.RandomZoom(0.08),
            layers.RandomTranslation(0.03, 0.03),
            layers.RandomContrast(0.08),
        ],
        name="medical_image_augmentation",
    )

    def augment_and_preprocess(images, labels):
        return preprocess(augmentation(images, training=True)), labels

    def preprocess_only(images, labels):
        return preprocess(images), labels

    autotune = tf.data.AUTOTUNE
    return (
        train_dataset.map(augment_and_preprocess, num_parallel_calls=autotune).prefetch(autotune),
        validation_dataset.map(preprocess_only, num_parallel_calls=autotune).prefetch(autotune),
        test_dataset.map(preprocess_only, num_parallel_calls=autotune).prefetch(autotune),
    )


def build_model(spec, num_classes):
    base_model = spec["builder"](include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3))
    base_model.trainable = False
    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs), base_model


def callbacks(checkpoint_path):
    return [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7),
        ModelCheckpoint(checkpoint_path, monitor="val_accuracy", mode="max", save_best_only=True),
    ]


def save_curves(model_name, history):
    epochs = range(1, len(history["accuracy"]) + 1)
    for metric, title, filename, ylabel in (
        ("accuracy", "Accuracy", "accuracy", "Accuracy"),
        ("loss", "Loss", "loss", "Loss"),
    ):
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history[metric], label=f"Training {metric}")
        plt.plot(epochs, history[f"val_{metric}"], label=f"Validation {metric}")
        plt.title(f"{model_name} {title}")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(MODEL_DIR / f"{model_name.lower()}_{filename}.png", dpi=160)
        plt.close()


def save_confusion_matrix(model_name, matrix, class_names):
    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for row in range(len(matrix)):
        for column in range(len(matrix[row])):
            plt.text(column, row, matrix[row][column], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / f"{model_name.lower()}_confusion_matrix.png", dpi=160)
    plt.close()


def train_one_model(model_name, spec, raw_train, raw_validation, raw_test, class_names):
    train_data, validation_data, test_data = model_datasets(raw_train, raw_validation, raw_test, spec["preprocess"])
    model, base_model = build_model(spec, len(class_names))
    checkpoint_path = MODEL_DIR / f"{model_name.lower()}_best.keras"
    started = time.perf_counter()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    first = model.fit(train_data, validation_data=validation_data, epochs=STAGE1_EPOCHS, callbacks=callbacks(checkpoint_path), verbose=1)

    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss="categorical_crossentropy", metrics=["accuracy"])
    second = model.fit(train_data, validation_data=validation_data, epochs=STAGE2_EPOCHS, callbacks=callbacks(MODEL_DIR / spec["filename"]), verbose=1)

    model_path = MODEL_DIR / spec["filename"]
    model.save(model_path)
    test_loss, test_accuracy = model.evaluate(test_data, verbose=0)
    probabilities = model.predict(test_data, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = np.concatenate([np.argmax(labels.numpy(), axis=1) for _, labels in test_data], axis=0)
    matrix = confusion_matrix(y_true, y_pred).tolist()
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    history = {key: first.history.get(key, []) + second.history.get(key, []) for key in set(first.history) | set(second.history)}
    save_curves(model_name, history)
    save_confusion_matrix(model_name, matrix, class_names)
    result = {
        "model": model_name,
        "model_file": spec["filename"],
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "training_time_seconds": round(time.perf_counter() - started, 2),
        "classification_report": report,
        "confusion_matrix": matrix,
    }
    with (MODEL_DIR / f"{model_name.lower()}_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    return result


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_dir, validation_dir, test_dir = prepare_dataset_if_needed()
    raw_train, raw_validation, raw_test, class_names = raw_datasets(train_dir, validation_dir, test_dir)
    with CLASS_NAMES_PATH.open("w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)

    results = []
    for model_name, spec in MODEL_SPECS.items():
        print(f"\n===== Training {model_name} =====")
        result = train_one_model(model_name, spec, raw_train, raw_validation, raw_test, class_names)
        results.append(result)
        print(f"{model_name} test accuracy: {result['test_accuracy']:.4f}")
        print(f"{model_name} precision: {result['precision_weighted']:.4f}")
        print(f"{model_name} recall: {result['recall_weighted']:.4f}")
        print(f"{model_name} F1-score: {result['f1_weighted']:.4f}")
        tf.keras.backend.clear_session()

    with COMPARISON_PATH.open("w", encoding="utf-8") as file:
        json.dump({"class_names": class_names, "models": results}, file, indent=2)
    shutil.copy2(MODEL_DIR / MODEL_SPECS["EfficientNetB0"]["filename"], MODEL_DIR / "lung_disease_model.keras")
    print(f"\nSaved comparison results to {COMPARISON_PATH}")
    print("EfficientNetB0 copied to model/lung_disease_model.keras for Flask deployment.")


if __name__ == "__main__":
    main()
