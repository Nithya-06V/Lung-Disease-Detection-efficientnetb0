import json
import shutil

import train_model as tm


def main():
    tm.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_dir, validation_dir, test_dir = tm.prepare_dataset_if_needed()
    raw_train, raw_validation, raw_test, class_names = tm.raw_datasets(train_dir, validation_dir, test_dir)

    with tm.CLASS_NAMES_PATH.open("w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)

    spec = tm.MODEL_SPECS["EfficientNetB0"]
    result = tm.train_one_model("EfficientNetB0", spec, raw_train, raw_validation, raw_test, class_names)

    model_file = tm.MODEL_DIR / spec["filename"]
    backup_file = tm.MODEL_DIR / "lung_disease_model.keras"
    if model_file.exists():
        shutil.copy2(model_file, backup_file)

    print(f"EfficientNetB0 model saved to: {model_file}")
    print(f"Backup copy saved to: {backup_file}")
    print(f"Test accuracy: {result['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
