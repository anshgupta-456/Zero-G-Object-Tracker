from ultralytics import YOLO
from pathlib import Path
import cv2
import os
import yaml
import matplotlib.pyplot as plt



def predict_and_save(model, image_path, output_path, output_path_txt):
    results = model.predict(image_path, conf=0.5)
    result = results[0]

 
    img = result.plot()
    cv2.imwrite(str(output_path), img)


    with open(output_path_txt, 'w') as f:
        for box in result.boxes:
            cls_id = int(box.cls)
            x_center, y_center, width, height = box.xywh[0].tolist()
            f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")


if __name__ == '__main__':
    this_dir = Path(__file__).parent
    os.chdir(this_dir)

    
    with open(this_dir / 'yolo_params.yaml', 'r') as file:
        data = yaml.safe_load(file)
        if 'test' in data and data['test'] is not None:
            images_dir = Path(data['test']) / 'images'
        else:
            print("No test field found in yolo_params.yaml.")
            exit()

    if not images_dir.exists() or not images_dir.is_dir() or not any(images_dir.iterdir()):
        print(f"Invalid or empty test image directory: {images_dir}")
        exit()

  
    detect_path = this_dir / "runs" / "detect"
    train_folders = [f for f in os.listdir(detect_path) if os.path.isdir(detect_path / f) and f.startswith("train")]
    if not train_folders:
        raise ValueError("No training folders found under runs/detect/")
    
    if len(train_folders) > 1:
        print("Select a training folder:")
        for i, folder in enumerate(train_folders):
            print(f"{i}: {folder}")
        choice = input("Enter number: ").strip()
        idx = int(choice) if choice.isdigit() else 0
    else:
        idx = 0

    model_path = detect_path / train_folders[idx] / "weights" / "best.pt"
    model = YOLO(model_path)


    output_dir = this_dir / "predictions"
    images_output_dir = output_dir / 'images'
    labels_output_dir = output_dir / 'labels'
    for d in [images_output_dir, labels_output_dir]:
        d.mkdir(parents=True, exist_ok=True)

    for img_path in images_dir.glob('*'):
        if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
            continue
        output_path_img = images_output_dir / img_path.name
        output_path_txt = labels_output_dir / img_path.with_suffix('.txt').name
        predict_and_save(model, img_path, output_path_img, output_path_txt)

    print(f"\n Predictions saved in: {images_output_dir}")
    print(f" Labels saved in:      {labels_output_dir}")

    print("\n Running model evaluation on test set...")
    metrics = model.val(data=this_dir / 'yolo_params.yaml', split="test")

    
    if hasattr(metrics, 'confusion_matrix') and metrics.confusion_matrix is not None:
        conf_matrix_path = output_dir / "confusion_matrix.png"
        try:
            metrics.confusion_matrix.plot(save_dir=output_dir)  
            print(f" Confusion matrix saved at: {conf_matrix_path}")
        except Exception as e:
            print(f"Failed to save confusion matrix: {e}")
    else:
        print("Confusion matrix not available.")
