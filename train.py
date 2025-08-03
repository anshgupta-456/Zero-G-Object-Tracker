
import argparse
from ultralytics import YOLO
import os
EPOCHS = 20
IMG_SIZE = 416
MOSAIC = 0.3
MIXUP = 0.1
OPTIMIZER = 'AdamW'
MOMENTUM = 0.2
LR0 = 0.001
LRF = 0.01
SINGLE_CLS = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=IMG_SIZE, help='Image size')
    parser.add_argument('--mosaic', type=float, default=MOSAIC, help='Mosaic augmentation')
    parser.add_argument('--mixup', type=float, default=MIXUP, help='Mixup augmentation')
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER, help='Optimizer')
    parser.add_argument('--momentum', type=float, default=MOMENTUM, help='Momentum')
    parser.add_argument('--lr0', type=float, default=LR0, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=LRF, help='Final learning rate')
    parser.add_argument('--single_cls', type=bool, default=SINGLE_CLS, help='Single class training')

    args = parser.parse_args()
    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)

    model = YOLO(os.path.join(this_dir, "yolov8s.pt"))

    results = model.train(
        data=os.path.join(this_dir, "yolo_params.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        device='cpu',  
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        single_cls=args.single_cls,
        mosaic=args.mosaic,
        mixup=args.mixup,
        verbose=True
    )
