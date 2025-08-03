
import os
import cv2


class YoloVisualizer:
    MODE_TRAIN = 0
    MODE_VAL = 1

    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder
        classes_file = os.path.join(dataset_folder, "classes.txt")

        with open(classes_file, "r") as f:
            self.classes = f.read().splitlines()
        self.classes = {i: c for i, c in enumerate(self.classes)}

        self.set_mode(YoloVisualizer.MODE_TRAIN)

    def set_mode(self, mode=MODE_TRAIN):
        if mode == self.MODE_TRAIN:
            self.images_folder = os.path.join(self.dataset_folder, "data", "train", "images")
            self.labels_folder = os.path.join(self.dataset_folder, "data", "train", "labels")
        else:
            self.images_folder = os.path.join(self.dataset_folder, "data", "val", "images")
            self.labels_folder = os.path.join(self.dataset_folder, "data", "val", "labels")

        image_files = [
            f for f in os.listdir(self.images_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png')) and os.path.getsize(os.path.join(self.images_folder, f)) > 0
        ]
        label_files = [
            f for f in os.listdir(self.labels_folder)
            if f.lower().endswith('.txt')
        ]

        image_stems = {os.path.splitext(f)[0] for f in image_files}
        label_stems = {os.path.splitext(f)[0] for f in label_files}
        common_stems = sorted(image_stems & label_stems)

        self.image_names = []
        self.label_names = []

        for stem in common_stems:
            for ext in ['.jpg', '.jpeg', '.png']:
                image_path = os.path.join(self.images_folder, stem + ext)
                if os.path.exists(image_path):
                    self.image_names.append(stem + ext)
                    self.label_names.append(stem + '.txt')
                    break  # Stop after first valid match

        self.num_images = len(self.image_names)
        assert self.num_images > 0, "No valid image-label pairs found."
        self.frame_index = 0
        print(f"Loaded {self.num_images} valid samples.")

    def next_frame(self):
        self.frame_index = (self.frame_index + 1) % self.num_images

    def previous_frame(self):
        self.frame_index = (self.frame_index - 1) % self.num_images

    def seek_frame(self, idx):
        attempts = 0
        while attempts < self.num_images:
            image_file = os.path.join(self.images_folder, self.image_names[idx])
            label_file = os.path.join(self.labels_folder, self.label_names[idx])

            print(f"Trying to read: {image_file}")
            if not os.path.exists(image_file) or os.path.getsize(image_file) == 0:
                print(f"Warning: Invalid or missing image: {image_file}")
                idx = (idx + 1) % self.num_images
                attempts += 1
                continue

            image = cv2.imread(image_file)
            if image is None:
                print(f"Warning: Couldn't read image: {image_file}. Skipping.")
                idx = (idx + 1) % self.num_images
                attempts += 1
                continue

            if not os.path.exists(label_file):
                print(f"Warning: Label file not found for: {image_file}")
                idx = (idx + 1) % self.num_images
                attempts += 1
                continue

            with open(label_file, "r") as f:
                lines = f.read().splitlines()

            for line in lines:
                try:
                    class_index, x, y, w, h = map(float, line.split())
                    cx = int(x * image.shape[1])
                    cy = int(y * image.shape[0])
                    w = int(w * image.shape[1])
                    h = int(h * image.shape[0])
                    x1 = cx - w // 2
                    y1 = cy - h // 2
                    cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                    cv2.putText(image, self.classes[int(class_index)], (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error parsing line in {label_file}: '{line}' — {e}")
                    continue
            self.frame_index = idx
            return image

        print("No valid image-label pairs found after filtering.")
        return None

    def run(self):
        while True:
            frame = self.seek_frame(self.frame_index)
            if frame is None:
                print("No frame to show. Exiting.")
                break

            frame = cv2.resize(frame, (640, 480))
            cv2.imshow(f"YOLO Visualizer — {'Train' if 'train' in self.images_folder else 'Val'}", frame)

            key = cv2.waitKey(0)
            if key in [ord('q'), 27, -1]:  
                break
            elif key == ord('d'):  
                self.next_frame()
            elif key == ord('a'):  
                self.previous_frame()
            elif key == ord('t'):  
                self.set_mode(YoloVisualizer.MODE_TRAIN)
            elif key == ord('v'):  
                self.set_mode(YoloVisualizer.MODE_VAL)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    vis = YoloVisualizer(os.path.dirname(__file__))
    vis.run()

