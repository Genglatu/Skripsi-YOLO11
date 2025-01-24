!pip install ultralytics
from IPython import display
display.clear_output()
import ultralytics
ultralytics.checks()

# Disable Weights & Biases completely
os.environ['WANDB_MODE'] = 'disabled'

# Define the dataset_path
dataset_path811 = ' /kaggle/input/skin-cancer-data-new'

# Set the path to the YAML file
yaml_file_path811 = os.path.join(dataset_path811, 'data.yaml')

# Load and print the contents of the YAML file
with open(yaml_file_path811, 'r') as file:
    yaml_content = yaml.load(file, Loader=yaml.FullLoader)
    print(yaml.dump(yaml_content, default_flow_style=False))

import yaml
from pathlib import Path
from collections import defaultdict
import os

# Membaca file YAML
with open(yaml_file_path811, 'r') as f:
    data_config = yaml.safe_load(f)

# Ambil kelas dari data.yaml
class_names = data_config['names']  # Daftar nama kelas

# Menetapkan path folder train, valid, dan test secara manual
data_paths = {
    "train": os.path.join(dataset_path811, "train", "labels"),
    "val": os.path.join(dataset_path811, "valid", "labels"),
    "test": os.path.join(dataset_path811, "test", "labels")
}

# Membuat dictionary untuk menyimpan jumlah gambar dan bounding box per kelas dan split
class_counts = defaultdict(lambda: {"train": {"images": set(), "bboxes": 0}, "val": {"images": set(), "bboxes": 0}, "test": {"images": set(), "bboxes": 0}})
split_totals = {"train": {"images": set(), "bboxes": 0}, "val": {"images": set(), "bboxes": 0}, "test": {"images": set(), "bboxes": 0}}

# Looping untuk membaca setiap file label dalam setiap folder (train, val, test)
for split, label_dir in data_paths.items():

    # Looping untuk semua file label .txt
    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            label_path = os.path.join(label_dir, label_file)

            with open(label_path, 'r') as f:
                lines = f.readlines()

                # Jika ada bounding box, tambahkan gambar dan bounding boxnya
                if lines:
                    image_id = os.path.splitext(label_file)[0]  # ID gambar
                    for line in lines:
                        class_id = int(line.split()[0])  # ID kelas dari file label
                        class_name = class_names[class_id]

                        # Tambahkan image ke set unik per kelas dan split, tambahkan jumlah bounding box
                        class_counts[class_name][split]["images"].add(image_id)
                        class_counts[class_name][split]["bboxes"] += 1

                    # Tambahkan ke total split
                    split_totals[split]["images"].add(image_id)
                    split_totals[split]["bboxes"] += len(lines)

# Print jumlah gambar dan bounding box per kelas untuk setiap split
print("Jumlah per kelas untuk setiap split:")
for class_name, splits in class_counts.items():
    print(f"Kelas: {class_name}")
    for split, counts in splits.items():
        num_images = len(counts["images"])
        num_bboxes = counts["bboxes"]
        print(f"  Split: {split}, Jumlah Gambar: {num_images}, Bounding Box: {num_bboxes}")

# Menghitung total gambar dan bounding box secara keseluruhan dan per split
total_images = sum(len(splits[split]["images"]) for splits in class_counts.values() for split in splits)
total_bboxes = sum(splits[split]["bboxes"] for splits in class_counts.values() for split in splits)

print("\nTotal keseluruhan:")
print(f"Total Jumlah Gambar: {total_images}")
print(f"Total Jumlah Bounding Box: {total_bboxes}")

print("\nTotal per split:")
for split, counts in split_totals.items():
    num_images = len(counts["images"])
    num_bboxes = counts["bboxes"]
    print(f"Split: {split}, Jumlah Gambar: {num_images}, Bounding Box: {num_bboxes}")

%%time
#Load Model
model = YOLO('yolo11n.pt')

#Train Model
results = model.train(data=yaml_file_path811, epochs=500, patience=50, batch=16, dropout=0.3, imgsz=640, plots=True, name="Skripsi Yolo11") 

#Validasi dan Uji
results_val = model.val(conf=0.25, name="Validasi")
results_test = model.val(conf=0.25, split='test',name="Uji")
