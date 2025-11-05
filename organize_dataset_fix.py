import json
import os
from pathlib import Path

def convert_bbox_to_yolo(bbox, img_width=1280, img_height=960):
    """Convert [x, y, width, height] to YOLO format [x_center/width, y_center/height, width/width, height/height]"""
    x, y, w, h = bbox
    
    # Normalize coordinates
    x_center = (x + w/2) / img_width
    y_center = (y + h/2) / img_height
    w = w / img_width
    h = h / img_height
    
    return [x_center, y_center, w, h]

def process_json_labels(json_path, output_dir):
    """Process JSON labels file and create YOLO format label files"""
    # Try different encodings
    encodings = ['euc-kr', 'cp949', 'utf-8']
    data = None
    
    for encoding in encodings:
        try:
            with open(json_path, 'r', encoding=encoding) as f:
                data = json.load(f)
            print(f"Successfully loaded JSON with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
        except json.JSONDecodeError:
            continue
    
    if data is None:
        raise ValueError(f"Could not decode JSON file with any of the tried encodings: {encodings}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for item in data:
        image_id = item['image_id']
        
        # Each image might have multiple individuals
        for individual in item['individual_factors']:
            bbox = individual['bbox']
            
            # Convert bbox to YOLO format
            yolo_bbox = convert_bbox_to_yolo(bbox)
            
            # Create label file (0 is the class index for person)
            label_path = os.path.join(output_dir, f"{image_id}.txt")
            with open(label_path, 'a') as f:  # 'a' mode to handle multiple people in same image
                f.write(f"0 {' '.join([str(x) for x in yolo_bbox])}\n")

def split_dataset(dataset_dir, split_ratio=0.2):
    """Split dataset into train and validation sets"""
    images_dir = dataset_dir / "train" / "images"
    labels_dir = dataset_dir / "train" / "labels"
    
    # Create validation directories
    val_images_dir = dataset_dir / "val" / "images"
    val_labels_dir = dataset_dir / "val" / "labels"
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    
    # Get list of all images
    image_files = list(images_dir.glob("*.jpg"))
    
    # Calculate number of validation images
    num_val = int(len(image_files) * split_ratio)
    
    # Randomly select validation images
    import random
    random.seed(42)  # for reproducibility
    val_images = random.sample(image_files, num_val)
    
    # Move validation images and their corresponding labels
    for img_path in val_images:
        img_name = img_path.name
        label_name = img_path.stem + ".txt"
        
        # Move image
        os.rename(img_path, val_images_dir / img_name)
        
        # Move label if it exists
        label_path = labels_dir / label_name
        if label_path.exists():
            os.rename(label_path, val_labels_dir / label_name)

def main():
    dataset_dir = Path("dataset")
    
    # Process training set
    train_json = dataset_dir / "train" / "고성능.json"
    train_labels_dir = dataset_dir / "train" / "labels"
    process_json_labels(train_json, train_labels_dir)
    
    # Split dataset into train and validation sets
    split_dataset(dataset_dir)
    
    # Create dataset.yaml file
    yaml_content = f"""path: {dataset_dir.absolute()}  # dataset root dir
train: train/images  # train images
val: val/images  # val images

# Classes
nc: 1  # number of classes
names: ['person']  # class names
"""
    
    with open(dataset_dir / "thermal_dataset.yaml", 'w') as f:
        f.write(yaml_content)
        
    print("Dataset organization completed!")

if __name__ == "__main__":
    main()