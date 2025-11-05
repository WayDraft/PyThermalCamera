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
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
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

def main():
    dataset_dir = Path("dataset")
    
    # Process training set
    train_json = dataset_dir / "train" / "고성능.json"
    train_labels_dir = dataset_dir / "train" / "labels"
    process_json_labels(train_json, train_labels_dir)
    
    # Check if validation set exists and process it
    val_json = dataset_dir / "val" / "고성능.json"
    if val_json.exists():
        val_labels_dir = dataset_dir / "val" / "labels"
        process_json_labels(val_json, val_labels_dir)
        
    # Create dataset.yaml file
    yaml_content = f"""
path: {dataset_dir.absolute()}  # dataset root dir
train: train/images  # train images
val: val/images  # val images

# Classes
names:
  0: person  # class names
"""
    
    with open(dataset_dir / "thermal_dataset.yaml", 'w') as f:
        f.write(yaml_content.strip())
        
    print("Dataset organization completed!")

if __name__ == "__main__":
    main()