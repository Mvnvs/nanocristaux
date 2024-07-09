"""import cv2
import json
import os

# Variables globales
annotations = []
current_annotation = None
image = None
image_copy = None
annotation_done = False
classes = ['cubic', 'pseudo-cubic', 'ball']  # Ajoutez toutes les classes ici

def draw_rectangle(event, x, y, flags, param):
    global current_annotation, image, image_copy
    
    if event == cv2.EVENT_LBUTTONDOWN:
        current_annotation = {'start': (x, y), 'end': (x, y)}
    
    elif event == cv2.EVENT_MOUSEMOVE and current_annotation is not None:
        current_annotation['end'] = (x, y)
        image_copy = image.copy()
        cv2.rectangle(image_copy, current_annotation['start'], current_annotation['end'], (0, 255, 0), 2)
        cv2.imshow('image', image_copy)
    
    elif event == cv2.EVENT_LBUTTONUP:
        current_annotation['end'] = (x, y)
        cv2.rectangle(image, current_annotation['start'], current_annotation['end'], (0, 255, 0), 2)
        cv2.imshow('image', image)
        label = input(f"Enter label {classes}: ")
        if label in classes:
            current_annotation['label'] = label
            annotations.append(current_annotation)
        else:
            print("Invalid label. Annotation discarded.")
        current_annotation = None

def annotate_image(image_path):
    global image, image_copy, annotations, annotation_done
    annotations = []
    annotation_done = False
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    image_copy = image.copy()
    
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_rectangle)
    
    while not annotation_done:
        cv2.imshow('image', image)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):  # Quit annotation
            break
        elif key == ord('n'):  # Next image
            annotation_done = True
    
    cv2.destroyAllWindows()
    
    # Save annotations to a JSON file
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    annotation_path = base_filename + '_annotations.json'
    with open(annotation_path, 'w') as f:
        json.dump(annotations, f, indent=4)
    
    print(f"Annotations saved to {annotation_path}")

if __name__ == "__main__":
    image_folder = 'C:/Users/emman/Downloads/cubesnano'
    if not os.path.exists(image_folder):
        print(f"Folder does not exist: {image_folder}")
    else:
        print(f"Annotating images in folder: {image_folder}")
        files = os.listdir(image_folder)
        print(f"Files in folder: {files}")
        for image_file in files:
            print(f"Processing file: {image_file}")
            if image_file.lower().endswith(('jpg', 'png', 'jpeg', 'bmp')):  # Vérification insensible à la casse
                print(f"Annotating {image_file}")
                annotate_image(os.path.join(image_folder, image_file))
            else:
                print(f"Skipping file (not an image): {image_file}")
import cv2
import json
import os

def draw_annotations(image_path, json_path, output_path):
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Load the annotations from the JSON file
    with open(json_path) as f:
        annotations = json.load(f)
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Draw the annotations
    for annotation in annotations:
        start_point = tuple(annotation['start'])
        end_point = tuple(annotation['end'])
        label = annotation['label']
        color = (0, 255, 0)  # Green color for rectangle
        thickness = 2  # Thickness of the rectangle
        
        # Draw the rectangle
        cv2.rectangle(image, start_point, end_point, color, thickness)
        
        # Put the label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        text_x = start_point[0]
        text_y = start_point[1] - 10 if start_point[1] - 10 > 10 else start_point[1] + 10
        cv2.putText(image, label, (text_x, text_y), font, font_scale, color, font_thickness, cv2.LINE_AA)
    
    # Save the annotated image
    cv2.imwrite(output_path, image)
    print(f"Annotated image saved to {output_path}")

# Process all images and their corresponding JSON annotations
image_folder = 'C:/Users/emman/Downloads/cubesnano'
json_folder = 'C:/Users/emman/Downloads/New_annoted_2'
output_folder = 'C:/Users/emman/Downloads/annotated_images'

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

for json_file in os.listdir(json_folder):
    if json_file.endswith('_annotations.json'):
        base_filename = json_file.replace('_annotations.json', '')
        image_filename = base_filename + '.BMP'
        image_path = os.path.join(image_folder, image_filename)
        json_path = os.path.join(json_folder, json_file)
        output_path = os.path.join(output_folder, base_filename + '_annotated.BMP')
        
        draw_annotations(image_path, json_path, output_path)


print("fdp")

import json
import os
import cv2

def convert_to_yolo(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box['start'][0] + box['end'][0]) / 2.0
    y = (box['start'][1] + box['end'][1]) / 2.0
    w = abs(box['end'][0] - box['start'][0])
    h = abs(box['end'][1] - box['start'][1])
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def json_to_yolo(json_folder, image_folder, yolo_folder, classes):
    if not os.path.exists(yolo_folder):
        os.makedirs(yolo_folder)
    
    for json_file in os.listdir(json_folder):
        if json_file.endswith('.json'):
            with open(os.path.join(json_folder, json_file)) as f:
                annotations = json.load(f)
            
            image_filename = json_file.replace('_annotations.json', '.bmp')
            image_path = os.path.join(image_folder, image_filename)
            if not os.path.exists(image_path):
                print(f"Image not found for {json_file}")
                continue
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue
            
            height, width, _ = image.shape
            
            yolo_annotations = []
            for annotation in annotations:
                label = annotation['label']
                if label in classes:
                    class_id = classes.index(label)
                    bbox = convert_to_yolo((width, height), annotation)
                    yolo_annotations.append(f"{class_id} {' '.join(map(str, bbox))}")
                else:
                    print(f"Unknown label {label} in {json_file}")
            
            yolo_annotation_path = os.path.join(yolo_folder, image_filename.replace('.bmp', '.txt'))
            with open(yolo_annotation_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            
            print(f"Converted {json_file} to YOLO format")

if __name__ == "__main__":
    json_folder = 'C:/Users/emman/Downloads/cubesnano_annotations'  # Remplacez par le chemin de votre dossier JSON
    image_folder = 'C:/Users/emman/Downloads/cubesnano'  # Remplacez par le chemin de votre dossier BMP
    yolo_folder = 'C:/Users/emman/Downloads/cubesnano_yolo_annotations'  # Chemin de sortie pour les fichiers YOLO
    classes = ['cubic', 'pseudo-cubic', 'ball']  # Ajoutez toutes vos classes ici
    
    json_to_yolo(json_folder, image_folder, yolo_folder, classes)"""

import cv2
import os

def draw_annotations(image_path, txt_path, output_path):
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    height, width, _ = image.shape
    
    # Load the annotations from the TXT file
    annotations = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            class_id = parts[0]
            x_center = float(parts[1]) * width
            y_center = float(parts[2]) * height
            w = float(parts[3]) * width
            h = float(parts[4]) * height
            start_point = (int(x_center - w / 2), int(y_center - h / 2))
            end_point = (int(x_center + w / 2), int(y_center + h / 2))
            label = f'class {class_id}'
            annotations.append({'start': start_point, 'end': end_point, 'label': label})
    
    # Draw the annotations
    for annotation in annotations:
        start_point = annotation['start']
        end_point = annotation['end']
        label = annotation['label']
        color = (0, 255, 0)  # Green color for rectangle
        thickness = 2  # Thickness of the rectangle
        
        # Draw the rectangle
        cv2.rectangle(image, start_point, end_point, color, thickness)
        
        # Put the label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        text_x = start_point[0]
        text_y = start_point[1] - 10 if start_point[1] - 10 > 10 else start_point[1] + 10
        cv2.putText(image, label, (text_x, text_y), font, font_scale, color, font_thickness, cv2.LINE_AA)
    
    # Save the annotated image with the same name as the input image
    output_filename = os.path.basename(image_path).split('.')[0] + '_annotated.jpg'
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, image)
    print(f"Annotated image saved to {output_path}")

# Process all images and their corresponding TXT annotations in the image_folder and txt_folder
image_folder = 'C:/Users/emman/Downloads/cubesnano_jpg'
txt_folder = 'C:/Users/emman/Downloads/cubesnano_yolo_annotations'
output_folder = 'C:/Users/emman/Downloads/annotated_images'

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Process each image in the image_folder
for image_file in os.listdir(image_folder):
    if image_file.lower().endswith(('.jpg', '.png')):
        base_filename, extension = os.path.splitext(image_file)
        txt_file = base_filename + '.txt'
        txt_path = os.path.join(txt_folder, txt_file)  # Use txt_folder for annotations
        image_path = os.path.join(image_folder, image_file)
        output_path = os.path.join(output_folder, base_filename + '_annotated.jpg')
        
        if os.path.exists(txt_path):
            draw_annotations(image_path, txt_path, output_path)
        else:
            print(f"Annotations not found for {image_file}")

