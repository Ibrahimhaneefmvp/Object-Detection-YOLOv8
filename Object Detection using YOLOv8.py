#I have added step-wise comments in order to build an easy walkthrough.


#Please mount your Google Drive before running the code.

# Step 1: Install the Ultralytics YOLOv8 package
!pip install ultralytics

# Step 2: Import necessary modules
import torch
from ultralytics import YOLO
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Step 3: Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # 'yolov8n' is the nano version, you can use 'yolov8s' for better accuracy

# Step 4: Define input and output folders
input_folder = '/content/drive/My Drive/ImageSamples'
output_folder = '/content/drive/My Drive/ImageSamples2'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load all images from the folder
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Function to run YOLOv8 inference and save images with bounding boxes and captions
def run_yolov8_and_save(image_path, output_path):
    results = model(image_path)  # Perform inference on the image

    # Load the image
    img = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # Generate caption and draw bounding boxes
    caption = "Objects detected: "
    for result in results:
        boxes = result.boxes  # Get bounding boxes
        for box in boxes:
            # Extract box information
            xyxy = box.xyxy[0]  # Box coordinates
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class ID
            label = model.names[cls]  # Class name from YOLOv8

            # Draw bounding box
            rect = patches.Rectangle((xyxy[0], xyxy[1]), xyxy[2] - xyxy[0], xyxy[3] - xyxy[1], linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            # Add label and confidence score to the box
            ax.text(xyxy[0], xyxy[1] - 10, f'{label}: {conf:.2f}', color='red', fontsize=12, weight='bold')

            # Update caption with class name and confidence
            caption += f'{label} ({conf:.2f}), '

    # Remove trailing comma in the caption
    if caption.endswith(', '):
        caption = caption[:-2]

    # Display the caption at the top of the image
    ax.text(0.5, 1.05, caption, fontsize=14, color='blue', weight='bold', ha='center', va='top', transform=ax.transAxes)

    # Save the annotated image with the caption
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Step 5: Run inference and save images with captions
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    output_path = os.path.join(output_folder, image_file)
    run_yolov8_and_save(image_path, output_path)

print("Processing complete. Annotated images are saved in the ImageSamples2 folder.")
