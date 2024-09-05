# Object Detection using YOLOv8

This project demonstrates object detection using the pre-trained YOLOv8 model, comparing it against YOLOv5x and Faster R-CNN. The project involves testing object detection accuracy on a custom image dataset, drawing bounding boxes around detected objects, and providing human-readable captions (like "Cat", "Dog", etc.) for each detection.

## Project Overview

- **Models Compared**: YOLOv5x, Faster R-CNN, and YOLOv8.
- **Final Model Used**: YOLOv8, which gave the best results in terms of both speed and accuracy.
- **Custom Image Set**: A custom image set was used to test the accuracy of the models, particularly in detecting closely resembling objects like cats, dogs, and lions.
- **Challenges**: Differentiating between closely resembling animals such as cows, sheep, and goats remains a challenge with pre-trained models.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository.git
   ```

2. Install dependencies:
   ```bash
   pip install ultralytics
   ```

3. Mount Google Drive for accessing the dataset (if using Google Colab):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

## Usage

To run object detection using YOLOv8 on your custom dataset:

1. Load the pre-trained YOLOv8 model:
   ```python
   from ultralytics import YOLO
   model = YOLO('yolov8n.pt')  # Use 'yolov8s.pt' for better accuracy
   ```

2. Run inference on images:
   ```python
   results = model('/path/to/your/image/folder')
   ```

3. The script will output images with bounding boxes and captions, saved in the specified output directory.

## Results

The project tested multiple architectures, and **YOLOv8** provided the best results for object detection. However, closely resembling animals such as **wolves, lions, and dogs** were sometimes misclassified, highlighting the need for further fine-tuning.

## Future Work

- **Model Fine-Tuning**: To improve accuracy, especially in differentiating between similar-looking animals.
- **Custom Dataset Expansion**: Adding more diverse samples to improve model generalization.
