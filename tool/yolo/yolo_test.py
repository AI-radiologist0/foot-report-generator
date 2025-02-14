# -----------------------------------------------------------
# 
# written by Jeongmin Kim (jm.kim@dankook.ac.kr)
#
# -----------------------------------------------------------
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Load a pre-trained YOLOv5 model (small version)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # YOLOv5s: Lightweight model

# Upload your image
image_path = './data/foot/gout/CAUNURI/CAUHGOUT1003_20221201202749_CR/1.2.276.0.7230010.3.1.4.67515890.5552.1670892124.186419.jpg'  # Replace with your image path

# Perform inference
results = model(image_path)

# Visualize the results
results.show()  # Display the detected objects in the image

# Save the results to a file
results.save(save_dir='output/')  # Save the output images to 'output/' directory

# Optional: Display the saved output image
output_image = Image.open('output/image0.jpg')  # Update with the correct output image path
plt.imshow(output_image)
plt.axis('off')
plt.show()