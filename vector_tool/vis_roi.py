import pickle
import matplotlib.pyplot as plt
import os

# Load the Pickle data
output_pkl_path = "data/pkl/output200x300.pkl"
with open(output_pkl_path, 'rb') as pkl_file:
    final_data = pickle.load(pkl_file)

# Directory to save visualization outputs
visualization_dir = "bbx_visualization/"
os.makedirs(visualization_dir, exist_ok=True)

# Function to visualize 34 bounding boxes in a single plot
def visualize_bbx(image_id, bbx_images, save_dir):
    fig, axes = plt.subplots(6, 6, figsize=(12, 12))  # 6x6 grid (max 36 images)
    fig.suptitle(f"Image ID: {image_id}", fontsize=16)

    for ax, img in zip(axes.flat, bbx_images):
        ax.imshow(img)
        ax.axis('off')

    # Hide unused subplots
    for i in range(len(bbx_images), 36):
        axes.flat[i].axis('off')

    # Save the visualization
    save_path = os.path.join(save_dir, f"{image_id}_bbx_visualization.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization: {save_path}")

# Process and visualize each image's bounding boxes
for image_id, data in final_data.items():
    bbx_images = data["bbx"]  # Extract bounding boxes
    visualize_bbx(image_id, bbx_images, visualization_dir)
    break
