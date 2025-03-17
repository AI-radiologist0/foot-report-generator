import os
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from models.decoder import get_decoder_with_image_embedding
from utils.collate_fn import TokenizedReportCollateFn
from dataset.joint_patches import FootPatchesDataset
from config import cfg, update_config
import pickle

def load_model_and_tokenizer(cfg, device):
    """
    Load the trained model and tokenizer.
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.DECODER.NAME)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if tokenizer.eos_token != "[EOS]":
        tokenizer.add_special_tokens({'eos_token': "[EOS]"})
    if tokenizer.bos_token != "[BOS]":
        tokenizer.add_special_tokens({'bos_token': "[BOS]"})
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["[FINDING]", "[CONCLUSION]", "[RECOMMEND]"]
    })
    tokenizer.padding_side = "left"

    # Load model
    model = get_decoder_with_image_embedding(cfg.DECODER.NAME)
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(cfg.DECODER.EXTRA.PT, map_location=device))
    model.to(device)
    model.eval()

    return model, tokenizer


def generate_texts(model, tokenizer, feature_extractor, dataloader, device):
    """
    Generate texts for images using the trained model.
    """
    generated_texts = []
    reference_texts = []
    images = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            input_images = batch["images"].to(device)
            patches = batch["patch_tensors"].to(device)
            class_labels = batch["class_labels"]

            # Generate image embeddings
            image_embeddings = feature_extractor(input_images, patches)

            # Generate text
            outputs = model.generate(
                input_ids=None,
                image_embeddings=image_embeddings,
                max_length=150,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                early_stopping=True
            )

            ## Decode generated texts
            batch_generated_texts = [
                tokenizer.decode(output, skip_special_tokens=False).replace("[PAD]", "").split("[EOS]")[0].strip() for output in outputs
            ]
            generated_texts.extend(batch_generated_texts)

            # Decode reference texts
            batch_reference_texts = [
                tokenizer.decode(label, skip_special_tokens=False).replace("[PAD]", "").split("[EOS]")[0].strip() for label in batch["labels"]
            ]
            reference_texts.extend(batch_reference_texts)

            # Collect images and labels
            images.extend(input_images.cpu())
            labels.extend(class_labels.cpu())

    return images, labels, generated_texts, reference_texts


def save_results_as_images(images, labels, generated_texts, reference_texts, output_dir):
    """
    Save inference results as images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Adjust font path for your system
    font = ImageFont.truetype(font_path, size=20)

    for i, (image, label, generated_text, reference_text) in enumerate(zip(images, labels, generated_texts, reference_texts)):
        # Convert tensor to PIL image
        image = image.permute(1, 2, 0).numpy()  # Convert to HWC format
        image = (image * 255).astype("uint8")  # Scale to 0-255
        pil_image = Image.fromarray(image)

        # Create a blank canvas
        canvas_width = pil_image.width + 600
        canvas_height = pil_image.height
        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
        canvas.paste(pil_image, (0, 0))

        # Draw text
        draw = ImageDraw.Draw(canvas)
        text_x = pil_image.width + 10
        draw.text((text_x, 10), f"Disease Class: {label}", fill="black", font=font)
        draw.text((text_x, 50), "Reference Report:", fill="black", font=font)
        draw.text((text_x, 90), reference_text, fill="black", font=font)
        draw.text((text_x, 250), "Generated Report:", fill="black", font=font)
        draw.text((text_x, 290), generated_text, fill="black", font=font)

        # Save the result
        output_path = os.path.join(output_dir, f"result_{i + 1}.png")
        canvas.save(output_path)


def main():
    # Load configuration
    cfg_path = "path/to/config.yaml"  # Update with the actual path to your config file
    update_config(cfg, cfg_path)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer, model, and feature extractor
    model, tokenizer = load_model_and_tokenizer(cfg, device)

        # Load feature extractor model
    feature_extractor_name = cfg.MODEL.NAME
    feature_extractor = eval('models.' + feature_extractor_name + '.get_feature_extractor')(
        cfg, is_train=False, remove_classifier=True
    )
    feature_extractor = feature_extractor.eval()
    feature_extractor = feature_extractor.to(device)
    print("Feature extractor model loaded.")


    # Load dataset and dataloader
    with open(cfg.DATASET.PKL, "rb") as f:
        pkl_data = pickle.load(f)
    dataset = FootPatchesDataset(cfg, pkl_data)
    collate_fn = TokenizedReportCollateFn(tokenizer, max_length=cfg.DECODER.EXTRA.MAX_SEQ_LENGTH)
    dataloader = DataLoader(dataset, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU, shuffle=False, collate_fn=collate_fn)

    # Generate texts
    images, labels, generated_texts, reference_texts = generate_texts(model, tokenizer, feature_extractor, dataloader, device)

    # Save results as images
    output_dir = "inference_results"
    save_results_as_images(images, labels, generated_texts, reference_texts, output_dir)
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()