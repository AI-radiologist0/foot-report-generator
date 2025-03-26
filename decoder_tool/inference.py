import os
import pickle
import argparse
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from tqdm import tqdm  # Add tqdm for progress bar

import _init_path
import models
from models.decoder import get_decoder_with_image_embedding, DecoderWithLoRA
from utils.collate_fn import TokenizedReportCollateFn
from dataset.joint_patches import FootPatchesDataset
from config import cfg, update_config

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
        "additional_special_tokens": ['<img>', "[FINDING]", "[CONCLUSION]", "[RECOMMEND]"]
    })
    tokenizer.padding_side = "left"
    
    if cfg.DECODER.NAME == 'GPT2':
        community = "openai-community"
    
    decoder_huggingface_name = f"{community}/{cfg.DECODER.NAME.lower()}"

    lora_config = LoraConfig(
        r=8,  # LoRA rank
        lora_alpha=32,  # Scaling factor
        target_modules=["c_attn", "image_projection"],  # LoRA applied modules
        lora_dropout=0.1,  # Dropout
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Load model
    model = get_decoder_with_image_embedding(decoder_huggingface_name)
    model.resize_token_embeddings(len(tokenizer))
    # Apply LoRA to the decoder model
    lora_wrapped_model = get_peft_model(model, lora_config)

    # Wrap the LoRA model with the custom wrapper to support image_embeddings
    image_token_id = tokenizer.convert_tokens_to_ids('<img>')
    model = DecoderWithLoRA(lora_wrapped_model, lora_config, image_token_id)

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
        for batch in tqdm(dataloader, desc="Generating texts"):
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
                tokenizer.decode([token for token in label.tolist() if token >= 0], skip_special_tokens=False).replace("[PAD]", "").split("[EOS]")[0].strip()
                for label in batch["labels"]
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

    for i, (image, label, generated_text, reference_text) in enumerate(tqdm(zip(images, labels, generated_texts, reference_texts), desc="Saving results", total=len(images))):
        # Convert tensor to PIL image
        image = image.permute(1, 2, 0).numpy()  # Convert to HWC format
        image = (image * 255).astype("uint8")  # Scale to 0-255
        pil_image = Image.fromarray(image)

        # Create a blank canvas
        canvas_width = max(pil_image.width + 400, 800)  # Ensure a minimum width for text
        canvas_height = pil_image.height + 300  # Adjust height for additional text
        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

        # Center the image on the canvas
        image_x = (canvas_width - pil_image.width) // 2
        image_y = 10  # Top margin
        canvas.paste(pil_image, (image_x, image_y))

        # Draw text below the image
        draw = ImageDraw.Draw(canvas)
        text_x = 10  # Left margin for text
        text_y = pil_image.height + 20  # Start below the image

        draw.text((text_x, text_y), f"Disease Class: {label}", fill="black", font=font)
        text_y += 40  # Move down for the next line

        draw.text((text_x, text_y), "Reference Report:", fill="black", font=font)
        text_y += 30  # Move down for the reference text
        draw.text((text_x, text_y), reference_text, fill="black", font=font)
        text_y += 60  # Move down for the next section

        draw.text((text_x, text_y), "Generated Report:", fill="black", font=font)
        text_y += 30  # Move down for the generated text
        draw.text((text_x, text_y), generated_text, fill="black", font=font)

        print(f"{i} 번째 환자\n")
        print(f"report(Reference): {reference_text}")
        print(f"report(Generated): {generated_text}")

        # Save the result
        output_path = os.path.join(output_dir, f"result_{i + 1}.png")
        canvas.save(output_path)

        if i == 10:  # Save results for the first 10 patients only
            break

# def save_results_as_images(images, labels, generated_texts, reference_texts, output_dir):
#     """
#     Save inference results as images.
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Adjust font path for your system
#     font = ImageFont.truetype(font_path, size=20)

#     for i, (image, label, generated_text, reference_text) in enumerate(tqdm(zip(images, labels, generated_texts, reference_texts), desc="Saving results", total=len(images))):
#         # Convert tensor to PIL image
#         image = image.permute(1, 2, 0).numpy()  # Convert to HWC format
#         image = (image * 255).astype("uint8")  # Scale to 0-255
#         pil_image = Image.fromarray(image)

#         # Create a blank canvas
#         canvas_width = pil_image.width + 400
#         canvas_height = pil_image.height
#         canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
#         canvas.paste(pil_image, (0, 0))

#         # Draw text
#         draw = ImageDraw.Draw(canvas)
#         text_x = pil_image.width + 10
#         draw.text((text_x, 10), f"Disease Class: {label}", fill="black", font=font)
#         draw.text((text_x, 50), "Reference Report:", fill="black", font=font)
#         draw.text((text_x, 90), reference_text, fill="black", font=font)
#         draw.text((text_x, 200), "Generated Report:", fill="black", font=font)  # 위치 조정
#         draw.text((text_x, 240), generated_text, fill="black", font=font)  # 위치 조정
        
#         print(f"{i} 번째 환자\n")
#         print(f"report(Reference) {reference_text}")
#         print(f"report(Generated) {generated_text}")

#         # Save the result
#         output_path = os.path.join(output_dir, f"result_{i + 1}.png")
#         canvas.save(output_path)
        
#         if i == 10:
#             break


def main():
    parser = argparse.ArgumentParser(description="Foot Report Generator Inference")
    parser.add_argument("--cfg", required=True, help="Path to the configuration YAML file")
    args = parser.parse_args()

    # Load configuration
    update_config(cfg, args)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if cfg.PHASE != "test":
        print(f"PHASE is Not test cfg.PHASE {cfg.PHASE}")
        return
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

    # Split dataset into 85% train and 15% test
    train_size = int(len(dataset) * 0.85)
    test_size = len(dataset) - train_size
    _, test_dataset = random_split(dataset, [train_size, test_size])

    collate_fn = TokenizedReportCollateFn(tokenizer, max_length=cfg.DECODER.EXTRA.MAX_SEQ_LENGTH)
    dataloader = DataLoader(test_dataset, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU, shuffle=False, collate_fn=collate_fn)

    # Generate texts
    images, labels, generated_texts, reference_texts = generate_texts(model, tokenizer, feature_extractor, dataloader, device)

    # Save results as images
    output_dir = "inference_results"
    save_results_as_images(images, labels, generated_texts, reference_texts, output_dir)
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()