import os
import torch
import pickle
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from lib.dataset.joint_patches import FootPatchesDataset
from lib.utils.collate_fn import TokenizedReportCollateFn
import models

def load_model(cfg, model_path):
    """
    Load the trained feature extractor model.

    Args:
        cfg: Configuration object.
        model_path: Path to the saved model checkpoint.

    Returns:
        Loaded model.
    """
    model = eval('models.' + cfg.MODEL.NAME + '.get_feature_extractor')(cfg, is_train=False)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def generate_embeddings(cfg, model, dataloader):
    """
    Generate embeddings using the trained model.

    Args:
        cfg: Configuration object.
        model: Trained feature extractor model.
        dataloader: DataLoader for the dataset.

    Returns:
        List of embeddings.
    """
    embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch['images']
            images = images.to(cfg.DEVICE)

            # Extract embeddings
            embedding = model(images)
            embeddings.append(embedding.cpu())
    return embeddings

def main():
    # Load configuration
    from config import cfg, update_config
    cfg_path = 'config/test.yaml'  # Path to your config file
    update_config(cfg, cfg_path)

    # Device setup
    device = torch.device('cuda' if cfg.DEVICE == 'GPU' and torch.cuda.is_available() else 'cpu')
    cfg.DEVICE = device

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

    # Load dataset
    with open(cfg.DATASET.PKL, 'rb') as f:
        pkl_data = pickle.load(f)
    
    dataset = FootPatchesDataset(cfg, pkl_data)
    collate_fn = TokenizedReportCollateFn(tokenizer, max_length=cfg.MODEL.MAX_SEQ_LENGTH)
    dataloader = DataLoader(dataset, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU, shuffle=False, collate_fn=collate_fn)

    # Load trained model
    model_path = 'path/to/saved_model.pth'  # Replace with the actual path to your saved model
    model = load_model(cfg, model_path)
    model.to(device)

    # Generate embeddings
    embeddings = generate_embeddings(cfg, model, dataloader)

    # Save embeddings
    output_path = 'output/embeddings.pkl'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)

    print(f"Embeddings saved to {output_path}")

if __name__ == '__main__':
    main()