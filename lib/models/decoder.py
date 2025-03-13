from transformers import AutoModelForCausalLM
from torch import nn
import torch
from peft import PeftModel  # Import LoRA wrapper

class DecoderWithImageEmbedding(AutoModelForCausalLM):
    def __init__(self, config, image_embedding_dim=2816, text_embedding_dim=768):
        super().__init__(config)
        # Add a projection layer to map image embeddings to text embedding space
        self.image_projection = nn.Linear(image_embedding_dim, text_embedding_dim)

    def forward(self, input_ids, attention_mask=None, labels=None, image_embeddings=None, **kwargs):
        if image_embeddings is not None:
            # Project image embeddings to match text embedding dimension
            projected_image_embeddings = self.image_projection(image_embeddings)

            # Get text embeddings
            input_embeddings = self.get_input_embeddings()(input_ids)

            # Concatenate along the sequence dimension
            combined_embeddings = torch.cat([projected_image_embeddings.unsqueeze(1), input_embeddings], dim=1)

            # Replace the input embeddings with the combined embeddings
            self.set_input_embeddings(torch.nn.Embedding.from_pretrained(combined_embeddings))

        # Pass remaining arguments to the parent class's forward method
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

    def to(self, *args, **kwargs):
        # Ensure the image projection layer is moved to the correct device
        self.image_projection.to(*args, **kwargs)
        return super().to(*args, **kwargs)


class DecoderWithLoRA(nn.Module):
    """
    A custom wrapper that combines LoRA with the DecoderWithImageEmbedding class.
    This implementation processes image embeddings before passing to the LoRA model.
    """
    def __init__(self, lora_model, peft_config):
        super().__init__()
        self.lora_model = lora_model
        self.peft_config = peft_config

        # Extract the image projection layer from the original model
        if hasattr(lora_model.base_model.model, 'image_projection'):
            self.image_projection = lora_model.base_model.model.image_projection
        else:
            # Default image projection if not found
            self.image_projection = nn.Linear(2816, 768)

    def forward(self, input_ids, attention_mask=None, labels=None, image_embeddings=None, **kwargs):
        if image_embeddings is not None:
            # Project image embeddings to match text embedding dimension
            projected_image_embeddings = self.image_projection(image_embeddings)

            # Get text embeddings from the base model
            input_embeddings = self.lora_model.get_input_embeddings()(input_ids)

            # Concatenate along the sequence dimension
            combined_embeddings = torch.cat([projected_image_embeddings.unsqueeze(1), input_embeddings], dim=1)

            # Create new input IDs that account for the additional token
            new_input_ids = torch.cat([
                torch.zeros((input_ids.size(0), 1), dtype=input_ids.dtype, device=input_ids.device),  # Placeholder for image token
                input_ids
            ], dim=1)

            # Adjust attention mask if provided
            if attention_mask is not None:
                new_attention_mask = torch.cat([
                    torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device),  # Enable attention for image token
                    attention_mask
                ], dim=1)
            else:
                new_attention_mask = None

            # Adjust labels if provided
            if labels is not None:
                # Ensure labels is 2D for concatenation
                if labels.dim() == 1:
                    labels = labels.unsqueeze(1)  # Make it [batch_size, seq_len]

                # Create new_labels by inserting -100 at the position of the image embedding
                # Shape: [batch_size, seq_len + 1]
                new_labels = torch.cat([
                    torch.full((labels.size(0), 1), -100, dtype=labels.dtype, device=labels.device),  # -100 for image token
                    labels
                ], dim=1)

                # Ensure new_labels matches new_input_ids' seq_len
                if new_labels.size(1) > new_input_ids.size(1):
                    # Trim new_labels to match new_input_ids' seq_len
                    new_labels = new_labels[:, :new_input_ids.size(1)]
                elif new_labels.size(1) < new_input_ids.size(1):
                    # Pad new_labels to match new_input_ids' seq_len
                    padding = torch.full(
                        (new_labels.size(0), new_input_ids.size(1) - new_labels.size(1)),
                        -100,  # Ignore index
                        dtype=new_labels.dtype,
                        device=new_labels.device
                    )
                    new_labels = torch.cat([new_labels, padding], dim=1)
            else:
                new_labels = None

            # Forward pass with the modified inputs
            outputs = self.lora_model(
                input_ids=new_input_ids,
                attention_mask=new_attention_mask,
                labels=new_labels,
                **kwargs
            )

            return outputs
        else:
            # Standard forward pass without image embeddings
            return self.lora_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )

    def to(self, *args, **kwargs):
        # Ensure all components are moved to the correct device
        self.image_projection.to(*args, **kwargs)
        self.lora_model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def generate(self, input_ids, attention_mask=None, image_embeddings=None, **kwargs):
        """Support for text generation with optional image conditioning."""
        if image_embeddings is not None:
            # Project image embeddings to match text embedding dimension
            projected_image_embeddings = self.image_projection(image_embeddings)

            # Create new input IDs with a placeholder for the image token
            new_input_ids = torch.cat([
                torch.zeros((input_ids.size(0), 1), dtype=input_ids.dtype, device=input_ids.device),  # Placeholder token for image
                input_ids
            ], dim=1)

            # Adjust attention mask if provided
            if attention_mask is not None:
                new_attention_mask = torch.cat([
                    torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device),  # Enable attention for image token
                    attention_mask
                ], dim=1)
            else:
                # If no attention mask is provided, create one that enables attention for all tokens
                new_attention_mask = torch.ones((new_input_ids.size(0), new_input_ids.size(1)), dtype=torch.long, device=new_input_ids.device)

            # Generate text with the modified inputs
            return self.lora_model.generate(
                input_ids=new_input_ids,
                attention_mask=new_attention_mask,
                **kwargs
            )
        else:
            # Standard generation without image embeddings
            return self.lora_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )

def get_decoder_with_image_embedding(model_name_or_path):
    # Load the decoder model with the custom class
    decoder = DecoderWithImageEmbedding.from_pretrained(model_name_or_path)
    return decoder