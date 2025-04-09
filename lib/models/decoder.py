from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig
from .feature_extractor import get_feature_extractor
from utils.utils import insert_before_bos, prepare_generate_inputs
from torch import nn
import torch
from peft import PeftModel  # Import LoRA wrapper
import logging
logger = logging.getLogger(__name__)  # 기존 logger와 동일한 인스턴스 사용


class DecoderWithFeatureExtractor(nn.Module):
    def __init__(self, cfg, decoder_name, feature_extractor_ckpt, image_embedding_dim=2816, text_embedding_dim=768, freeze_feature_extractor=False):
        """
        Combines a FeatureExtractor and a DecoderWithImageEmbedding.

        Args:
            config: Configuration for the decoder (AutoModelForCausalLM).
            feature_extractor_ckpt (str): Path to the pre-trained FeatureExtractor checkpoint.
            image_embedding_dim (int): Dimension of the image embeddings from the FeatureExtractor.
            text_embedding_dim (int): Dimension of the text embeddings for the decoder.
            freeze_feature_extractor (bool): Whether to freeze the FeatureExtractor during training.
        """
        super().__init__()

        # Initialize the FeatureExtractor
        self.feature_extractor = get_feature_extractor(cfg, is_train=True, remove_classifier=True)
        self.feature_extractor.load_from_checkpoint(cfg.MODEL.EXTRA.CKPT)

        if freeze_feature_extractor: # will be changed to using (cfg)
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # Initialize the decoder (GPT-2 or other causal LM)
        self.decoder = AutoModelForCausalLM.from_pretrained(decoder_name)

        # Add a projection layer to map image embeddings to text embedding space
        self.image_projection = nn.Linear(image_embedding_dim, text_embedding_dim)

    def forward(self, input_ids, attention_mask=None, labels=None, image=None, patches=None, **kwargs):
        """
        Forward pass for the combined model.

        Args:
            input_ids: Input token IDs for the decoder.
            attention_mask: Attention mask for the decoder.
            labels: Labels for the decoder (optional, for training).
            image: Global image input for the FeatureExtractor.
            patches: Patch input for the FeatureExtractor.

        Returns:
            Decoder outputs.
        """
        if image is None or patches is None:
            raise ValueError("Both 'image' and 'patches' inputs are required for the FeatureExtractor.")

        # Extract features from the FeatureExtractor
        with torch.set_grad_enabled(self.training):
            extracted_features = self.feature_extractor(image, patches)

        # Project extracted features to the text embedding space
        projected_image_embeddings = self.image_projection(extracted_features)

        # Get text embeddings for the input tokens
        input_embeddings = self.decoder.get_input_embeddings()(input_ids)

        # Concatenate image embeddings with text embeddings
        combined_embeddings = torch.cat([projected_image_embeddings.unsqueeze(1), input_embeddings], dim=1)

        # Update the attention mask to account for the image token
        if attention_mask is not None:
            new_attention_mask = torch.cat([
                torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device),
                attention_mask
            ], dim=1)
        else:
            new_attention_mask = torch.ones((combined_embeddings.size(0), combined_embeddings.size(1)), dtype=torch.long, device=combined_embeddings.device)

        # Forward pass through the decoder
        return self.decoder(
            inputs_embeds=combined_embeddings,
            attention_mask=new_attention_mask,
            labels=labels,
            **kwargs
        )

    def to(self, *args, **kwargs):
        # Ensure all components are moved to the correct device
        self.feature_extractor.to(*args, **kwargs)
        self.image_projection.to(*args, **kwargs)
        self.decoder.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def train(self, mode=True):
        """
        Override train mode to ensure feature extractor respects freeze settings.
        """
        super().train(mode)
        if hasattr(self, 'feature_extractor') and not any(p.requires_grad for p in self.feature_extractor.parameters()):
            self.feature_extractor.eval()  # Keep feature extractor in eval mode if frozen


class DecoderWithImageEmbedding(nn.Module):
    def __init__(self, base_model, image_embedding_dim=2816, text_embedding_dim=768, img_token_id=None):
        super().__init__()
        self.base_model = base_model
        self.text_embedding_dim = self.base_model.config.hidden_size
        self.image_projection = nn.Linear(image_embedding_dim, self.text_embedding_dim)
        self.img_token_id = img_token_id

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def forward(self, input_ids, attention_mask=None, labels=None, image_embeddings=None, **kwargs):
        if image_embeddings is not None:
            projected_image_embeddings = self.image_projection(image_embeddings)  # (B, D)
            input_embeddings = self.get_input_embeddings()(input_ids)

            combined_embeddings, new_attention_mask, new_labels = insert_before_bos(
                input_ids=input_ids,
                input_embeddings=input_embeddings,
                image_embeddings=projected_image_embeddings,
                bos_token_id=self.base_model.config.bos_token_id,
                attention_mask=attention_mask,
                labels=labels
            )

            return self.base_model.forward(
                inputs_embeds=combined_embeddings,
                attention_mask=new_attention_mask,
                labels=new_labels,
                **kwargs
            )
        else:
            return self.base_model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )

    def generate(self, input_ids=None, attention_mask=None, image_embeddings=None, **kwargs):
        if image_embeddings is not None:
            projected_image_embeddings = self.image_projection(image_embeddings)

            if input_ids is None:
                input_ids, attention_mask = prepare_generate_inputs(
                    img_token_id=self.img_token_id,
                    bos_token_id=self.base_model.config.bos_token_id,
                    device=projected_image_embeddings.device,
                    batch_size=projected_image_embeddings.size(0)
                )

            input_embeddings = self.get_input_embeddings()(input_ids)
            
            combined_embeddings, new_attention_mask, _ = insert_before_bos(
                input_ids=input_ids,
                input_embeddings=input_embeddings,
                image_embeddings=projected_image_embeddings,
                bos_token_id=self.base_model.config.bos_token_id,
                attention_mask=attention_mask
            )

            return self.base_model.generate(
                inputs_embeds=combined_embeddings,
                attention_mask=new_attention_mask,
                **kwargs
            )
        else:
            return self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )

    def to(self, *args, **kwargs):
        self.image_projection.to(*args, **kwargs)
        return super().to(*args, **kwargs)
    

class DecoderWithLoRA(nn.Module):
    def __init__(self, lora_model, peft_config, img_token_id):  # ⭕ 오타 수정
        super().__init__()
        self.lora_model = lora_model
        self.peft_config = peft_config
        self.img_token_id = img_token_id  # ✅ 올바른 변수명 적용
        self.text_embedding_dim = self.lora_model.config.hidden_size

        # Extract the image projection layer from the original model
        if hasattr(lora_model.base_model.model, 'image_projection'):
            self.image_projection = lora_model.base_model.model.image_projection
        else:
            self.image_projection = nn.Linear(2816, self.text_embedding_dim)

    def forward(self, input_ids, attention_mask=None, labels=None, image_embeddings=None, **kwargs):
        if image_embeddings is not None:
            projected_image_embeddings = self.image_projection(image_embeddings)
            input_embeddings = self.lora_model.get_input_embeddings()(input_ids)

            combined_embeddings, new_attention_mask, new_labels = insert_before_bos(
                input_ids=input_ids,
                input_embeddings=input_embeddings,
                image_embeddings=projected_image_embeddings,
                bos_token_id=self.lora_model.config.bos_token_id,
                attention_mask=attention_mask,
                labels=labels
            )

            return self.lora_model(
                inputs_embeds=combined_embeddings,
                attention_mask=new_attention_mask,
                labels=new_labels,
                **kwargs
            )
        else:
            return self.lora_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )

    def generate(self, input_ids=None, attention_mask=None, image_embeddings=None, **kwargs):
        if image_embeddings is not None:
            projected_image_embeddings = self.image_projection(image_embeddings)

            if input_ids is None:
                input_ids, attention_mask = prepare_generate_inputs(
                    self.img_token_id,
                    self.lora_model.config.bos_token_id,
                    projected_image_embeddings.device,
                    batch_size=projected_image_embeddings.size(0)
                )

            input_embeddings = self.lora_model.get_input_embeddings()(input_ids)
            combined_embeddings, new_attention_mask, _ = insert_before_bos(
                input_ids=input_ids,
                input_embeddings=input_embeddings,
                image_embeddings=projected_image_embeddings,
                bos_token_id=self.lora_model.config.bos_token_id,
                attention_mask=attention_mask,
                labels=None
            )

            return self.lora_model.generate(
                inputs_embeds=combined_embeddings,
                attention_mask=new_attention_mask,
                **kwargs
            )
        else:
            return self.lora_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )

                
    def to(self, *args, **kwargs):
        # Ensure all components are moved to the correct device
        self.image_projection.to(*args, **kwargs)
        self.lora_model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

def get_decoder_with_image_embedding(model_name_or_path):
    # Load the decoder model with the custom class
    decoder = DecoderWithImageEmbedding.from_pretrained(model_name_or_path)
    return decoder

def get_decoder(cfg, tokenizer, model_name=None):
    model_name = cfg.DECODER.NAME  if model_name is None else model_name # e.g., "dmis-lab/meerkat-7b-v1.0, open-ai/gpt2"
    # base_model = AutoModelForCausalLM.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.eos_token_id = tokenizer.eos_token_id
    base_model.config.bos_token_id = tokenizer.bos_token_id
    base_model.resize_token_embeddings(len(tokenizer))
    # Load base model
    if cfg.DECODER.USE_LORA:
        logger.info("Loading decoder with LoRA...")
        # base_model = DecoderWithImageEmbedding.from_pretrained(model_name)
        image_token_id = tokenizer.convert_tokens_to_ids("<img>")

        if "GPT" in model_name:
            target_modules = ["c_attn"]
        else:
            target_modules = ["q_proj", "k_proj", "v_proj"]

        # LoRA 설정
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )

        lora_wrapped = get_peft_model(base_model, lora_config)
        decoder = DecoderWithLoRA(lora_wrapped, lora_config, image_token_id)
    else:
        logger.info("Loading decoder without LoRA...")
        decoder = DecoderWithImageEmbedding(base_model=base_model)

    return decoder