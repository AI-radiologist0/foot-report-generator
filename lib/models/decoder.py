from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers import GenerationMixin, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from .flamingoAttn import FlamingoCrossAttention
from .feature_extractor import get_model
from utils.utils import insert_before_bos, prepare_generate_inputs
from torch import nn
import torch
from peft import PeftModel  # Import LoRA wrapper
import logging
logger = logging.getLogger(__name__)  # 기존 logger와 동일한 인스턴스 사용

def build_causal_attention_mask(input_ids, past_key_values, num_heads):
    B, T_q = input_ids.shape
    device = input_ids.device
    dtype = torch.float32

    past_len = past_key_values[0][0].size(-2) if past_key_values else 0
    T_k = past_len + T_q

    # causal mask: [T_q, T_k]
    causal = torch.tril(torch.ones((T_q, T_k), device=device, dtype=dtype))  # (T_q, T_k)
    causal = causal[None, None, :, :].expand(B, num_heads, T_q, T_k)  # (B, H, T_q, T_k)

    return (1.0 - causal) * torch.finfo(dtype).min

class FeatureExtractor(nn.Module):
    def __init__(self, cfg, pretrained=True, return_seqeunce=True, **kwarg):
        super(FeatureExtractor, self).__init__()

        self.target_classes = cfg.DATASET.TARGET_CLASSES
        self.is_binary = len(self.target_classes) == 2
        self.output_dim = 1 if self.is_binary else len(self.target_classes)
        self.return_sequence = return_seqeunce

        self.global_feature_dim, self.global_branch, self.local_feature_dim, self.local_branch = get_model(cfg, pretrained=pretrained)
        
        self.patch_proj = nn.Linear(self.local_feature_dim, self.global_feature_dim)

        # Validation checks
        assert isinstance(self.global_feature_dim, int), "Global feature dimension must be an integer"
        assert isinstance(self.local_feature_dim, int), "Local feature dimension must be an integer"
        assert isinstance(self.global_branch, nn.Module), "Global branch must be a PyTorch module"
        assert isinstance(self.local_branch, nn.Module), "Local branch must be a PyTorch module"

        logging.info(f"local: {self.local_feature_dim}, global: {self.global_feature_dim}")
        
            

        self.classifier = nn.Sequential(
            nn.Linear(self.global_feature_dim + self.local_feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, self.output_dim)
        )
        if cfg.MODEL.EXTRA.USE_CKPT and cfg.MODEL.EXTRA.CKPT:
            logging.info(f"feature extractor load ckpt flag: {cfg.MODEL.EXTRA.USE_CKPT}")
            self.load_from_checkpoint(cfg.MODEL.EXTRA.CKPT)
            

    def forward(self, image, patches):
        # Global features
        global_features = self.global_branch.forward_features(image)

        if global_features.dim() == 4: # (B, 7, 7, 768)
            global_features = global_features.mean(dim=[1, 2])  # (B, C)
        elif global_features.dim() == 3:
            global_features = global_features.mean(dim=1)       # (B, C)
        global_features = global_features.unsqueeze(1)          # (B, 1, C)

        # Local (patch) features
        B, N, C, H, W = patches.shape
        patches = patches.view(B * N, C, H, W)
        local_features = self.local_branch(patches)             # (B*N, D)
        local_features = self.patch_proj(local_features)         # (B*N, D) -> (B*N, C)
        local_features = local_features.view(B, N, -1)          # (B, N, D)

        # Combine to form a sequence of image tokens
        image_tokens = torch.cat([global_features, local_features], dim=1)  # (B, 1+N, C)

        if self.return_sequence:
            return image_tokens  # <-- return image token sequence

        # [기존 분류 목적 분기 유지]
        combined_features = torch.cat((global_features.squeeze(1), local_features.mean(dim=1)), dim=1)
        if self.is_binary:
            return torch.sigmoid(self.classifier(combined_features))
        
        return self.classifier(combined_features)

    def load_from_checkpoint(self, checkpoint_path, map_location=None):
        """
        Load model weights from a checkpoint.

        Args:
            checkpoint_path (str): Path to the .pth file.
            map_location (str or torch.device, optional): Device to map the checkpoint. Defaults to None.
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        self.load_state_dict(checkpoint)
        logging.info(f"Model weights loaded from {checkpoint_path}")


class DecoderWithFeatureExtractor(nn.Module):
    def __init__(self, cfg, base_model, feature_extractor_ckpt=None, image_embedding_dim=768, text_embedding_dim=768, freeze_feature_extractor=False):
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
        self.feature_extractor = FeatureExtractor(cfg, pretrained=True, return_seqeunce=True)
        # self.feature_extractor.load_from_checkpoint(cfg.MODEL.EXTRA.CKPT)

        if freeze_feature_extractor: # will be changed to using (cfg)
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # Initialize the decoder (GPT-2 or other causal LM)
        # self.decoder = AutoModelForCausalLM.from_pretrained(base_model)
        self.decoder = base_model

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
            raise ValueError("Both 'image' and 'patches' inputs are required.")

        # feature_extractor → (B, N_img_tokens, D_feat)
        with torch.set_grad_enabled(self.training):
            extracted_features = self.feature_extractor(image, patches)  # → (B, N, image_embedding_dim)

        # # image token projection → (B, N, D_text)
        projected_image_embeddings = self.image_projection(extracted_features)  # 복수 토큰 형태

        # text token embedding → (B, T, D_text)
        input_embeddings = self.decoder.get_input_embeddings()(input_ids)

        # 시퀀스 연결 → (B, N + T, D_text)
        combined_embeddings = torch.cat([projected_image_embeddings, input_embeddings], dim=1)

        # attention mask도 확장
        if attention_mask is not None:
            B, N_img_tokens, _ = projected_image_embeddings.shape
            img_attention = torch.ones((B, N_img_tokens), dtype=attention_mask.dtype, device=attention_mask.device)
            new_attention_mask = torch.cat([img_attention, attention_mask], dim=1)
        else:
            new_attention_mask = torch.ones((combined_embeddings.size(0), combined_embeddings.size(1)), dtype=torch.long, device=combined_embeddings.device)

        # 6. Label도 이미지 토큰만큼 -100 padding 추가 (CrossEntropyLoss에서 무시됨)
        if labels is not None:
            B, T = labels.shape
            img_token_len = projected_image_embeddings.shape[1]
            ignore_labels = torch.full(
                (B, img_token_len),
                fill_value=-100,
                dtype=labels.dtype,
                device=labels.device
            )
            new_labels = torch.cat([ignore_labels, labels], dim=1)
        else:
            new_labels = None
        
        
        # decoder forward
        return self.decoder(
            inputs_embeds=combined_embeddings,
            attention_mask=new_attention_mask,
            labels=new_labels,
            **kwargs
        )

    def to(self, *args, **kwargs):
        # Ensure all components are moved to the correct device
        self.feature_extractor.to(*args, **kwargs)
        self.image_projection.to(*args, **kwargs)
        self.decoder.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'freeze_feature_extractor') and self.freeze_feature_extractor:
            self.feature_extractor.eval()
            
    @torch.no_grad()
    def generate_from_image(
        self,
        image,
        patches,
        tokenizer,
        max_new_tokens=50,
        do_sample=False,
        top_k=50,
        top_p=0.95,
        bos_token_id=None,
        eos_token_id=None,
        device=None,
        **generate_kwargs
    ):
        """
        Generate text from image inputs using the decoder-only multimodal model.
        """
        self.eval()
        device = device or next(self.parameters()).device
        image, patches = image.to(device), patches.to(device)

        # Safe fallback for special tokens
        bos_token_id = bos_token_id or tokenizer.bos_token_id or tokenizer.cls_token_id or tokenizer.pad_token_id
        eos_token_id = eos_token_id or tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id

        # 1. Feature extraction + projection
        image_tokens = self.image_projection(self.feature_extractor(image, patches))  # (B, N_img_tokens, D)

        # 2. BOS token embedding
        bos_embed = self.decoder.get_input_embeddings()(
            torch.full((image.size(0), 1), bos_token_id, dtype=torch.long, device=device)
        )
        inputs_embeds = torch.cat([image_tokens, bos_embed], dim=1)

        # 3. Generate
        outputs = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            early_stopping=True,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **generate_kwargs
        )


        return tokenizer.batch_decode(outputs, skip_special_tokens=True)


class DecoderWithFlamingoAdapter(nn.Module, GenerationMixin):
    def __init__(self, cfg, base_model):
        super().__init__()
        self.cfg = cfg
        self.device = 'cuda' if cfg.DEVICE == "GPU" else 'cpu'
        self.image_encoder = self.build_image_encoder(cfg).to(self.device)
        self.decoder = base_model
        self.decoder_config = self.decoder.config
        self.decoder_config.is_encoder_decoder = False
        self.hidden_dim = self.decoder.config.hidden_size
        self.num_heads = self.decoder.config.num_attention_heads

        self.image_proj = nn.Sequential(
            nn.Linear(768, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )

        self.insert_layer_idx = cfg.DECODER.FLAMINGO.ADAPTER_LAYERS
        self.cross_attn_blocks = nn.ModuleDict({
            str(idx): FlamingoCrossAttention(self.hidden_dim, self.num_heads)
            for idx in self.insert_layer_idx
        })

    def build_image_encoder(self, cfg):
        return FeatureExtractor(cfg, pretrained=True, return_seqeunce=True)

    def can_generate(self):
        return True

    def forward(
        self,
        input_ids,
        image=None,
        patches=None,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        **kwargs
    ):
        # 1. 이미지 인코딩
        with torch.set_grad_enabled(self.training):
            img_tokens = self.image_encoder(image, patches)  # (B, N, D_img)
        img_tokens = self.image_proj(img_tokens)  # → (B, N, D_model)

        # 2. 텍스트 임베딩
        inputs_embeds = self.decoder.get_input_embeddings()(input_ids)
        hidden_states = inputs_embeds

        # num_heads 추출
        num_heads = self.decoder.config.n_head if hasattr(self.decoder.config, "n_head") else self.decoder.config.num_attention_heads

        # 마스크 생성
        attention_mask = build_causal_attention_mask(
            input_ids=input_ids,
            past_key_values=past_key_values,
            num_heads=num_heads
        )
        # 4. Transformer block loop
        use_cache = True
        new_past_key_values = [] if use_cache else None

        for i, block in enumerate(self.decoder.transformer.h):
            past = past_key_values[i] if past_key_values is not None and len(past_key_values) > i else None

            outputs = block(
                hidden_states,
                layer_past=past,
                attention_mask=attention_mask,
                use_cache=use_cache
            )
            hidden_states = outputs[0]

            if use_cache:
                new_past_key_values.append(outputs[1])

            # 5. Flamingo-style cross-attn
            if str(i) in self.cross_attn_blocks:
                hidden_states = self.cross_attn_blocks[str(i)](hidden_states, img_tokens)

        # 6. Final projection
        logits = self.decoder.lm_head(hidden_states)

        # 7. Loss (only for language tokens)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutputWithPast(
            logits=logits,
            loss=loss,
            past_key_values=tuple(new_past_key_values) if use_cache else None,
            hidden_states=None,
            attentions=None
        )


class FlamingoWrapperForGeneration(nn.Module, GenerationMixin):
    def __init__(self, flamingo_model, tokenizer):
        super().__init__()
        self.flamingo = flamingo_model
        self.tokenizer = tokenizer
        self.config = flamingo_model.decoder.config
        self.config.is_encoder_decoder = False
        self.generation_config = GenerationConfig.from_model_config(self.config)
        self.main_input_name = "input_ids"
        self._supports_cache_class = True
        self.device = next(self.parameters()).device

    def forward(self, input_ids, image=None, patches=None, attention_mask=None, past_key_values=None, labels=None, **kwargs):
        
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device) if labels is not None else None
        
        return self.flamingo(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            image=image,
            patches=patches
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "image": kwargs.get("image"),
            "patches": kwargs.get("patches")
        }

    def can_generate(self):
        return True

    def _reorder_cache(self, past_key_values, beam_idx):
        """
        HuggingFace beam search를 위한 필수 함수.
        beam index에 따라 past_key_values를 재정렬함.
        """
        reordered_past = []
        for layer_past in past_key_values:
            reordered_layer = tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            reordered_past.append(reordered_layer)
        return tuple(reordered_past)



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
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    # base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.eos_token_id = tokenizer.eos_token_id
    base_model.config.bos_token_id = tokenizer.bos_token_id
    base_model.resize_token_embeddings(len(tokenizer))
    if cfg.DECODER.FLAMINGO.USE_FLAMINGO:
        decoder = DecoderWithFlamingoAdapter(cfg, base_model)
        return FlamingoWrapperForGeneration(decoder, tokenizer)
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
        if cfg.MODEL.EXTRA.FREEZE:
            logger.info("Loading decoder without LoRA...")
            decoder = DecoderWithImageEmbedding(base_model=base_model)
        else:
            logger.info("Loading DecoderWithFeatureExtractor")
            decoder = DecoderWithFeatureExtractor(cfg, base_model=base_model)
    return decoder