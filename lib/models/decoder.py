from transformers import AutoModelForCausalLM
from .feature_extractor import get_feature_extractor
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

# Example usage:
# decoder_with_fe = DecoderWithFeatureExtractor(config, feature_extractor_ckpt="path/to/checkpoint.pth", freeze_feature_extractor=False)

class DecoderWithImageEmbedding(AutoModelForCausalLM):
    def __init__(self, config, image_embedding_dim=2816, text_embedding_dim=768):
        super().__init__(config)
        # Add a projection layer to map image embeddings to text embedding space
        self.image_projection = nn.Linear(image_embedding_dim, text_embedding_dim)

    def forward(self, input_ids, attention_mask=None, labels=None, image_embeddings=None, **kwargs):
        if image_embeddings is not None:
            # logger.info("Process Image_embedding")
            # 이미지 임베딩을 텍스트 임베딩 크기로 변환
            projected_image_embeddings = self.image_projection(image_embeddings)
            # 텍스트 토큰에 대한 임베딩 추출
            input_embeddings = self.get_input_embeddings()(input_ids)
            
            # BOS Token 위치 
            # [BOS] 토큰의 위치를 찾아 <img>를 그 앞에 삽입
            bos_token_id = self.lora_model.config.bos_token_id
            bos_positions = (input_ids == bos_token_id).nonzero(as_tuple=True)  # [batch_idx, position]

            # 새로운 input_ids 생성
            new_input_ids = []
            for batch_idx in range(input_ids.size(0)):
                bos_position = bos_positions[1][batch_idx]  # [BOS]의 위치
                before_bos = input_ids[batch_idx, :bos_position]  # [BOS] 이전의 토큰
                after_bos = input_ids[batch_idx, bos_position:]  # [BOS] 포함 이후의 토큰
                new_input_ids.append(torch.cat([
                    torch.tensor([self.img_token_id], dtype=input_ids.dtype, device=input_ids.device),
                    before_bos,
                    after_bos
                ]))
            new_input_ids = torch.stack(new_input_ids)

            # 새로운 attention_mask 생성
            if attention_mask is not None:
                new_attention_mask = []
                for batch_idx in range(attention_mask.size(0)):
                    bos_position = bos_positions[1][batch_idx]
                    before_bos = attention_mask[batch_idx, :bos_position]
                    after_bos = attention_mask[batch_idx, bos_position:]
                    new_attention_mask.append(torch.cat([
                        torch.tensor([1], dtype=attention_mask.dtype, device=attention_mask.device),  # <img> 토큰 활성화
                        before_bos,
                        after_bos
                    ]))
                new_attention_mask = torch.stack(new_attention_mask)
            else:
                new_attention_mask = torch.ones_like(new_input_ids, dtype=torch.long)

            # 새로운 labels 생성
            if labels is not None:
                new_labels = []
                for batch_idx in range(labels.size(0)):
                    bos_position = bos_positions[1][batch_idx]
                    before_bos = labels[batch_idx, :bos_position]
                    after_bos = labels[batch_idx, bos_position:]
                    new_labels.append(torch.cat([
                        torch.tensor([-100], dtype=labels.dtype, device=labels.device),  # <img> 토큰은 손실 계산에서 제외
                        before_bos,
                        after_bos
                    ]))
                new_labels = torch.stack(new_labels)
            else:
                new_labels = None

            # 이미지 임베딩과 텍스트 임베딩 결합
            combined_embeddings = torch.cat([
                projected_image_embeddings.unsqueeze(1),  # <img> 임베딩
                input_embeddings
            ], dim=1)
            
            # 이미지 임베딩과 텍스트 임베딩을 concat (이미지 임베딩을 앞에 추가)
            combined_embeddings = torch.cat([projected_image_embeddings.unsqueeze(1), input_embeddings], dim=1)
            
            # attention_mask에도 이미지 토큰에 해당하는 1을 prepend
            if attention_mask is not None:
                new_attention_mask = torch.cat([
                    torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device),
                    attention_mask
                ], dim=1)
            else:
                new_attention_mask = torch.ones((combined_embeddings.size(0), combined_embeddings.size(1)), dtype=torch.long, device=combined_embeddings.device)
            
            # kwargs 내부에 image_embeddings 로깅 (이미 전달된 경우)
            if "image_embeddings" in kwargs:
                logger.info(f"✅ DecoderWithImageEmbedding.forward(): image_embeddings detected in kwargs: {kwargs['image_embeddings'].shape}")
            
            # input_ids 대신 inputs_embeds와 새 attention_mask를 사용해 forward 호출
            return super().forward(inputs_embeds=combined_embeddings, attention_mask=new_attention_mask, labels=labels, **kwargs)
        else:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)


    def to(self, *args, **kwargs):
        # Ensure the image projection layer is moved to the correct device
        self.image_projection.to(*args, **kwargs)
        return super().to(*args, **kwargs)


class DecoderWithLoRA(nn.Module):
    def __init__(self, lora_model, peft_config, img_token_id):  # ⭕ 오타 수정
        super().__init__()
        self.lora_model = lora_model
        self.peft_config = peft_config
        self.img_token_id = img_token_id  # ✅ 올바른 변수명 적용

        # Extract the image projection layer from the original model
        if hasattr(lora_model.base_model.model, 'image_projection'):
            self.image_projection = lora_model.base_model.model.image_projection
        else:
            self.image_projection = nn.Linear(2816, 768)

    def forward(self, input_ids, attention_mask=None, labels=None, image_embeddings=None, **kwargs):
        
        if image_embeddings is not None:
            # logger.info(f"✅ Processing Image Embeddings: Shape {image_embeddings.shape}")

            # 이미지 임베딩을 텍스트 임베딩 크기로 변환
            projected_image_embeddings = self.image_projection(image_embeddings)
            input_embeddings = self.lora_model.get_input_embeddings()(input_ids)

            # [BOS] 토큰의 위치를 찾아 <img>를 그 앞에 삽입
            bos_token_id = self.lora_model.config.bos_token_id
            bos_positions = (input_ids == bos_token_id).nonzero(as_tuple=True)  # [batch_idx, position]

            # 새로운 input_ids 생성
            new_input_ids = []
            for batch_idx in range(input_ids.size(0)):
                bos_position = bos_positions[1][batch_idx]  # [BOS]의 위치
                before_bos = input_ids[batch_idx, :bos_position]  # [BOS] 이전의 토큰
                after_bos = input_ids[batch_idx, bos_position:]  # [BOS] 포함 이후의 토큰
                new_input_ids.append(torch.cat([
                    torch.tensor([self.img_token_id], dtype=input_ids.dtype, device=input_ids.device),
                    before_bos,
                    after_bos
                ]))
            new_input_ids = torch.stack(new_input_ids)

            # 새로운 attention_mask 생성
            if attention_mask is not None:
                new_attention_mask = []
                for batch_idx in range(attention_mask.size(0)):
                    bos_position = bos_positions[1][batch_idx]
                    before_bos = attention_mask[batch_idx, :bos_position]
                    after_bos = attention_mask[batch_idx, bos_position:]
                    new_attention_mask.append(torch.cat([
                        torch.tensor([1], dtype=attention_mask.dtype, device=attention_mask.device),  # <img> 토큰 활성화
                        before_bos,
                        after_bos
                    ]))
                new_attention_mask = torch.stack(new_attention_mask)
            else:
                new_attention_mask = torch.ones_like(new_input_ids, dtype=torch.long)

            # 새로운 labels 생성
            if labels is not None:
                new_labels = []
                for batch_idx in range(labels.size(0)):
                    bos_position = bos_positions[1][batch_idx]
                    before_bos = labels[batch_idx, :bos_position]
                    after_bos = labels[batch_idx, bos_position:]
                    new_labels.append(torch.cat([
                        torch.tensor([-100], dtype=labels.dtype, device=labels.device),  # <img> 토큰은 손실 계산에서 제외
                        before_bos,
                        after_bos
                    ]))
                new_labels = torch.stack(new_labels)
            else:
                new_labels = None

            # 이미지 임베딩과 텍스트 임베딩 결합
            combined_embeddings = torch.cat([
                projected_image_embeddings.unsqueeze(1),  # <img> 임베딩
                input_embeddings
            ], dim=1)

        
            # combined_embeddings = torch.cat([projected_image_embeddings.unsqueeze(1), input_embeddings], dim=1)

            # new_input_ids = torch.cat([
            #     torch.full((input_ids.size(0), 1), self.img_token_id, dtype=input_ids.dtype, device=input_ids.device),  
            #     input_ids
            # ], dim=1)

            # if attention_mask is not None:
            #     new_attention_mask = torch.cat([
            #         torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device),  
            #         attention_mask
            #     ], dim=1)
            # else:
            #     new_attention_mask = None

            # if labels is not None:
            #     new_labels = torch.cat([
            #         torch.full((labels.size(0), 1), -100, dtype=labels.dtype, device=labels.device),  
            #         labels
            #     ], dim=1)

            #     if new_labels.size(1) > new_input_ids.size(1):
            #         new_labels = new_labels[:, :new_input_ids.size(1)]
            #     elif new_labels.size(1) < new_input_ids.size(1):
            #         padding = torch.full(
            #             (new_labels.size(0), new_input_ids.size(1) - new_labels.size(1)),
            #             -100,
            #             dtype=new_labels.dtype,
            #             device=new_labels.device
            #         )
            #         new_labels = torch.cat([new_labels, padding], dim=1)
            # else:
            #     new_labels = None
            

            outputs = self.lora_model(
                inputs_embeds = combined_embeddings,
                attention_mask=new_attention_mask,
                labels=new_labels,
                **kwargs
            )

            return outputs
        else:
            return self.lora_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )

    def generate(self, input_ids=None, attention_mask=None, image_embeddings=None, **kwargs):
        """Generate text using both image and text embeddings."""
        if image_embeddings is not None:
            # logger.info(f"✅ DecoderWithLoRA.generate(): image_embeddings detected! Shape: {image_embeddings.shape}")

            projected_image_embeddings = self.image_projection(image_embeddings)
            # 입력 토큰이 없는 경우 기본 입력 생성
            if input_ids is None:
                input_ids = torch.tensor(
                    [[self.img_token_id, self.lora_model.config.bos_token_id, self.lora_model.config.additional_special_tokens_ids[0]]],
                    dtype=torch.long,
                    device=projected_image_embeddings.device
                )

            # 입력 토큰에 대한 임베딩 생성
            input_embeddings = self.lora_model.get_input_embeddings()(input_ids)
            combined_embeddings = torch.cat([
                projected_image_embeddings.unsqueeze(1),  # <img> 임베딩
                input_embeddings
            ], dim=1)

            # attention_mask 생성
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            new_attention_mask = torch.cat([
                torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device),  # <img> 토큰 활성화
                attention_mask
            ], dim=1)

            # 텍스트 생성
            return self.lora_model.generate(
                inputs_embeds=combined_embeddings,
                attention_mask=new_attention_mask,
                **kwargs
            )
        else:
            # 이미지 임베딩 없이 기존 텍스트로 생성
            return self.lora_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
        #     if input_ids is None:
        #         # ✅ 텍스트 없이 이미지 임베딩만으로 생성
        #         new_input_ids = torch.full(
        #             (projected_image_embeddings.size(0), 1),
        #             self.img_token_id,
        #             dtype=torch.long,
        #             device=projected_image_embeddings.device
        #         )
        #         input_embeddings = projected_image_embeddings.unsqueeze(1)  # [batch, 1, embedding_dim]
        #     else:
        #         # ✅ 기존 텍스트 임베딩 가져오기
        #         new_input_ids = torch.cat([
        #             torch.full((input_ids.size(0), 1), self.img_token_id, dtype=input_ids.dtype, device=input_ids.device),
        #             input_ids
        #         ], dim=1)

        #         input_embeddings = self.lora_model.get_input_embeddings()(input_ids)
        #         input_embeddings = torch.cat([projected_image_embeddings.unsqueeze(1), input_embeddings], dim=1)

        #     if attention_mask is not None:
        #         # 기존 attention_mask 앞에 이미지 토큰에 해당하는 1 추가
        #         new_attention_mask = torch.cat([
        #             torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device),
        #             attention_mask
        #         ], dim=1)
        #     else:
        #         # attention_mask가 없을 경우, new_input_ids와 동일한 크기의 1로 채워진 마스크 생성
        #         new_attention_mask = torch.ones((new_input_ids.size(0), new_input_ids.size(1)), dtype=torch.long, device=new_input_ids.device)

        #     # position_ids = torch.arange(input_embeddings.size(1), dtype=torch.long, device=input_embeddings.device).unsqueeze(0)

        #     # ✅ `input_ids` 대신 `inputs_embeds`로 텍스트 생성
        #     return self.lora_model.generate(
        #         input_ids=new_input_ids,  # ✅ input_ids 대신 new_input_ids 사용
        #         inputs_embeds=input_embeddings,  # ✅ generate()에서도 inputs_embeds 사용
        #         attention_mask=new_attention_mask,
        #         **kwargs
        #     )
        # else:
        #     return self.lora_model.generate(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         **kwargs
        #     )
                
    def to(self, *args, **kwargs):
        # Ensure all components are moved to the correct device
        self.image_projection.to(*args, **kwargs)
        self.lora_model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

def get_decoder_with_image_embedding(model_name_or_path):
    # Load the decoder model with the custom class
    decoder = DecoderWithImageEmbedding.from_pretrained(model_name_or_path)
    return decoder