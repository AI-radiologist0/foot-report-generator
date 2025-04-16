import logging
import time
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from utils.utils import AverageMeter
from utils.generation import generate_with_structured_prompt
from core.evaluate import clean_special_tokens, compute_bertscore, compute_bleu_rouge, compute_semantic_similarity, compute_tfidf_similarity
from models.decoder import DecoderWithFeatureExtractor, DecoderWithFlamingoAdapter, FlamingoWrapperForGeneration


class ImageProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ImageProjection, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, image_embeddings):
        return self.projection(image_embeddings)


class DecoderTrainer:
    def __init__(self, cfg, model, tokenizer, feature_extractor, writer_dict):
        self.device = 'cuda' if cfg.DEVICE == "GPU" else 'cpu'
        self.feature_extractor = feature_extractor
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.writer_dict = writer_dict
        self.print_freq = cfg.PRINT_FREQ
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        self.scaler = GradScaler()
        self.accumulation_steps = cfg.TRAIN.GRADIENT_ACCUMULATION_STEPS
        self.cfg = cfg

    def update_model(self, new_model):
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
        
        self.model = new_model.to(self.device)

    def generate_image_embeddings(self, feature_extractor, batch, device):
        """
        Generate image embeddings using the feature extractor.
        Args:
            feature_extractor: The trained feature extractor model.
            dataloader: Dataloader containing image data.
            device: Device to run the model on.

        Returns:
            List of image embeddings.
        """
        feature_extractor.eval()
        embeddings = []
        with torch.no_grad():
            images = batch["images"].to(device)  # Assuming "images" key in dataset
            patches = batch["patch_tensors"].to(device)
            embedding = feature_extractor(images, patches)
            embeddings.append(embedding)
        
        return torch.cat(embeddings, dim=0)  # Combine all embeddings
        

    def train(self, epoch, data_loader, optimizer):
        logger = logging.getLogger("Training")
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter()

        self.model.train()
        end = time.time()

        for i, batch in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}", leave=False)):
            data_time.update(time.time() - end)

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            optimizer.zero_grad()

            with autocast(device_type=self.device):
                if isinstance(self.model, DecoderWithFeatureExtractor):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        image=batch["images"].to(self.device),
                        patches=batch["patch_tensors"].to(self.device)
                    )
                elif isinstance(self.model, FlamingoWrapperForGeneration):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        image=batch["images"].to(self.device),
                        patches=batch["patch_tensors"].to(self.device)
                    )
                else:
                    image_embedding = self.generate_image_embeddings(self.feature_extractor, batch, self.device)
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        image_embeddings=image_embedding
                    )
                loss = outputs.loss / self.accumulation_steps

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer=optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(optimizer=optimizer)
            self.scaler.update()

            loss_meter.update(loss.item(), input_ids.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:
                msg = f'Epoch: [{epoch}] [{i}/{len(data_loader)}] \t ' \
                    f'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s) \t' \
                    f'Data Time: {data_time.val:.3f}s ({data_time.avg:.3f}s) \t' \
                    f'Loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f})'
                logger.info(msg)

            writer = self.writer_dict['writer']
            global_steps = self.writer_dict['train_global_steps']
            writer.add_scalar('train_loss', loss_meter.val, global_steps)
            self.writer_dict['train_global_steps'] = global_steps + 1

        logger.info(f'Epoch [{epoch}] - Training Loss: {loss_meter.avg:.4f}')
        
        return loss_meter.avg

    def validate(self, epoch, data_loader):
        logger = logging.getLogger("Validation")
        self.model.eval()
        loss_meter = AverageMeter()

        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader, desc=f"Validation Epoch {epoch}", leave=False)):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                with autocast(device_type=self.device):
                    if isinstance(self.model, DecoderWithFeatureExtractor):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            image=batch["images"].to(self.device),
                            patches=batch["patch_tensors"].to(self.device)
                        )
                    elif isinstance(self.model, FlamingoWrapperForGeneration):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            image=batch["images"].to(self.device),
                            patches=batch["patch_tensors"].to(self.device)
                        )
                    else:
                        image_embedding = self.generate_image_embeddings(self.feature_extractor, batch, self.device)
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            image_embeddings=image_embedding
                        )
                    loss = outputs.loss

                loss_meter.update(loss.item(), input_ids.size(0))

        logger.info(f"[Epoch {epoch}] Validation Loss: {loss_meter.avg:.4f}")
        return loss_meter.avg


    def inference(self, epoch, data_loader, max_samples=16):
        logger = logging.getLogger("Inference")
        self.model.eval()

        sample_predictions = []
        all_preds, all_labels = [], []
        bleu_scores, rouge_l_scores = [], []

        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader, desc=f"Inference Epoch {epoch}", leave=False)):
                if isinstance(self.model, DecoderWithFeatureExtractor):
                    # FeatureExtractor 기반 이미지 생성
                    image = batch["images"].to(self.device)
                    patches = batch["patch_tensors"].to(self.device)

                    ref_lens = [len([token for token in label.tolist() if token >= 0]) for label in batch["labels"]]
                    max_new_tokens = max(ref_lens) + 20

                    predictions = self.model.generate_from_image(
                        image=image,
                        patches=patches,
                        tokenizer=self.tokenizer,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        top_p=0.9,
                        top_k=50,
                        num_beams=2
                    )
                    decoded_preds = [clean_special_tokens(p) for p in predictions]
                elif isinstance(self.model, FlamingoWrapperForGeneration):
                    image = batch["images"].to(self.device)
                    patches = batch["patch_tensors"].to(self.device)

                    ref_lens = [len([token for token in label.tolist() if token >= 0]) for label in batch["labels"]]
                    max_new_tokens = max(ref_lens) + 20

                    input_ids = self.tokenizer("Describe the image.", return_tensors="pt").input_ids.to(self.device)
                    attention_mask = torch.ones_like(input_ids).to(self.device)

                    generated = self.model.generate(
                        input_ids=input_ids.expand(image.size(0), -1),
                        attention_mask=attention_mask.expand(image.size(0), -1),
                        image=image,
                        patches=patches,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        top_k=50,
                        top_p=0.9,
                        num_beams=2,
                        temperature=0.7,
                        repetition_penalty=1.2,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                    decoded_preds = [clean_special_tokens(self.tokenizer.decode(p, skip_special_tokens=True)) for p in generated]
                else:
                    # 기존 방식 유지
                    image_embeddings = self.generate_image_embeddings(self.feature_extractor, batch, self.device)

                    ref_lens = [len([token for token in label.tolist() if token >= 0]) for label in batch["labels"]]
                    max_new_tokens = max(ref_lens) + 20

                    predictions = generate_with_structured_prompt(
                        decoder=self.model,
                        tokenizer=self.tokenizer,
                        image_embeddings=image_embeddings,
                        image=None,
                        patches=None,
                        prompt_dict={
                            "[FINDING]": "Describe findings.",
                            "[CONCLUSION]": "Summarize the result.",
                            "[RECOMMEND]": "Suggest a follow-up."
                        },
                        max_new_tokens=max_new_tokens,
                        temperature=0.7,
                        top_p=0.9,
                        num_beams=2
                    )
                    decoded_preds = [clean_special_tokens(self.tokenizer.decode(p, skip_special_tokens=False)) for p in predictions]

                # 공통 라벨 처리
                decoded_labels = [
                    clean_special_tokens(self.tokenizer.decode([token for token in l.tolist() if token >= 0], skip_special_tokens=False))
                    for l in batch["labels"]
                ]

                # metric 계산
                for j, (pred, label) in enumerate(zip(decoded_preds, decoded_labels)):
                    bleu, rouge_l = compute_bleu_rouge(label, pred)
                    bleu_scores.append(bleu)
                    rouge_l_scores.append(rouge_l)
                    all_preds.append(pred)
                    all_labels.append(label)
                    if len(sample_predictions) < max_samples:
                        sample_predictions.append({
                            "reference": label,
                            "prediction": pred,
                            "bleu_score": bleu,
                            "rouge_l": rouge_l
                        })

        # BERTScore 계산
        bert_f1_scores = compute_bertscore(all_preds, all_labels)
        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        avg_rouge = sum(rouge_l_scores) / len(rouge_l_scores)
        avg_bert = sum(bert_f1_scores) / len(bert_f1_scores)

        # 추가 metric 계산
        semantic_score_list, semantic_score = compute_semantic_similarity(all_preds, all_labels)
        tfidf_score_list, tfidf_score = compute_tfidf_similarity(all_preds, all_labels)

        for idx in range(len(sample_predictions)):
            sample_predictions[idx]["bert_f1"] = bert_f1_scores[idx]
            sample_predictions[idx]["semantic"] = semantic_score_list[idx]
            sample_predictions[idx]["tfidf"] = tfidf_score_list[idx]


        logger.info(
            f"[Epoch {epoch}] Inference Scores — BLEU: {avg_bleu:.4f}, ROUGE-L: {avg_rouge:.4f}, "
            f"BERT F1: {avg_bert:.4f}, Semantic: {semantic_score:.4f}, TF-IDF: {tfidf_score:.4f}"
        )

        return avg_bleu, avg_rouge, avg_bert, semantic_score, tfidf_score, sample_predictions


