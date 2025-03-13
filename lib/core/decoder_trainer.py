import logging
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.utils import AverageMeter


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

            # Load inputs from batch
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Generate image embeddings
            image_embedding = self.generate_image_embeddings(self.feature_extractor, batch, self.device)

            optimizer.zero_grad()

            # Forward pass with image embeddings and text inputs
            outputs = self.model(input_ids=input_ids, 
                                attention_mask=attention_mask, 
                                labels=labels, 
                                image_embeddings=image_embedding)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), input_ids.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:
                msg = f'Epoch: [{epoch}] [{i}/{len(data_loader)}] \t ' \
                    f'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s) \t' \
                    f'Data Time: {data_time.val:.3f}s ({data_time.avg:.3f}s) \t' \
                    f'Loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f})'
                logger.info(msg)

            # TensorBoard logging
            writer = self.writer_dict['writer']
            global_steps = self.writer_dict['train_global_steps']
            writer.add_scalar('train_loss', loss_meter.val, global_steps)
            self.writer_dict['train_global_steps'] = global_steps + 1

        logger.info(f'Epoch [{epoch}] - Training Loss: {loss_meter.avg:.4f}')
        return loss_meter.avg


    def validate(self, epoch, data_loader):
        from nltk.translate.bleu_score import sentence_bleu
        logger = logging.getLogger("Validation")
        self.model.eval()

        loss_meter = AverageMeter()
        bleu_scores = []  # BLEU 점수를 저장할 리스트

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Validation Epoch {epoch}", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Generate image embeddings
                image_embedding = self.generate_image_embeddings(self.feature_extractor, batch, self.device)

                # Forward pass with image embeddings and text inputs
                outputs = self.model(input_ids=input_ids, 
                                    attention_mask=attention_mask, 
                                    labels=labels, 
                                    image_embeddings=image_embedding)
                loss = outputs.loss
                loss_meter.update(loss.item(), input_ids.size(0))

                # Use the model's generate method for predictions
                predictions = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_embeddings=image_embedding,
                    max_length=256,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                # Decode predictions and labels
                decoded_preds = [self.tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
                decoded_labels = [
                    self.tokenizer.decode([token for token in label.tolist() if token >= 0], skip_special_tokens=True)
                    for label in labels
                ]

                # BLEU 점수 계산
                for pred, label in zip(decoded_preds, decoded_labels):
                    reference = [label.split()]  # 참조 텍스트를 토큰화
                    candidate = pred.split()  # 생성된 텍스트를 토큰화
                    bleu_score = sentence_bleu(reference, candidate)
                    bleu_scores.append(bleu_score)

        # BLEU 점수 평균 계산
        avg_bleu_score = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
        logger.info(f'Epoch [{epoch}] - Validation Loss: {loss_meter.avg:.4f}, BLEU Score: {avg_bleu_score:.4f}')
        return loss_meter.avg, avg_bleu_score