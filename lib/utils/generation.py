import torch

def generate_with_prompt(decoder, tokenizer, image_embeddings, prompt_text, max_length=128, **gen_kwargs):
    device = image_embeddings.device
    batch_size = image_embeddings.size(0)

    prompt_ids = tokenizer(prompt_text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
    prompt_ids = prompt_ids.expand(batch_size, -1)

    img_ids = torch.full((batch_size, 1), decoder.img_token_id, dtype=torch.long, device=device)
    bos_ids = torch.full((batch_size, 1), decoder.config.bos_token_id, dtype=torch.long, device=device)

    input_ids = torch.cat([img_ids, bos_ids, prompt_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)

    return decoder.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        **gen_kwargs
    )

def generate_with_structured_prompt(decoder, tokenizer, image_embeddings, prompt_dict, max_new_tokens=128, **gen_kwargs):
    device = image_embeddings.device
    batch_size = image_embeddings.size(0)

    prompt_text = " ".join([f"{section} {text}" for section, text in prompt_dict.items()])
    prompt_ids = tokenizer(prompt_text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
    prompt_ids = prompt_ids.expand(batch_size, -1)

    img_ids = torch.full((batch_size, 1), decoder.img_token_id, dtype=torch.long, device=device)
    bos_ids = torch.full((batch_size, 1), tokenizer.bos_token_id, dtype=torch.long, device=device)

    input_ids = torch.cat([img_ids, bos_ids, prompt_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)

    return decoder.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        **gen_kwargs
    )
