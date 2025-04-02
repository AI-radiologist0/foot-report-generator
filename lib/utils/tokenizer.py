from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)

def load_and_prepare_tokenizer(model_name: str, additional_special_tokens=None):
    
    is_meerkat = "meerkat" in model_name.lower()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token != "[PAD]":
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        logger.info("Added [PAD] token to tokenizer.")

    if tokenizer.eos_token != "[EOS]":
        tokenizer.add_special_tokens({'eos_token': '[EOS]'})
        logger.info("Added [EOS] token to tokenizer.")

    if tokenizer.bos_token != "[BOS]":
        tokenizer.add_special_tokens({'bos_token': '[BOS]'})
        logger.info("Added [BOS] token to tokenizer.")

    if additional_special_tokens is not None:
        tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens})

    tokenizer.padding_side = "left"

    # Logging Special Token IDs.
    logger.info(f"Special Tokens: {tokenizer.special_tokens_map}")
    logger.info(f"BOS Token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    logger.info(f"EOS Token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    logger.info(f"PAD Token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    logger.info(f"Additional Special Tokens: {tokenizer.additional_special_tokens}")
    logger.info(f"Additional Special Token IDs: {tokenizer.convert_tokens_to_ids(tokenizer.additional_special_tokens)}")

    return tokenizer