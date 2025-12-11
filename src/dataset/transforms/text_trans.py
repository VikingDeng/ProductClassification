from transformers import AutoTokenizer
from src.core import TRANSFORMS
import logging

@TRANSFORMS.register("BertTokenizer")
class BertTokenizerTransform:
    def __init__(self, model_name="bert-base-uncased", max_len=128):
        logger = logging.getLogger(__name__)
        self.max_len = max_len
        
        logger.info(f"[Tokenizer] Loading: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __call__(self, text):
        return self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )