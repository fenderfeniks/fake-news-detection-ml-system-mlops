import logging
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

class HFTokenizerBuilder:
    """
    Фабрика для безопасной загрузки и настройки HuggingFace токенизаторов.
    """

    def __init__(
        self,
        tokenizer_name: str,
        use_fast: bool = True,
        padding_side: str = "right",
        add_eos_token: bool = False,
    ):
        self.tokenizer_name = tokenizer_name
        self.use_fast = use_fast
        self.padding_side = padding_side
        self.add_eos_token = add_eos_token

    def build(self) -> PreTrainedTokenizerBase:
        logger.info(f"Загрузка токенизатора: {self.tokenizer_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            use_fast=self.use_fast,
            add_eos_token=self.add_eos_token,
        )

        tokenizer.padding_side = self.padding_side

        if tokenizer.pad_token is None:
            logger.warning(
                f"У токенизатора {self.tokenizer_name} нет pad_token. "
                "Устанавливаем pad_token_id = eos_token_id"
            )
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer