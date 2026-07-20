import torch
from typing import Any
from transformers import PreTrainedTokenizerBase

class DynamicTextCollator:
    """
    Коллатор для сборки батчей, токенизации и динамического паддинга.
    """
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizerBase, 
        max_length: int = 512,
        text_column: str = "text",
        target_column: str = "label",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.target_column = target_column

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [feature[self.text_column] for feature in features]
        
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Безопасная проверка: добавляем таргеты только если они есть в батче (нужно для train/val)
        if self.target_column in features[0]:
            targets = [feature[self.target_column] for feature in features]
            # Для классификации оставляем torch.long. 
            # Если позже добавишь регрессию, здесь нужно будет поменять на torch.float
            batch["labels"] = torch.tensor(targets, dtype=torch.long)
            
        return batch