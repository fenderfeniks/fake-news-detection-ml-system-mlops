from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PathsConfig:
    root_dir: str
    data_dir: str
    hf_cache_dir: str
    processed_data_dir: str
    log_dir: str


@dataclass
class HydraRunConfig:
    dir: str


@dataclass
class HydraJobConfig:
    chdir: bool


@dataclass
class HydraConfig:
    run: HydraRunConfig
    job: HydraJobConfig


@dataclass
class OptimizerConfig:
    _target_: str
    lr: float
    weight_decay: float
    _partial_: bool = True


@dataclass
class ModelModuleConfig:
    _target_: str
    num_classes: int
    optimizer_cfg: OptimizerConfig
    target_precision: float | None = None
    target_recall: float | None = None


@dataclass
class RootDataModuleConfig:
    _target_: str
    data_cfg: Any
    _recursive_: bool = False


@dataclass
class InferenceConfig:
    threshold: float


@dataclass
class TokenizerConfig:
    _target_: str
    tokenizer_name: str
    use_fast: bool
    padding_side: str
    add_eos_token: bool


@dataclass
class ModelBuilderConfig:
    _target_: str
    model_name_or_path: str
    cache_dir: str
    trust_remote_code: bool
    auto_model_class: str
    torch_dtype: str
    num_labels: int
    finetuning_type: str
    quantization_config: Any | None = None
    peft_config: Any | None = None

    # --- НАСТРОЙКИ MLFLOW REGISTRY ---
    use_mlflow_registry: bool = False
    mlflow_tracking_uri: str | None = None
    mlflow_model_name: str | None = None
    mlflow_model_alias: str | None = None


@dataclass
class FinetuningConfig:
    type: str
    peft_config: Any | None = None


@dataclass
class ModelConfig:
    tokenizer: TokenizerConfig
    builder: ModelBuilderConfig
    finetuning: FinetuningConfig
    model_name: str | None = None


@dataclass
class DataCleanerPipelineConfig:
    _target_: str
    cleaners: list[Any] = field(default_factory=list)


@dataclass
class DataLoaderConfig:
    _target_: str
    batch_size: int
    num_workers: int
    pin_memory: bool


@dataclass
class DataDataModuleConfig:
    _target_: str


@dataclass
class DataSourceConfig:
    _target_: str
    source_type: str
    raw_dir: str
    dataset_name: str | None = None
    file_name: str | None = None
    token: str | None = None
    sep: str | None = None


@dataclass
class DataConfig:
    source: DataSourceConfig
    max_length: int
    val_split_size: float
    seed: int
    preprocessing_num_workers: int
    preprocessing_batch_size: int
    cleaner: DataCleanerPipelineConfig
    collator: Any
    dataloader: DataLoaderConfig
    datamodule: DataDataModuleConfig
    paths: Any = None
    text_column: str | None = None
    target_column: str | None = None
    stratify_column: str | None = None


@dataclass
class MLFlowLoggerConfig:
    _target_: str
    experiment_name: str
    tracking_uri: str
    run_name: str
    log_model: bool
    tags: dict[str, str]


@dataclass
class TrainerConfig:
    _target_: str
    max_epochs: int
    accelerator: str
    precision: str
    logger: Any
    devices: int | str | None = None
    limit_train_batches: Any = None
    limit_val_batches: Any = None
    limit_test_batches: Any = None
    callbacks: list[Any] = field(default_factory=list)


@dataclass
class TelegramWebhookConfig:
    path: str
    url: str


@dataclass
class TelegramMessagesConfig:
    welcome: str
    error: str


@dataclass
class TelegramConfig:
    bot_token: str
    messages: TelegramMessagesConfig


@dataclass
class EnvironmentConfig:
    name: str


@dataclass
class APIConfig:
    host: str
    port: int
    domain: str
    title: str
    description: str
    version: str
    telegram_webhook: TelegramWebhookConfig
    cors_origins: list[str]
    telegram: TelegramConfig
    log_level: str | None = None


@dataclass
class OptunaConfig:
    n_trials: int
    direction: str
    metric_name: str
    enable_pruning: bool
    storage: str | None = None
    study_name: str | None = None


@dataclass
class ConfigSchema:
    seed: int
    project_name: str
    environment: EnvironmentConfig
    paths: PathsConfig
    model: ModelConfig
    data: DataConfig
    trainer: TrainerConfig
    api: APIConfig
    logger: MLFlowLoggerConfig
    model_module: ModelModuleConfig
    datamodule: RootDataModuleConfig
    hydra: HydraConfig
    optuna: OptunaConfig | None = None
    inference: InferenceConfig | None = None
