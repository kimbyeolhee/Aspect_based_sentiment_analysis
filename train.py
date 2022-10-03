import json
import numpy as np

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from models.model import RoBertaBaseClassifier
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from trainer.trainer import Trainer

from dataset.dataset import get_dataset
from configs.configs import config


# Set random seed
SEED = 12
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)  # type: ignore
torch.backends.cudnn.deterministic = True  # type: ignore
torch.backends.cudnn.benchmark = True  # type: ignore

# config.json 로드
def jsonload(fname, encoding="utf-8"):
    with open(fname, encoding=encoding) as f:
        j = json.load(f)

    return j


# jsonl 데이터 파일 읽어서 리스트에 저장
def jsonlload(fname, encoding="utf-8"):
    json_list = []

    with open(fname, encoding=encoding) as f:
        for line in f.readlines():
            json_list.append(json.loads(line))

    return json_list


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_id_to_name = ["True", "False"]
    polarity_id_to_name = ["positive", "negative", "neutral"]
    special_tokens_dict = {
        "additional_special_tokens": [
            "&name&",
            "&affiliation&",
            "&social-security-num&",
            "&tel-num&",
            "&card-num&",
            "&bank-account&",
            "&num&",
            "&online-account&",
        ]
    }

    train_data = jsonlload(config.train_data_dir)
    dev_data = jsonlload(config.valid_data_dir)

    # tokenizer 정의
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)  # 🔥

    # Dataset
    entity_property_train_dataset, polarity_train_dataset = get_dataset(
        train_data, tokenizer, config
    )
    entity_property_dev_dataset, polarity_dev_dataset = get_dataset(
        dev_data, tokenizer, config
    )

    # DataLoader
    entity_property_train_dataloader = DataLoader(
        entity_property_train_dataset, shuffle=True, batch_size=config.batch_size
    )
    entity_property_dev_dataloader = DataLoader(
        entity_property_dev_dataset, shuffle=True, batch_size=config.batch_size
    )

    polarity_train_dataloader = DataLoader(
        polarity_train_dataset, shuffle=True, batch_size=config.batch_size
    )
    polarity_dev_dataloader = DataLoader(
        polarity_dev_dataset, shuffle=True, batch_size=config.batch_size
    )

    # Load model
    entity_property_model = RoBertaBaseClassifier(
        config, num_label=len(label_id_to_name), len_tokenizer=len(tokenizer)
    )
    entity_property_model.to(device)

    polarity_model = RoBertaBaseClassifier(
        config, num_label=len(polarity_id_to_name), len_tokenizer=len(tokenizer)
    )
    polarity_model.to(device)

    # Entity_property_model Optimizer Setting
    FULL_FINETUNING = True
    if FULL_FINETUNING:
        entity_property_param_optimizer = list(entity_property_model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        entity_property_optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in entity_property_param_optimizer
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in entity_property_param_optimizer
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.0,
            },
        ]

    else:
        entity_property_param_optimizer = list(
            entity_property_model.classifier.named_parameters()
        )
        entity_property_optimizer_grouped_parameters = [
            {"params": [p for n, p in entity_property_param_optimizer]}
        ]

    entity_property_optimizer = AdamW(
        entity_property_optimizer_grouped_parameters,
        lr=config.learning_rate,
        eps=config.eps,
    )
    epochs = config.num_train_epochs
    max_grad_norm = 1.0
    total_steps = epochs * len(entity_property_train_dataloader)

    entity_property_scheduler = get_linear_schedule_with_warmup(
        entity_property_optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # Polarity_model Optimizer Setting
    if FULL_FINETUNING:
        polarity_param_optimizer = list(polarity_model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        polarity_optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in polarity_param_optimizer
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in polarity_param_optimizer
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.0,
            },
        ]
    else:
        polarity_param_optimizer = list(polarity_model.named_parameters())
        polarity_optimizer_grouped_parameters = [
            {"params": [p for n, p in polarity_param_optimizer]}
        ]

    polarity_optimizer = AdamW(
        polarity_optimizer_grouped_parameters, lr=config.learning_rate, eps=config.eps
    )
    epochs = config.num_train_epochs
    total_steps = epochs * len(polarity_train_dataloader)

    polarity_scheduler = get_linear_schedule_with_warmup(
        polarity_optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # Entity_property_model Train
    entity_property_model_trainer = Trainer(
        config,
        entity_property_model,
        None,
        entity_property_optimizer,
        None,
        device,
        entity_property_train_dataloader,
        entity_property_dev_dataloader,
        entity_property_scheduler,
        config.entity_property_model_path,
    )
    entity_property_model_trainer.train(label_len=len(label_id_to_name))

    # Polarity_model Train
    polarity_model_trainer = Trainer(
        config,
        polarity_model,
        None,
        polarity_optimizer,
        None,
        device,
        polarity_train_dataloader,
        polarity_dev_dataloader,
        polarity_scheduler,
        config.polarity_model_path,
    )
    polarity_model_trainer.train(label_len=len(polarity_id_to_name))
    print("!!END!!")


if __name__ == "__main__":
    config = config

    main(config)
