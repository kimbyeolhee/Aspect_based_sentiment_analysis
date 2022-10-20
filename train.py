import json
import numpy as np

import torch, gc
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from models.optimizer import get_optimizer
from trainer.trainer import Trainer

from dataset.dataloader import get_dataloader
from models.utils import get_model
from configs.configs import config
from utils.utils import get_labels, jsonlload

import wandb

# Set random seed
SEED = 12
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)  # type: ignore
torch.backends.cudnn.deterministic = True  # type: ignore
torch.backends.cudnn.benchmark = True  # type: ignore


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔥device : ", device)

    labels = get_labels()
    label_id_to_name = labels["label_id_to_name"]  # ["True", "False"]
    polarity_id_to_name = labels[
        "polarity_id_to_name"
    ]  # ["positive", "negative", "neutral"]
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
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    # DataLoader
    entity_property_train_dataloader, polarity_train_dataloader = get_dataloader(
        train_data, tokenizer, config
    )
    entity_property_dev_dataloader, polarity_dev_dataloader = get_dataloader(
        dev_data, tokenizer, config
    )

    # Load model
    entity_property_model = get_model(
        config, num_label=len(label_id_to_name), len_tokenizer=len(tokenizer)
    )
    entity_property_model.to(device)

    polarity_model = get_model(
        config, num_label=len(polarity_id_to_name), len_tokenizer=len(tokenizer)
    )
    polarity_model.to(device)

    # Entity_property_model Optimizer
    entity_property_optimizer = get_optimizer(config, entity_property_model)

    epochs = config.num_train_epochs
    total_steps = epochs * len(entity_property_train_dataloader)

    entity_property_scheduler = get_linear_schedule_with_warmup(
        entity_property_optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # Polarity_model Optimizer
    polarity_optimizer = get_optimizer(config, polarity_model)

    epochs = config.num_train_epochs
    total_steps = epochs * len(polarity_train_dataloader)

    polarity_scheduler = get_linear_schedule_with_warmup(
        polarity_optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    print("🔥Training Start --- Entity Property Model")
    wandb.init(
        project="20221009-Entity_property", entity="kimbyeolhee", name=config.wand_name
    )
    wandb.config.update(config)
    # Entity_property_model Train
    entity_property_model_trainer = Trainer(
        config,
        entity_property_model,
        None,
        entity_property_optimizer,
        device,
        entity_property_train_dataloader,
        entity_property_dev_dataloader,
        entity_property_scheduler,
        config.entity_property_model_path,
    )
    print("🔥🔥🔥Training🔥🔥🔥")
    gc.collect()
    torch.cuda.empty_cache()

    entity_property_model_trainer.train(label_len=len(label_id_to_name))

    # Polarity_model Train
    wandb.init(project="20221009-Polarity", entity="kimbyeolhee", name=config.wand_name)
    wandb.config.update(config)
    print("🔥Training Start --- Polarity Model")
    polarity_model_trainer = Trainer(
        config,
        polarity_model,
        None,
        polarity_optimizer,
        device,
        polarity_train_dataloader,
        polarity_dev_dataloader,
        polarity_scheduler,
        config.polarity_model_path,
    )
    print("🔥🔥🔥Training🔥🔥🔥")
    gc.collect()
    torch.cuda.empty_cache()
    polarity_model_trainer.train(label_len=len(polarity_id_to_name))

    print("Training END!!")


if __name__ == "__main__":
    config = config

    main(config)
