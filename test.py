import numpy as np
import copy

import torch
from transformers import AutoTokenizer
from dataset.dataset import get_dataset
from torch.utils.data import DataLoader
from models.model import RoBertaBaseClassifier

from configs.configs import config
from utils.util import jsonlload, jsondump


def predict_from_korean_form(tokenizer, ep_model, p_model, data, config, device):
    label_id_to_name = ["True", "False"]
    polarity_id_to_name = ["positive", "negative", "neutral"]

    ep_model.to(device)
    ep_model.eval()

    temp = []

    for sentence in data:
        form = sentence["sentence_form"]
        sentence["annotation"] = []

        if type(form) != str:
            print("form type is not str", form)
            continue

        for pair in config.entity_property_pair:
            tokenized_data = tokenizer(
                form,
                pair,
                padding="max_length",
                max_length=config.max_len,
                truncation=True,
            )

            input_ids = torch.tensor([tokenized_data["input_ids"]]).to(device)
            attention_mask = torch.tensor([tokenized_data["attention_mask"]]).to(device)

            with torch.no_grad():
                _, ep_logits = ep_model(input_ids, attention_mask)

            ep_predictions = torch.argmax(ep_logits, dim=-1)
            ep_result = label_id_to_name[ep_predictions[0]]

            if ep_result == "True":
                print("🦔")
                with torch.no_grad():
                    _, p_logits = p_model(input_ids, attention_mask)

                p_predictions = torch.argmax(p_logits, dim=-1)
                p_result = polarity_id_to_name[p_predictions[0]]

                sentence["annotation"].append([pair, p_result])
                temp.append(sentence["annotation"])
    print("🔥END")
    return temp


def main(config):
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### tokenizer ###
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    ### Dataset ###
    test_data = jsonlload(config.test_data_dir)
    # entity_property_test_dataset, polarity_test_dataset = get_dataset(
    #     test_data, tokenizer, config
    # )

    # ### DataLoader ###
    # entity_property_test_dataloader = DataLoader(
    #     entity_property_test_dataset, shuffle=True, batch_size=config.batch_size
    # )
    # polarity_test_dataloader = DataLoader(
    #     polarity_test_dataset, shuffle=True, batch_size=config.batch_size
    # )

    ### Load Model ###
    entity_property_model = RoBertaBaseClassifier(
        config, num_label=len(label_id_to_name), len_tokenizer=len(tokenizer)
    )
    entity_property_model.load_state_dict(
        torch.load(config.loaded_entity_property_model_path, map_location=device)
    )
    entity_property_model.to(device)
    entity_property_model.eval()

    polarity_model = RoBertaBaseClassifier(
        config, num_label=len(polarity_id_to_name), len_tokenizer=len(tokenizer)
    )
    polarity_model.load_state_dict(
        torch.load(config.loaded_polarity_model_path, map_location=device)
    )
    polarity_model.to(device)
    polarity_model.eval()

    # Predict
    pred_data = predict_from_korean_form(
        tokenizer,
        entity_property_model,
        polarity_model,
        copy.deepcopy(test_data),
        config,
        device,
    )
    jsondump(pred_data, "./pred_data.json")


if __name__ == "__main__":
    config = config

    main(config)
