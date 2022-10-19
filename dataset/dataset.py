import pandas as pd

import torch
from torch.utils.data import TensorDataset

from utils.utils import get_labels


def tokenize_and_align_labels(tokenizer, form, annotations, config):
    """sentence_form과 annotation을 받아서 속성범주 일치 여부 별 tokenize된 값들이 담긴 dictionary와 감성 정보 별 tokenized된 값들이 담긴 dictionary 반환

    Args:
        tokenizer : 정의한 tokenizer
        form (str): 데이터셋 문장
        annotations (list): 해당 문장의 라벨 : [['속성 범주', [명시적 표현 정보], '감성']]
        config : from configs.configs

    Returns:
        entity_property_data_dict (dict) : 속성범주들 일치 여부(label) 별 input_ids와 attention_mask가 담긴 dictionary
        polarity_data_dict (dict) : 감성 정보와 input_ids, attention mask가 담겨있음
    """

    entity_property_data_dict = {"input_ids": [], "attention_mask": [], "label": []}
    polarity_data_dict = {"input_ids": [], "attention_mask": [], "label": []}

    labels = get_labels()
    label_name_to_id = labels["label_name_to_id"]  # {'True': 0, 'False': 1}
    polarity_name_to_id = labels[
        "polarity_name_to_id"
    ]  # {'positive': 0, 'negative': 1, 'neutral': 2}

    for pair in config.entity_property_pair:
        isPairInOpinion = False

        if pd.isna(form):
            break

        tokenized_data = tokenizer(
            form, pair, padding="max_length", max_length=config.max_len, truncation=True
        )

        for annotation in annotations:
            entity_property = annotation[0]  # 속성 범주
            polarity = annotation[2]  # 감성

        if polarity == "------------":
            continue

        if entity_property == pair:
            entity_property_data_dict["input_ids"].append(tokenized_data["input_ids"])
            entity_property_data_dict["attention_mask"].append(
                tokenized_data["attention_mask"]
            )
            entity_property_data_dict["label"].append(label_name_to_id["True"])

            polarity_data_dict["input_ids"].append(tokenized_data["input_ids"])
            polarity_data_dict["attention_mask"].append(
                tokenized_data["attention_mask"]
            )
            polarity_data_dict["label"].append(polarity_name_to_id[polarity])

            isPairInOpinion = True
            break

        if isPairInOpinion is False:
            entity_property_data_dict["input_ids"].append(tokenized_data["input_ids"])
            entity_property_data_dict["attention_mask"].append(
                tokenized_data["attention_mask"]
            )
            entity_property_data_dict["label"].append(label_name_to_id["False"])

    return entity_property_data_dict, polarity_data_dict


def get_dataset(raw_data, tokenizer, config):
    """
    Args:
        raw_data (jsonl): jsonl 데이터
        tokenizer : 정의한 tokenizer
        config : from configs.configs

    Returns:
        속성 범주 별 TensorDataset
        감성 정보 별 TensorDataset
    """

    input_ids_list = []
    attention_mask_list = []
    token_labels_list = []

    polarity_input_ids_list = []
    polarity_attention_mask_list = []
    polarity_token_labels_list = []

    # utterance: 발화
    for utterance in raw_data:
        entity_property_data_dict, polarity_data_dict = tokenize_and_align_labels(
            tokenizer, utterance["sentence_form"], utterance["annotation"], config
        )

        input_ids_list.extend(entity_property_data_dict["input_ids"])
        attention_mask_list.extend(entity_property_data_dict["attention_mask"])
        token_labels_list.extend(entity_property_data_dict["label"])

        polarity_input_ids_list.extend(polarity_data_dict["input_ids"])
        polarity_attention_mask_list.extend(polarity_data_dict["attention_mask"])
        polarity_token_labels_list.extend(polarity_data_dict["label"])

    return TensorDataset(
        torch.tensor(input_ids_list),
        torch.tensor(attention_mask_list),
        torch.tensor(token_labels_list),
    ), TensorDataset(
        torch.tensor(polarity_input_ids_list),
        torch.tensor(polarity_attention_mask_list),
        torch.tensor(polarity_token_labels_list),
    )
