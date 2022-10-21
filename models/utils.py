from models.model import RoBertaBaseClassifier, KoElectraClassifier


def get_model(config, num_label, len_tokenizer):
    """config.model에 입력한 모델명에 따라 모델을 호출하는 함수

    Args:
        config : from configs.configs
        num_label (int): 분류하고자 하는 라벨 수
        len_tokenizer (int): len(tokenizer)

    Raises:
        Exception: 학습에 지원하지 않는 모델명을 입력했을 경우 에러 발생

    Returns:
        학습에 사용하고자 하는 모델
    """

    if config.model == "xlm-roberta-base":
        model = RoBertaBaseClassifier(config, num_label, len_tokenizer)

    elif config.model == "monologg/koelectra-base-v3-discriminator":
        model = KoElectraClassifier(config, num_label, len_tokenizer)

    else:
        raise Exception("check the model name again")

    return model
