from torch.optim import AdamW


def get_optimizer(config, model):
    """config.full_finetuning 값에 따라 full_finetuning 여부를 결정하고 사용할 optimizer를 정의함

    Args:
        config : configs.configs
        model : 사용할 모델

    Returns:
        optimizer
    """
    if config.full_finetuning:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.0,
            },
        ]

    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = AdamW(
        optimizer_grouped_parameters, lr=config.learning_rate, eps=config.eps
    )

    return optimizer
