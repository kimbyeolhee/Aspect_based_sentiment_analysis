from torch.utils.data import DataLoader
from dataset.dataset import get_dataset


def get_dataloader(raw_data, tokenizer, config):
    """
    Args:
        raw_data (jsonl): jsonl 데이터
        tokenizer : 정의한 tokenizer
        config : from configs.configs

    Returns:
        속성 범주 dataloader, 감성 정보 dataloader
    """
    entity_property_dataset, polarity_dataset = get_dataset(raw_data, tokenizer, config)

    entity_property_dataloader = DataLoader(
        entity_property_dataset, shuffle=True, batch_size=config.batch_size
    )
    polarity_dataloader = DataLoader(
        polarity_dataset, shuffle=True, batch_size=config.batch_size
    )

    return entity_property_dataloader, polarity_dataloader
