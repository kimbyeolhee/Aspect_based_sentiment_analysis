import argparse

args = argparse.ArgumentParser(description="PyTorch Template")

### WandB
args.add_argument("--wand_name", type=str, default="v_0.0")

### MODEL ###
args.add_argument(
    "--model",
    type=str,
    default="xlm-roberta-base",
    help="xlm-roberta-base | monologg/koelectra-base-v3-discriminator",
)
args.add_argument("--classifier_hidden_size", type=int, default=768)
args.add_argument("--classifier_dropout_prob", type=int, default=0.1)


### HyperParameters ###
args.add_argument("--batch_size", type=int, default=32)
args.add_argument("--learning_rate", type=float, default=3e-5)
args.add_argument("--eps", type=float, default=1e-8)
args.add_argument("--num_train_epochs", type=int, default=10)
args.add_argument("--max_grad_norm", type=float, default=1.0)

## Training ##
args.add_argument(
    "--full_finetuning", type=lambda s: s.lower() in ["true", "1"], default=True
)


### DATA PATH ###
args.add_argument(
    "--train_data_dir", type=str, default="./data/nikluge-sa-2022-train.jsonl"
)
args.add_argument(
    "--valid_data_dir", type=str, default="./data/nikluge-sa-2022-dev.jsonl"
)
args.add_argument(
    "--test_data_dir", type=str, default="./data/nikluge-sa-2022-test.jsonl"
)

### SAVED MODEL PATH ###
args.add_argument(
    "--entity_property_model_path", type=str, default="./saved_models/entity_property/"
)
args.add_argument("--polarity_model_path", type=str, default="./saved_models/polarity/")

### LOADED MODEL PATH ###
args.add_argument(
    "--loaded_entity_property_model_path",
    type=str,
    default="./saved_models/entity_property/saved_model_epoch_10.pt",
)
args.add_argument(
    "--loaded_polarity_model_path",
    type=str,
    default="./saved_models/polarity/saved_model_epoch_10.pt",
)

### ETC ###
args.add_argument("--max_len", type=int, default=256)
args.add_argument(
    "--entity_property_pair",
    type=list,
    default=[
        "제품 전체#일반",
        "제품 전체#가격",
        "제품 전체#디자인",
        "제품 전체#품질",
        "제품 전체#편의성",
        "제품 전체#인지도",
        "본품#일반",
        "본품#디자인",
        "본품#품질",
        "본품#편의성",
        "본품#다양성",
        "패키지/구성품#일반",
        "패키지/구성품#디자인",
        "패키지/구성품#품질",
        "패키지/구성품#편의성",
        "패키지/구성품#다양성",
        "브랜드#일반",
        "브랜드#가격",
        "브랜드#디자인",
        "브랜드#품질",
        "브랜드#인지도",
    ],
)


config = args.parse_args()
