import torch
import torch.nn as nn
from transformers import XLMRobertaModel
from transformers import ElectraModel


class SimpleClassifier(nn.Module):
    def __init__(self, config, num_label):
        super().__init__()
        self.dense = nn.Linear(
            config.classifier_hidden_size, config.classifier_hidden_size
        )
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.output = nn.Linear(config.classifier_hidden_size, num_label)

    def forward(self, features):
        # features [32, 256, 768]
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output(x)

        return x


class RoBertaBaseClassifier(nn.Module):
    def __init__(self, config, num_label, len_tokenizer):
        super(RoBertaBaseClassifier, self).__init__()

        self.num_label = num_label
        self.xlm_roberta = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.xlm_roberta.resize_token_embeddings(
            len_tokenizer
        )  # len_tokenizer = 250010

        self.labels_classifier = SimpleClassifier(config, self.num_label)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.xlm_roberta(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=None
        )
        sequence_output = outputs[0]
        logits = self.labels_classifier(sequence_output)

        loss = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_label), labels.view(-1))

        return loss, logits


class KoElectraClassifier(nn.Module):
    def __init__(self, config, num_label, len_tokenizer):
        super(KoElectraClassifier, self).__init__()

        self.num_label = num_label
        self.koelectra = ElectraModel.from_pretrained(
            "monologg/koelectra-base-v3-discriminator"
        )
        self.koelectra.resize_token_embeddings(len_tokenizer)
        self.labels_classifier = SimpleClassifier(config, self.num_label)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.koelectra(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=None
        )
        sequence_output = outputs[0]
        logits = self.labels_classifier(sequence_output)

        loss = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_label), labels.view(-1))

        return loss, logits
