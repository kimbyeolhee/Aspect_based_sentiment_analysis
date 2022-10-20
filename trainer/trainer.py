import torch

import os
from tqdm import trange
from models.metrics import evaluate

from configs.configs import config
import wandb


class Trainer:
    def __init__(
        self,
        config,
        model,
        criterion,
        optimizer,
        device,
        train_data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
        saved_path=None,
    ):
        self.config = config

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_dataloader = train_data_loader
        self.valid_dataloader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.saved_path = saved_path

        self.epochs = config.num_train_epochs

    def train_per_one_batch(self, batch, total_loss):
        """한 batch를 학습하고 total_loss를 업데이트하여 반환

        Args:
            batch : dataloader의 batch
        Returns:
            total_loss
        """
        batch = tuple(t.to(self.device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        self.model.zero_grad()

        loss, _ = self.model(b_input_ids, b_input_mask, b_labels)
        loss.backward()
        total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(
            parameters=self.model.parameters(),
            max_norm=self.config.max_grad_norm,
        )
        self.optimizer.step()
        self.lr_scheduler.step()
        return total_loss

    def train(self, label_len):
        if not os.path.exists(self.saved_path):
            os.makedirs(self.saved_path)

        epoch_step = 0

        for _ in trange(self.epochs, desc="Epoch"):
            self.model.train()
            epoch_step += 1
            total_loss = 0

            for step, batch in enumerate(self.train_dataloader):
                total_loss = self.train_per_one_batch(batch, total_loss)

            avg_train_loss = total_loss / len(self.train_dataloader)
            print(f"Epoch: {epoch_step} | avg train loss: {avg_train_loss}")
            wandb.log({"train_loss": avg_train_loss})

            model_saved_path = (
                self.saved_path + "saved_model_epoch_" + str(epoch_step) + ".pt"
            )
            torch.save(self.model.state_dict(), model_saved_path)

        # evaluation
        print("🔥evaluation🔥")
        self.model.eval()

        pred_list = []
        label_list = []
        total_loss = 0

        for batch in self.valid_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                loss, logits = self.model(b_input_ids, b_input_mask, b_labels)

            predictions = torch.argmax(logits, dim=-1)
            pred_list.extend(predictions)
            label_list.extend(b_labels)

        avg_valid_loss = total_loss / len(self.valid_dataloader)
        eval_acc = evaluate(label_list, pred_list, label_len)
        wandb.log({"avg_valid_loss": avg_valid_loss, "eval_acc": eval_acc})
