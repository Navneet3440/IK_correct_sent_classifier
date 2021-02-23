import config
import dataset
import engine
import torch
import pandas as pd
import torch.nn as nn
import numpy as np

from model import BERT_CLASSIFIER
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


def run():
    dfx = pd.read_csv(config.TRAINING_FILE)
    print("Shape of datframe:",dfx.shape)
    df_train, df_valid = model_selection.train_test_split(
        dfx, test_size=0.1, random_state=42, stratify=dfx.label.values
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    print("Shape of train datframe:",df_train.shape)
    print("Shape of validation dataframe:",df_valid.shape)

    train_dataset = dataset.BERTDataset(
        sent=df_train.sentences.values, target=df_train.label.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_dataset = dataset.BERTDataset(
        sent=df_valid.sentences.values, target=df_valid.label.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )

    device = torch.device(config.DEVICE)
    model = BERT_CLASSIFIER()
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.1,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=1e-6)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_accuracy = 0
    best_eval_loss = np.inf

    for epoch in range(config.EPOCHS):
        epoch_train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        outputs, targets, epoch_eval_loss = engine.eval_fn(valid_data_loader, model, device)
        outputs = np.array(outputs) >= config.ACC_CUTOFF
        accuracy = metrics.accuracy_score(targets, outputs)
        print("Train loss = ", epoch_train_loss)
        print("Validation Loss = ", epoch_eval_loss)
        print("Accuracy Score =", accuracy)
        if config.TRAINING_MODE == 'ba':
            best_eval_loss = np.inf
        if accuracy > best_accuracy and epoch_eval_loss < best_eval_loss:
            print("Saving Model state")
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy
            best_eval_loss = epoch_eval_loss
        else:
            print("Saving model in dump folder")
            torch.save(model.state_dict(), config.MODEL_PATH_2 + f"{epoch}.bin")


if __name__ == "__main__":
    run()
