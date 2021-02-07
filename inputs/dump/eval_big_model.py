import torch.nn as nn
import transformers
import torch
import pandas as pd
import numpy as np

dataframe_india_kanoon = pd.read_csv('IndianKanoon-Manually-Reviewed-Dataset_edited_3.csv')
for model_desc in ['bert-large-uncased_128_ds6_bvl']:
    model_column_name = 'pred_'+'_'.join([k for i in model_desc.split('_') for k in i.split('-')])
    if model_column_name in dataframe_india_kanoon.columns.tolist():
        dataframe_india_kanoon.drop(columns=[model_column_name], inplace=True)
    DEVICE = "cuda"
    LINEAR_INPUT = 1024
    if '128' in model_desc.split('_'):
        MAX_LEN = 128
    else:
        MAX_LEN = 64
    print(model_desc)
    print(MAX_LEN)
#     TRAIN_BATCH_SIZE = 8
#     VALID_BATCH_SIZE = 4
#     EPOCHS = 10
    BERT_PATH = model_desc.split('_')[0]
    print(BERT_PATH)
    MODEL_PATH = f'./../{model_desc}.bin'
    if 'cased' in [k for i in model_desc.split('_') for k in i.split('-')]:
        print('Cased tokenization')
        TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=False)
    else:
        print('Uncased tokenization')
        TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
    # TRAINING_FILE = "./inputs/final_training_data.csv"

    class BERTBaseUncased(nn.Module):
        def __init__(self):
            super(BERTBaseUncased, self).__init__()
            self.bert = transformers.BertModel.from_pretrained(BERT_PATH,return_dict=False)
            self.bert_drop = nn.Dropout(0.3)
            self.out = nn.Linear(LINEAR_INPUT, 1)

        def forward(self, ids, mask, token_type_ids):
            _, o2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
            bo = self.bert_drop(o2)
            output = self.out(bo)
            return output
    model = BERTBaseUncased()
    if torch.cuda.is_available():
        DEVICE = 'cuda'
        model.load_state_dict(torch.load(MODEL_PATH))
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        model.load_state_dict(torch.load(MODEL_PATH,map_location=torch.device("cpu")))
    model.to(device)
    model.eval()
    def predict_sent(sent):
        inputs = TOKENIZER.encode_plus(
                sent,
                None,
                truncation=True,
                add_special_tokens=True,
                max_length=MAX_LEN,
                padding='max_length'
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

        ids = ids.to(DEVICE, dtype=torch.long)
        token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
        mask = mask.to(DEVICE, dtype=torch.long)
        with torch.no_grad():
            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

        outputs = torch.sigmoid(outputs).cpu().detach().numpy()
        return outputs[0][0]
    dataframe_india_kanoon[f'pred_{model_desc}'] = [0.0]*dataframe_india_kanoon.shape[0]
    for i in range(dataframe_india_kanoon.shape[0]):
        dataframe_india_kanoon.at[i,f'pred_{model_desc}'] = predict_sent(dataframe_india_kanoon.at[i,'English Sentences'])
    dataframe_india_kanoon.rename(axis=1,mapper=lambda x:x.replace('-','_'),inplace=True)
    dataframe_india_kanoon.to_csv('./IndianKanoon-Manually-Reviewed-Dataset_edited_3.csv',index=False)
