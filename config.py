import transformers

DEVICE = "cuda"
MAX_LEN = 128
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 4
EPOCHS = 20
ACM_GRAD_NUM_BATCH = 4
ACC_CUTOFF = 0.75
TRAINING_MODE = 'bvl'
BERT_PATH = "bert-large-cased"
DATASET_FILE_COUNT = 9
MODEL_PATH = f"./inputs/{BERT_PATH}_{MAX_LEN}_ds{DATASET_FILE_COUNT}_{TRAINING_MODE}_ga{ACM_GRAD_NUM_BATCH}.bin"
MODEL_PATH_2 = f"./inputs/dump/{BERT_PATH}_{MAX_LEN}_ds{DATASET_FILE_COUNT}_"
TRAINING_FILE = f"./inputs/final_training_data_{DATASET_FILE_COUNT}.csv"
if 'cased' in BERT_PATH.split('-'):
    TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=False)
else:
    TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
if 'large' in BERT_PATH.split('-'):
    LINEAR_INPUT_SIZE = 1024
else:
    LINEAR_INPUT_SIZE = 768
