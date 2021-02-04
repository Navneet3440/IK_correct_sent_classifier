import transformers

DEVICE = "cuda"
MAX_LEN = 128
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 20
BERT_PATH = "bert-base-uncased"
DATASET_FILE_COUNT = 5
MODEL_PATH = f"./inputs/{BERT_PATH}_{MAX_LEN}_ds{DATASET_FILE_COUNT}_bvl.bin"
TRAINING_FILE = f"./inputs/final_training_data_{DATASET_FILE_COUNT}.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
LINEAR_INPUT_SIZE = 768