import transformers

DEVICE = "cuda"
MAX_LEN = 128
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 20
BERT_PATH = "bert-base-uncased"
MODEL_PATH = "./inputs/model.bin"
TRAINING_FILE = "./inputs/final_training_data_5"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
LINEAR_INPUT_SIZE = 768