"""
拿到一个dataset 每个元素都包括
    input_ids,token_type_ids，attention_mask
    pairs=None， sbo_labels=None  pairs是挖空了的span的左右边界的序号
    masked_lm_labels=None,
    next_sentence_label=None

"""

from transformers import BertTokenizer
from datasets import load_dataset

data_files= {
    "train": "./data/raw_corpus/train.txt",
    "validation":"./data/raw_corpus/valid.txt"
}
TOKENIZER_PATH = "./prepare_tokenizer/tokenizer_for_spanbert_base_zh"

raw_dataset=load_dataset("text",data_files=data_files)
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)



