from transformers import AutoTokenizer
import re
import os

os.chdir('/home/yingying/sigir2024/spanbert-zh/pretrain_spanbert_zh')

batch_size=8000
zh_bert_model_type = "/home/yingying/models/bert-base-chinese"
corpus_file_path = "./data/corpus.txt"

def get_training_corpus(corpus_file):
    with open(corpus_file, "r") as f:
        while True:
            batch = []
            for _ in range(batch_size):
                line = f.readline()
                if not line:
                    break
                batch.append(line)
            if not batch:
                break
            yield get_concated_sentences(batch)


def get_concated_sentences(batch_text:list):
    textual_batch = "ls".join(batch_text)
    return textual_batch.strip().split("\n\n\n")


if __name__ == "__main__":
    training_corpus = get_training_corpus(corpus_file_path)
    for batch in training_corpus:
        print(len(batch))
        break
    tokenizer = AutoTokenizer.from_pretrained(zh_bert_model_type)
    new_tokenizer = tokenizer.train_new_from_iterator(training_corpus, 52000)
    tokenizer.save_pretrained("tokenizer_for_spanbert_base_zh")