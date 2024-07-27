import random

input_file= "./raw_corpus/corpus_v3.txt"
train_file = "./raw_corpus/train.txt"
valid_file = "./raw_corpus/valid.txt"

with open(train_file, "w") as train_f:
    with open(valid_file, "w") as valid_f:
        with open(input_file, "r") as input_f:
            for line in input_f:
                if line:
                    if random.uniform(0,1) > 0.9:
                        valid_f.write(line)
                    else:
                        train_f.write(line)
