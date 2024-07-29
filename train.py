"""
拿到一个dataset 每个元素都包括
    input_ids,token_type_ids，attention_mask
    pairs=None， sbo_labels=None  pairs是挖空了的span的左右边界的序号
    masked_lm_labels=None,
    next_sentence_label=None

"""
from utils import *
from models import *
from transformers import set_seed, TrainingArguments, AutoTokenizer, Trainer, BertConfig
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset
from pyltp import Segmentor
import numpy as np
import math
import os
import wandb

WANDB_API_KEY = "*************************"
wandb.login()
os.environ["WANDB_PROJECT"] = "******"
os.environ["WANDB_MODE"] = "offline"

data_files = {
    "train": "./data/raw_corpus/train.txt",
    "validation": "./data/raw_corpus/valid.txt"
}
TOKENIZER_PATH = "./prepare_tokenizer/tokenizer_for_spanbert_base_zh"
WORD_SEG_PATH = "./ltp_data_v3.4.0/cws.model"
OUTPUT_PATH = "./output"
PADDING = False
TRUNCATION = True
MAX_SEQ_LENGTH = 512
DOC_STRIDE = 128
MASK_RATIO = 0.15
MAX_PAIR_TARGET = 20
SPAN_LOWER = 1
SPAN_UPPER = 10
GEOMETRIC_P = 0.2
MAX_DATA_SAMPLES = None  # 10000 #None
NUM_PROC = 72

BATCH_SIZE = 16
SAVE_STRATEGY = "epoch"
SAVE_TOTAL_LIMIT = 1
FP16 = False
GRADIENT_ACCUMULATION_STEPS = 4
DO_TRAIN = True
DO_EVAL = True
EVALUATION_STRATEGY = "epoch"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
LR_SCHEDULER_TYPE = "polynomial"
LOGGING_STEPS = 2000
NUM_TRAIN_EPOCHS = 8
WARMUP_STEPS = 10000
OVERWRITE_OUTPUT_DIR = False #True
REPORT_TO = "wandb"
RUN_NAME = "spanbert-zh-pretrain"
RESUME_FROM_CHECKPOINT= True # None

config = BertConfig(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    type_vocab_size=2,
    clamp_attention=False,
    initializer_range=0.02,
    layer_norm_eps=1e-12
)

training_args = TrainingArguments(
    output_dir=OUTPUT_PATH,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    save_strategy=SAVE_STRATEGY,
    save_total_limit=SAVE_TOTAL_LIMIT,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    do_train=DO_TRAIN,
    do_eval=DO_EVAL,
    evaluation_strategy=EVALUATION_STRATEGY,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    weight_decay=WEIGHT_DECAY,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    logging_steps=LOGGING_STEPS,
    fp16=FP16,
    overwrite_output_dir=OVERWRITE_OUTPUT_DIR,
    report_to=REPORT_TO,
    run_name=RUN_NAME,
    resume_from_checkpoint=RESUME_FROM_CHECKPOINT
)

# Detecting last checkpoint.
last_checkpoint = None
if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )
    elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        print(
            f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
            "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        )

raw_dataset = load_dataset("text", data_files=data_files)
#raw_dataset.cleanup_cache_files()
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True)
segmentor = Segmentor(WORD_SEG_PATH)
PAD_TOKEN_ID = tokenizer.pad_token_id
MASK_TOKEN_ID = tokenizer.mask_token_id

set_seed(training_args.seed)
config.vocab_size = tokenizer.vocab_size
model = SpanBERT(config=config, ignored_id=PAD_TOKEN_ID, no_nsp=True)


class PairWithSpanMaskingScheme:
    def __init__(self):
        self.mask_ratio = MASK_RATIO
        self.max_pair_targets = MAX_PAIR_TARGET
        self.span_lower = SPAN_LOWER
        self.span_upper = SPAN_UPPER
        self.pad_token_id = PAD_TOKEN_ID
        self.mask_id = MASK_TOKEN_ID
        self.span_lens = list(range(self.span_lower, self.span_upper + 1))
        self.geometric_p = GEOMETRIC_P
        self.span_lens_distrib = [self.geometric_p * (1 - self.geometric_p) ** (i - 1) for i in
                                  range(self.span_lower, self.span_upper + 1)]
        self.span_lens_distrib = [x / (sum(self.span_lens_distrib)) for x in self.span_lens_distrib]
        self.all_vocab_token_ids = [i for i in range(tokenizer.vocab_size)]
        print(self.span_lens_distrib, self.span_lens)

    def mask(self, sentence, word_ids):
        sent_length = len(sentence)
        mask_num = math.ceil(sent_length * self.mask_ratio)  # 确定掩码span的个数
        mask = set()  # 用来存放所有mask的token的index
        word_piece_map = get_word_piece_map(word_ids)  # list:[bool] ,里面第i个位置的元素代表第i个token是不是一个词的开始
        # 这个地方中文的和英文的略有区别 英文的span一定包含了一个完整的单词，但按照目前的写法我们的span只是对应了一个一个字，是有可能把一个词切开的
        spans = []
        while len(mask) < mask_num:
            span_len = np.random.choice(self.span_lens, p=self.span_lens_distrib)
            anchor = np.random.randint(1, sent_length - 1)  # 选中被mask的span的第一个token的index
            # 这么设定anchor抽样区间的原因是要避免anchor选中最开头的[cls]和最后的[sep]
            if anchor in mask:
                continue
            # 找到anchor所在的词的start_index,end_index
            anchor_word_start, anchor_word_end = get_word_start(anchor, word_piece_map), get_word_end(anchor,
                                                                                                      word_piece_map)
            spans.append([anchor_word_start, anchor_word_end])
            # 如果mask掉的token数量多于mask_num就去掉后面数量超出来的的token
            for i in range(anchor_word_start, anchor_word_end):
                if len(mask) >= mask_num:
                    break
                mask.add(i)
                spans[-1][-1] = i
                if i == sent_length - 1:
                    breakpoint()
                    print("1")
            num_words = 1
            # 如果anchor指向的字或词的长度小于span_len,则需要进一步补充下一个字或者词进来，方法与掩码anchor所在的词方法一致
            rest_words_end = anchor_word_end
            while num_words < span_len and len(mask) < mask_num and rest_words_end < sent_length - 2:
                rest_words_starts = rest_words_end
                rest_words_ends = get_word_end(rest_words_starts, word_piece_map)
                num_words += 1
                for i in range(rest_words_starts, rest_words_ends):
                    if len(mask) >= mask_num:
                        break
                    mask.add(i)
                    spans[-1][-1] = i
                    if i == sent_length - 1:
                        breakpoint()
                        print("2")
        sentence, mlm_target, pair_targets = span_masking(sentence, spans, self.pad_token_id, self.mask_id,
                                                          self.max_pair_targets, mask, self.all_vocab_token_ids)
        return sentence, mlm_target, pair_targets

    def sentences_mask(self, tokenized_sentences):
        sentences, mlm_targets, pair_targets = [], [], []
        num_sents = len(tokenized_sentences["input_ids"])
        for i in range(num_sents):
            sentence = tokenized_sentences["input_ids"][i]
            if len(sentence) - 1 <= 1:
                breakpoint()
                print("??")
                continue
            word_ids = tokenized_sentences.word_ids(i)
            masked_sentence, mlm_target, pair_target = self.mask(sentence, word_ids)
            sentences.append(masked_sentence)
            mlm_targets.append(mlm_target)
            pair_targets.append(pair_target)
        assert len(sentences)==len(mlm_targets)==len(pair_targets)
        return sentences, mlm_targets, pair_targets


masking_scheme = PairWithSpanMaskingScheme()


def prepare_features(examples):
    text_splited = [segmentor.segment(sent) for sent in examples["text"]]

    tokenized_examples = tokenizer(
        text_splited,
        truncation=TRUNCATION,
        max_length=MAX_SEQ_LENGTH,
        padding=PADDING,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        is_split_into_words=True
    )
    
    tokenized_examples["input_ids"], tokenized_examples["masked_lm_labels"],tokenized_examples[
        "pairs"] = masking_scheme.sentences_mask(tokenized_examples)
    return tokenized_examples


def data_processor(split):
    split_examples = raw_dataset[split]
    column_names = split_examples.column_names
    if MAX_DATA_SAMPLES is not None:
        split_examples = split_examples.select(range(MAX_DATA_SAMPLES))
    with training_args.main_process_first(desc=f"{split} dataset map pre-processing"):
        splited_dataset = split_examples.map(
            prepare_features,
            batched=True,
            num_proc=NUM_PROC,
            remove_columns=column_names,
            desc=f"Running tokenizer on {split} dataset"
        )
    return splited_dataset


train_dataset = data_processor("train")
eval_dataset = data_processor("validation")


def collate_fn(features):
    features_name_to_pad = ["input_ids", "attention_mask", "token_type_ids"]
    features_to_pad = [{k: v for k, v in feature.items() if k in features_name_to_pad} for
                       feature in features]
    batch = tokenizer.pad(
        features_to_pad,
        padding=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt"
    )
    pair_targets = [feature["pairs"] for feature in features]
    collated_pairs_targets = collate_2d(pair_targets, PAD_TOKEN_ID, MAX_PAIR_TARGET + 2)
    batch["pairs"] = collated_pairs_targets[:, :, :2]
    batch["sbo_labels"] = collated_pairs_targets[:, :, 2:]
    mlm_targets = [feature["masked_lm_labels"] for feature in features]
    batch["masked_lm_labels"] = collate_tokens(mlm_targets, PAD_TOKEN_ID)
    return batch


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=collate_fn
)

if training_args.do_train:
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    trainer.save_state()
else:
    trainer.model = model.from_pretrained(training_args.output_dir, config=config, ignored_id=PAD_TOKEN_ID, no_nsp=True)

if training_args.do_eval:
    metrics = trainer.evaluate()
    perplexity = math.exp(metrics["eval_loss"])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

wandb.finish()
