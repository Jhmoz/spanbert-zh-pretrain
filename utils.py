import numpy as np
import torch

def get_word_start(anchor, word_piece_map):
    left = anchor
    #最小值到1是为了不考虑[CLS]
    while left > 1 and word_piece_map[left] == False:
        left -= 1
    return left

def get_word_end(anchor, word_piece_map):
    right = anchor + 1
    # 到len(sentence)-2 是为了不考虑[SEP]
    while right < len(word_piece_map)-1 and word_piece_map[right] == False:
        right += 1
    return right


def get_word_piece_map(sent_word_ids):
    sent_length = len(sent_word_ids)
    word_piece_map = [True] * sent_length
    previous_word_id = -1
    for i in range(sent_length):
        cur_word_id = sent_word_ids[i]
        if cur_word_id == previous_word_id:
            word_piece_map[i] = False
        previous_word_id = cur_word_id
    return word_piece_map

def is_start_word(index,word_piece_map):
    return word_piece_map[index]

def merge_intervals(intervals):
    intervals = sorted(intervals, key=lambda x: x[0])
    merged_intervals = []
    for interval in intervals:
        if not merged_intervals or merged_intervals[-1][1]+1<interval[0]:
            # 如果 merged_intervals为空 ，或者前后两个被mask掉的span没有重叠的地方,就直接append
            merged_intervals.append(interval)
        else:
            merged_intervals[-1][1]= max(merged_intervals[-1][1], interval[1])
    return merged_intervals

def pad_to_len(sbo_pair_labels,pad_token_id,max_pair_labels_lens):
    for i in range(len(sbo_pair_labels)):
        sbo_pair_labels[i] = sbo_pair_labels[i][:max_pair_labels_lens]
        this_len = len(sbo_pair_labels[i])
        for j in range(max_pair_labels_lens - this_len):
            sbo_pair_labels[i].append(pad_token_id)
    return sbo_pair_labels


def span_masking(sentence,spans,pad_token_id,mask_token_id,pad_len,masked_tokens_indices,all_token_ids):
    sentence = sentence.copy()
    sent_length = len(sentence)
    lm_labels = [pad_token_id] * sent_length
    sbo_pairs_labels = []
    spans = merge_intervals(spans) #把联通的几个小的分散的span合成一个大的span
    assert len(masked_tokens_indices) == sum([e - s + 1 for s, e in spans])
    for start,end in spans:
        lower_limit = 0
        upper_limit = sent_length
        if start>lower_limit and end<upper_limit:
            sbo_pairs_labels += [[start-1,end+1]]
            sbo_pairs_labels[-1] += [sentence[i] for i in range(start,end+1)]
        #以一个span为单位决定是用mask掩码还是随机词或本身
        rand = np.random.random()
        for i in range(start,end+1):
            assert i in masked_tokens_indices
            lm_labels[i] = sentence[i]
            if rand < 0.8:
                sentence[i] = mask_token_id
            elif rand<0.9:
                sentence[i] = np.random.choice(all_token_ids)
                # all_token_ids 是vocab中所有token的id集合
    sbo_pairs_labels = pad_to_len(sbo_pairs_labels,pad_token_id,pad_len+2)
    return sentence,lm_labels,sbo_pairs_labels

def collate_2d(pairs_targets,pad_idx,max_token_num_per_sbo_targets):
    #把一个嵌套列表的pairs_targets,做pad,使之可以直接转换成一个tensor
    max_pairs_num_of_each_sents = max(len(pair_tg) for pair_tg in pairs_targets) # 在这个batch中每个句子最多有多少个sbo_pairs
    max_span_len_of_each_sents = max_token_num_per_sbo_targets
    new_pairs_targets = pairs_targets.copy()
    for i in range(len(new_pairs_targets)):
        num_of_pairs = len(new_pairs_targets[i])
        #用每个句子的最后一个pair做填充,后面把label设为pad ignore掉
        pad_pairs = [pad_idx] * max_span_len_of_each_sents
        pad_pairs[0],pad_pairs[1] = 0,0
        new_pairs_targets[i].extend([pad_pairs]*(max_pairs_num_of_each_sents-num_of_pairs))
    return torch.tensor(new_pairs_targets)

def collate_tokens(values, PAD_TOKEN_ID):
    max_sentence_lens = max(len(v) for v in values)
    collated_values = values.copy()
    for v in collated_values:
        this_lens = len(v)
        pad_lens = max_sentence_lens - this_lens
        v.extend([PAD_TOKEN_ID] * pad_lens)
    return torch.tensor(collated_values)






