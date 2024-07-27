"""
把 span_bert 的架构拿过来，继承 HuggingFace中的 BertPreTrainedModel 这个类
预训练 Span_bert 类的参照transformers当中的BertForPretraining来写
"""
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertLMPredictionHead, BertForPreTrainingOutput
from transformers.activations import ACT2FN
from transformers import BertConfig, BertModel
from torch.nn import CrossEntropyLoss
from torch import nn
import torch
from typing import Optional, Tuple
from dataclasses import dataclass
from transformers.utils import ModelOutput


class MLPWithLayerNorm(nn.Module):
    def __init__(self, config, input_size):
        super(MLPWithLayerNorm, self).__init__()
        self.config = config
        self.linear1 = nn.Linear(input_size, config.hidden_size)
        self.non_lin1 = ACT2FN[config.hidden_act] if isinstance(config.hidden_act, str) else config.hidden_act
        # SpanBert 项目中这里使用的是自定义的 BertLayerNorm 类
        # 详见 https://github.com/facebookresearch/SpanBERT/blob/main/pretraining/fairseq/models/pair_bert.py#L128
        # 但是Huggingface中定义LayerNorm都是通过nn.LayerNorm来实现
        # 我尝试了在debug的过程中将BertLayerNorm 换成 nn.LayerNorm, 发现这里F.layer_norm() 不接受数据类型为Half的变量
        # 推测因为原来的项目为了能供使用半精度训练所以不得不自定义了一个专门的BertLayerNorm
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.non_lin2 = ACT2FN[config.hidden_act] if isinstance(config.hidden_act, str) else config.hidden_act
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden):
        return self.layer_norm2(self.non_lin2(self.linear2(self.layer_norm1(self.non_lin1(self.linear1(hidden))))))


class BertPairTargetPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights, max_targets=20, position_embedding_size=200):
        super(BertPairTargetPredictionHead, self).__init__()
        self.position_embeddings = nn.Embedding(max_targets, position_embedding_size)
        self.mlp_layer_norm = MLPWithLayerNorm(config, config.hidden_size * 2 + position_embedding_size)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        # 项目里所有单独实例化一个bias的原因而不直接放在linear里去做的原因，应该都是为了能够用全0的方式来初始化参数
        self.max_targets = max_targets

    def forward(self, hidden_states, pairs):
        bs, num_pairs, _ = pairs.size()
        # pairs : [batch_size , num_pairs,2]
        bs, seq_len, dim = hidden_states.size()
        # pair indices: (bs, num_pairs)
        left, right = pairs[:, :, 0], pairs[:, :, 1]
        # left是所有的左边界的位置索引 , right 是所有右边界的位置索引
        # 下面从隐藏状态中提取左边界和右边界的向量表示
        # left.unsqueeze(2) : [bs ,num_pairs ,1]
        # left.unsqueeze(2).repeat(1, 1, dim) : [bs ,num_pairs, dim], dim=config.hidden_size
        max_index = torch.max(pairs)
        left_hidden = torch.gather(hidden_states, 1, left.unsqueeze(2).repeat(1, 1, dim))
        # left_hidden :(bs, num_pairs, dim)
        # pair states: bs * num_pairs, max_targets, dim
        # 然后将左右边界的隐藏表示重复max_targets次，方便后面与位置嵌入进行拼接
        # contiguous方法改变了多维数组在内存中的存储顺序，以便配合view方法使用, 参考 https://blog.csdn.net/qq_37828380/article/details/107855070
        left_hidden = left_hidden.contiguous().view(bs * num_pairs, dim).unsqueeze(1).repeat(1, self.max_targets, 1)
        right_hidden = torch.gather(hidden_states, 1, right.unsqueeze(2).repeat(1, 1, dim))
        right_hidden = right_hidden.contiguous().view(bs * num_pairs, dim).unsqueeze(1).repeat(1, self.max_targets, 1)
        # right_hidden/left_hidden :[ bs * num_pairs, max_targets, dim]

        # 获取位置嵌入并拼接左右边界的隐藏表示
        position_embeddings = self.position_embeddings.weight  # 这里直接把位置嵌入的权重矩阵取出来，后面采用直接拼接的方式，实现上更加简便
        # position_embeddings :(max_targets, position_embedding_size)
        # 然后,把每个挖空的span的左右边界的隐藏表示和position_embedding拼接在一起
        # 这里无论span原本的长度有多少,都是放入对应长度为max_target长度的position_embedding
        # right_hidden/left_hidden :[ bs * num_pairs, max_targets, dim]
        # position_embeddings.unsqueeze(0).repeat(bs * num_pairs, 1, 1) : [bs*num*pairs,max_targets,position_embedding_size]
        hidden_states = self.mlp_layer_norm(
            torch.cat((left_hidden, right_hidden, position_embeddings.unsqueeze(0).repeat(bs * num_pairs, 1, 1)), -1))
        # hidden_states : [bs, num_pairs ,max_targets, hidden_size]
        target_scores = self.decoder(hidden_states) + self.bias
        # target scores : [bs * num_pairs, max_targets, vocab_size]
        return target_scores


class BertLMPredictionHeadForSpanBERT(BertLMPredictionHead):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHeadForSpanBERT, self).__init__(config)
        # 这里之所以要全部重新初始化各种Head的原因在于Huggingface中的各种Head(包括LMHead和NSPHead)初始化decoder的时候权重是随机初始化的
        # 而Spanbert原项目在预训练的时候,LMPredictionHead与PairTargetHead的解码器的权重与BERT模型的嵌入权重相同，但有独立的偏置项
        self.decoder.weight = bert_model_embedding_weights


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, max_pair_targets, bert_model_embedding_weights, pair_pos_embedding_size,no_nsp=False):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config)
        self.pair_target_predictions = BertPairTargetPredictionHead(config, bert_model_embedding_weights,
                                                                    max_pair_targets)
        self.seq_relationship = None if no_nsp else nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pairs, pooled_output):
        # 这里输入的 pairs 是 一个batch中每个sentence里num_pairs个被挖空的span左边界和右边界的位置索引，他的shape是[bs,num_pairs,2]
        # sequence_output 是 bert 在最后一层的隐藏状态，他的shape是[bs,seq_lens,hidden_size]
        prediction_scores = self.predictions(sequence_output)
        pair_target_scores = self.pair_target_predictions(sequence_output, pairs)
        seq_relationship_score = None if self.seq_relationship is None else self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score, pair_target_scores


class SpanBERT(BertPreTrainedModel):
    def __init__(self, config, max_pair_targets=20, pair_pos_embedding_size=200, ignored_id=-1,
                 remove_head=False, no_nsp=False):
        super(SpanBERT, self).__init__(config)
        self.config = config
        self.ignored_id = ignored_id
        # self.bert = BertModel(config, remove_head=remove_head, remove_pooled=remove_pooled, no_nsp=no_nsp)
        # 这里的remove_head和remove_pooled决定了预训练时的原来的bert是否保留BertModel的head模块:也就是 bert.pooler为None/BertPooler
        # 检查Huggingface上的spanbert-base-cased，该模型pooler模块也是没有预训练参数的。
        # Spanbert论文和预训练的过程中都去掉了NSP任务，即no_nsp=True。
        # 我读了一下项目里相关的代码，发现只要no_nsp=True，无论remove_head和remove_pooled是怎么设置的,self.pooler都是None
        # 详见 https://github.com/facebookresearch/SpanBERT/blob/main/pretraining/fairseq/models/pair_bert.py#L618
        # 因此实例化一个bert时不再考虑remove_head和remove_pooled两个参数重写类，直接用Huggingface的BertModel来实现，no_nsp参数用该类自带的add_pooling_layer参数来代替
        self.bert = BertModel(config, add_pooling_layer=False)
        self.remove_head = remove_head
        if not remove_head:
            # 因为去掉了NSP任务,所以SpanBert的Heads跟原始的BertForPretraining略有不同
            # 其余的地方是一样的
            self.cls = BertPreTrainingHeads(config, max_pair_targets, self.bert.embeddings.word_embeddings.weight,
                                            pair_pos_embedding_size,no_nsp=no_nsp)
        self.post_init()

    def forward(self, input_ids, pairs=None, token_type_ids=None, attention_mask=None, masked_lm_labels=None,sbo_labels=None):
        # 删除了原来项目中冗余的部分,用huggingface的架构时原有写在这部分的padding和truncation的过程都在tokenizer中完成了
        # 又根据BertModelOutput的结果对应给出原来需要的变量：sequence_output,pooled_output
        # 如需对比原来的项目代码可以见 https://github.com/facebookresearch/SpanBERT/blob/main/pretraining/fairseq/models/pair_bert.py#L630
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output, pooled_output = outputs.last_hidden_state, outputs.pooler_output

        if self.remove_head:
            # 如果初始化的时候设置了不要head就可以直接返回bert的编码结果了
            return sequence_output, pooled_output

        prediction_scores, seq_relationship_score, pair_target_scores = self.cls(sequence_output, pairs, pooled_output)
        # 这里给出的seq_relation_score=None
        # paire_target_scores的shape是[bs * num_pairs, max_targets, vocab_size]
        # prediction_scores的shape [bs, seq_lens, vocab_size]

        total_loss = None
        if masked_lm_labels is not None and sbo_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=self.ignored_id)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            sbo_loss = loss_fct(pair_target_scores.view(-1, self.config.vocab_size), sbo_labels.view(-1))
            total_loss = masked_lm_loss + sbo_loss
            # next_sentence_label = None,
            # if next_sentence_label is not None:
            #     next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            #     total_loss += next_sentence_loss

        return SpanBertModelOutput(
            loss=total_loss,
            pairs_logits=pair_target_scores,
            mlm_logits=prediction_scores,
            sequence_output=sequence_output
        )

@dataclass
class SpanBertModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    pairs_logits: torch.FloatTensor = None
    mlm_logits: torch.FloatTensor = None
    sequence_output: torch.FloatTensor = None

