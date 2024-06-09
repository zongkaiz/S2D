#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn

from base import PushToHubFriendlyModel
from model import MT5Copy
from modeling_mt5 import MT5ForConditionalGeneration
from modeling_t5 import T5ForConditionalGeneration
from prefix_encoder import SyntaxPrefixEncoder


class Model(PushToHubFriendlyModel):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer

        """The prefix-tuning code"""

        self.preseqlen = args.prefix_tuning['prefix_sequence_length']
        self.mid_dim = args.prefix_tuning['mid_dim']

        print("prefix-tuning sequence length is {}.".format(self.preseqlen))

        # Load tokenizer and model.
        if self.args.model_name.startswith("google/mt5-"):
            self.pretrain_model = MT5ForConditionalGeneration.from_pretrained(self.args.model_name,
                                                                              cache_dir=self.args.cache_dir)

        elif self.args.model_name.startswith("copy+google/mt5-"):
            model_name = self.args.model_name.split('copy+', 1)[1]
            self.pretrain_model = MT5Copy.from_pretrained(model_name, cache_dir=self.args.cache_dir,
                                                          output_attentions=True)

        self.encoder = SyntaxPrefixEncoder(args)
        self.config = self.pretrain_model.config

        if isinstance(self.pretrain_model, (T5ForConditionalGeneration)):
            self.match_n_layer = self.config.num_decoder_layers
            self.match_n_head = self.config.num_heads
            self.n_embd = self.config.d_model
            self.match_n_embd = self.config.d_kv
        elif isinstance(self.pretrain_model, (MT5ForConditionalGeneration)):
            self.match_n_layer = self.config.num_decoder_layers
            self.match_n_head = self.config.num_heads
            self.n_embd = self.config.d_model
            self.match_n_embd = self.config.d_kv
        else:
            raise ValueError("Other models are not supported yet!")
        if self.args.syntax_hidden_size:
            self.n_embd = self.n_embd + self.args.syntax_hidden_size

        self.pretrain_model.resize_token_embeddings(len(self.tokenizer))

        # Prefix related.
        self.register_buffer('input_tokens', torch.arange(self.preseqlen).long())#这一行代码创建了一个名为 'input_tokens' 的缓冲区（buffer），并将其内容初始化为一个长整型的Tensor，其中包含从0到self.preseqlen-1的整数。缓冲区通常用于存储不需要梯度更新的参数，如常量或固定的索引。

        self.wte = nn.Embedding(self.preseqlen, self.n_embd)#word token embeddings 这一行代码定义了一个嵌入层（Embedding Layer），名为 'wte'。这个嵌入层的目的是将输入的整数标记（在这里是 'input_tokens' 缓冲区中的整数）映射到一个低维的连续向量空间，其中 self.preseqlen 是输入标记的数量，self.n_embd 是嵌入向量的维度。
        self.control_trans = nn.Sequential(#定义了一个Sequential容器，其中包含了一系列层
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        )
        if self.args.model['knowledge_usage'] == 'separate':
            self.knowledge_trans = nn.Sequential(
                nn.Linear(self.n_embd, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
            )

        self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans_enc = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        )
        if self.args.model['knowledge_usage'] == 'separate':
            self.knowledge_trans_enc = nn.Sequential(
                nn.Linear(self.n_embd, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
            )

        self.wte_dec = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans_dec = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        )

        # Knowledge prompt.
        if self.args.model['knowledge_usage'] == 'separate':
            self.knowledge_trans_dec = nn.Sequential(
                nn.Linear(self.n_embd, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
            )

        self.dropout = nn.Dropout(args.prefix_tuning['prefix_dropout'])
        self.past_prompt = args.past_prompt
        self.pretrain_model.past_prompt = args.past_prompt


    def get_prompt(self, bsz=None, sample_size=1, description=None, knowledge=None):
        old_bsz = bsz
        bsz = bsz * sample_size
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1)
        temp_control = self.wte(input_tokens)#使用嵌入层 self.wte 处理输入标记 input_tokens，将其映射为连续的向量表示
        if description is not None:
            temp_control = temp_control + description.repeat_interleave(sample_size, dim=0).unsqueeze(1)
        past_key_values = self.control_trans(temp_control)  # bsz, seqlen, layer*emb
        if knowledge is not None:
            past_key_values = torch.cat(
                [past_key_values, self.knowledge_trans(knowledge.repeat_interleave(sample_size, dim=0))], dim=1)

        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(#将张量的形状重新调整为指定的大小
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        # Cross prefix
        temp_control_dec = self.wte_dec(input_tokens)
        if description is not None:
            temp_control_dec = temp_control_dec + description.repeat_interleave(sample_size, dim=0).unsqueeze(1)
        past_key_values_dec = self.control_trans_dec(
            temp_control_dec
        )  # bsz, seqlen, layer*emb
        if knowledge is not None:
            past_key_values_dec = torch.cat(
                [past_key_values_dec, self.knowledge_trans_dec(knowledge.repeat_interleave(sample_size, dim=0))], dim=1)

        bsz, seqlen, _ = past_key_values_dec.shape
        past_key_values_dec = past_key_values_dec.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values_dec = self.dropout(past_key_values_dec)
        past_key_values_dec = past_key_values_dec.permute([2, 0, 3, 1, 4]).split(2)

        # Encoder prefix
        input_tokens_enc = (
            self.input_tokens.unsqueeze(0).expand(old_bsz, -1)
        )
        temp_control_enc = self.wte_enc(input_tokens_enc)
        if description is not None:
            temp_control_enc = temp_control_enc + description.unsqueeze(1)
        past_key_values_enc = self.control_trans_enc(
            temp_control_enc
        )  # bsz, seqlen, layer*emb
        if knowledge is not None:
            past_key_values_enc = torch.cat([past_key_values_enc, self.knowledge_trans_enc(knowledge)], dim=1)

        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(
            bsz_enc,
            seqlen,
            self.match_n_layer * 2,
            self.match_n_head,
            self.match_n_embd,
        )
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val in enumerate(past_key_values):
            temp = dict()
            temp["decoder_prompt"] = {
                "prev_key": key_val[0].contiguous(),
                "prev_value": key_val[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz, seqlen)
                    .to(key_val.device)
                    .bool()
                # bsz, preseqlen
            }
            key_val_dec = past_key_values_dec[i]
            temp["cross_attention_prompt"] = {
                "prev_key": key_val_dec[0].contiguous(),
                "prev_value": key_val_dec[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz, seqlen)
                    .to(key_val_dec.device)
                    .bool(),
            }
            key_val_enc = past_key_values_enc[i]
            temp["encoder_prompt"] = {
                "prev_key": key_val_enc[0].contiguous(),
                "prev_value": key_val_enc[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen)
                    .to(key_val_enc.device)
                    .bool(),
            }
            result.append(temp)

        return result

    def get_description_representation(self, kwargs):
        if self.args.model['use_description'] and self.args.model['map_description']:
            description_input_ids = kwargs.pop("description_input_ids")
            description_attention_mask = kwargs.pop("description_attention_mask")
            if self.args.bert.location in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]:
                description_outputs = self.pretrain_model.encoder(
                    input_ids=description_input_ids,
                    attention_mask=description_attention_mask,
                )
                description = description_outputs.last_hidden_state[:, 0]  # TODO: the first token from the encoder.
            elif self.args.bert.location in ["facebook/bart-base", "facebook/bart-large"]:
                description_outputs = self.pretrain_model.model.encoder(
                    input_ids=description_input_ids,
                    attention_mask=description_attention_mask,
                )
                description = description_outputs.last_hidden_state[:, 0]  # TODO: the first token from the encoder.
            else:
                raise ValueError()
        else:
            description = None

        return description

    def get_knowledge_representation(self, encoder_input_ids, encoder_attention_mask, encoder_pos_ids,
                                     encoder_syntax_mask, prefix_ids):
        if self.args.model['knowledge_usage'] == 'separate':
            knowledge_outputs = self.encoder(encoder_input_ids=encoder_input_ids,
                                             encoder_attention_mask=encoder_attention_mask,
                                             encoder_pos_ids=encoder_pos_ids,
                                             encoder_syntax_mask=encoder_syntax_mask)#编码依赖结构，对应论文（4、5、6)式
            knowledge = knowledge_outputs
            batch_len, prefix_len = prefix_ids.size()
            _, _, hidden_size = knowledge.size()
            prefix_input = torch.FloatTensor(batch_len, prefix_len, hidden_size).cuda()
            for i in range(batch_len):
                prefix_input[i] = knowledge[i][prefix_ids[i]]
            # for i in range(batch_len):
            #     prefix_input[i] += knowledge[i][prefix_ids[0]]
            # prefix_input = prefix_input / 2
        elif self.args.model['knowledge_usage'] == 'concatenate':
            prefix_input = None
        else:
            raise ValueError()

        return prefix_input

    def forward(self, batch):
        bsz = batch.enc_idxs.shape[0]#bsz:batch_size

        # Encode description.
        description_representation = self.get_description_representation(batch)

        # Encode knowledge.
        encoder_input_ids = batch.encoder_input_ids
        encoder_attention_mask = batch.encoder_attention_mask
        encoder_pos_ids = batch.encoder_pos_ids
        encoder_syntax_mask = batch.encoder_syntax_mask
        prefix_ids = batch.prefix_ids
        knowledge_representation = self.get_knowledge_representation(
            encoder_input_ids=encoder_input_ids,
            encoder_attention_mask=encoder_attention_mask,
            encoder_pos_ids=encoder_pos_ids,
            encoder_syntax_mask=encoder_syntax_mask,
            prefix_ids=prefix_ids,
        )

        past_prompt = self.get_prompt(
            bsz=bsz, description=description_representation, knowledge=knowledge_representation,
        )
        if self.past_prompt:
            outputs = self.pretrain_model(
                input_ids=batch.enc_idxs,
                attention_mask=batch.enc_attn,
                decoder_input_ids=batch.dec_idxs,
                decoder_attention_mask=batch.dec_attn,
                labels=batch.lbl_idxs,
                past_prompt=past_prompt,
                return_dict=True,
            )
        else:
            outputs = self.pretrain_model(
                input_ids=batch.enc_idxs,
                attention_mask=batch.enc_attn,
                decoder_input_ids=batch.dec_idxs,
                decoder_attention_mask=batch.dec_attn,
                labels=batch.lbl_idxs,
                return_dict=True,
            )
        # outputs = self.pretrain_model(
        #     input_ids=batch.enc_idxs,
        #     attention_mask=batch.enc_attn,
        #     labels=batch.lbl_idxs,
        #     past_prompt=past_prompt,
        #     return_dict=True,
        # )
        loss = outputs['loss']
        return loss

    def predict(self, batch, num_beams=1, max_length=100):
        self.eval()
        with torch.no_grad():
            if num_beams == 1:
                self.pretrain_model._cache_input_ids = batch.enc_idxs
            else:
                expanded_return_idx = (
                    torch.arange(batch.enc_idxs.shape[0]).view(-1, 1).repeat(1, num_beams).view(-1).to(
                        batch.enc_idxs.device)
                )
                input_ids = batch.enc_idxs.index_select(0, expanded_return_idx)
                self.pretrain_model._cache_input_ids = input_ids

            bsz = batch.enc_idxs.shape[0]

            # Encode description.
            description_representation = self.get_description_representation(batch)

            # Encode knowledge.
            encoder_input_ids = batch.encoder_input_ids
            encoder_attention_mask = batch.encoder_attention_mask
            encoder_pos_ids = batch.encoder_pos_ids
            encoder_syntax_mask = batch.encoder_syntax_mask
            prefix_ids = batch.prefix_ids
            knowledge_representation = self.get_knowledge_representation(
                encoder_input_ids=encoder_input_ids,
                encoder_attention_mask=encoder_attention_mask,
                encoder_pos_ids=encoder_pos_ids,
                encoder_syntax_mask=encoder_syntax_mask,
                prefix_ids=prefix_ids,
            )

            past_prompt = self.get_prompt(
                bsz=bsz, sample_size=num_beams, description=description_representation,
                knowledge=knowledge_representation,
            )
            if self.past_prompt:
                outputs = self.pretrain_model.generate(
                    input_ids=batch.enc_idxs,
                    attention_mask=batch.enc_attn,
                    past_prompt=past_prompt,
                    use_cache=True,
                    num_beams=num_beams,
                    max_length=max_length,
                    forced_bos_token_id=None)
            else:
                outputs = self.pretrain_model.generate(
                    input_ids=batch.enc_idxs,
                    attention_mask=batch.enc_attn,
                    use_cache=True,
                    num_beams=num_beams,
                    max_length=max_length,
                    forced_bos_token_id=None)

        # decode outputs
        final_output = []
        for bid in range(len(batch.enc_idxs)):
            output_sentence = self.tokenizer.decode(outputs[bid], skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=True)
            final_output.append(output_sentence)
        self.train()

        return final_output
