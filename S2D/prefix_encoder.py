import torch
import torch.nn as nn
from transformers import MT5EncoderModel
from transformer import TransformerEncoderLayer


class SyntaxPrefixEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.model_name.startswith("google/mt5-"):
            self.MT5 = MT5EncoderModel.from_pretrained(config.model_name, cache_dir=config.cache_dir)
        elif config.model_name.startswith("copy+google/mt5-"):
            model_name = config.model_name.split('copy+', 1)[1]
            self.MT5 = MT5EncoderModel.from_pretrained(model_name, cache_dir=config.cache_dir,
                                                       output_attentions=True)
        else:
            raise NotImplementedError

        self.syntax_num = config.syntax_num
        self.syntax_hidden_size = config.syntax_hidden_size
        self.num_layers = config.num_layers

        self.hidden_size = self.MT5.config.hidden_size
        self.feature_hidden_size = self.hidden_size + self.syntax_hidden_size
        self.emb_pos = nn.Embedding(self.syntax_num, self.syntax_hidden_size)

        self.encoder_layer = TransformerEncoderLayer(
            d_model=self.feature_hidden_size,
            nhead=1,
            dim_feedforward=self.feature_hidden_size,
            activation='relu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=self.num_layers
        )

        for param in self.MT5.parameters():
            param.requires_grad = False

    def forward(self, encoder_input_ids, encoder_attention_mask, encoder_pos_ids, encoder_syntax_mask):
        encoding = self.MT5(input_ids=encoder_input_ids,
                            attention_mask=encoder_attention_mask)
        last_hidden_state = encoding.last_hidden_state
        pos_emb = self.emb_pos(encoder_pos_ids)
        embedding = torch.cat([last_hidden_state, pos_emb], dim=-1)
        output = self.encoder(embedding, mask=encoder_syntax_mask)#编码依赖结构，对应论文（4、5、6)式
        return output
