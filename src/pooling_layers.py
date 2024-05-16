from config import CFG
from transformers import AutoConfig
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")


def get_last_hidden_state(backbone_outputs):
    last_hidden_state = backbone_outputs[0]
    return last_hidden_state


def get_all_hidden_states(backbone_outputs):
    all_hidden_states = torch.stack(backbone_outputs[1])
    return all_hidden_states


def get_input_ids(inputs):
    return inputs["input_ids"]


def get_attention_mask(inputs):
    return inputs["attention_mask"]


class AttentionPooling(nn.Module):
    def __init__(self, backbone_model, hiddendim_fc, dropout):
        super(AttentionPooling, self).__init__()
        self.num_hidden_layers = AutoConfig.from_pretrained(backbone_model).num_hidden_layers
        self.hidden_size = AutoConfig.from_pretrained(backbone_model).hidden_size
        self.hiddendim_fc = hiddendim_fc
        self.dropout = nn.Dropout(dropout)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.hidden_size))
        self.q = nn.Parameter(torch.from_numpy(q_t)).float().to(self.device)
        w_ht = np.random.normal(loc=0.0, scale=0.1, size=(self.hidden_size, self.hiddendim_fc))
        self.w_h = nn.Parameter(torch.from_numpy(w_ht)).float().to(self.device)

        self.output_dim = self.hiddendim_fc

    def attention(self, h):
        v = torch.matmul(self.q, h.transpose(-2, -1)).squeeze(1)
        v = F.softmax(v, -1)
        v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
        v = torch.matmul(self.w_h.transpose(1, 0), v_temp).squeeze(2)
        return v

    def forward(self, inputs, backbone_outputs):
        all_hidden_states = get_all_hidden_states(backbone_outputs)
        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze() for layer_i in range(1, self.num_hidden_layers + 1)], dim=-1)
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out = self.attention(hidden_states)
        out = self.dropout(out)
        return out


class ConcatPooling(nn.Module):
    def __init__(self, backbone_model, num_pooling_layers):
        super(ConcatPooling, self).__init__()
        self.n_layers = num_pooling_layers
        self.output_dim = AutoConfig.from_pretrained(backbone_model).hidden_size * num_pooling_layers

    def forward(self, inputs, backbone_outputs):
        all_hidden_states = get_all_hidden_states(backbone_outputs)
        concatenate_pooling = torch.cat([all_hidden_states[-(i + 1)] for i in range(self.n_layers)], -1)
        concatenate_pooling = concatenate_pooling[:, 0]
        return concatenate_pooling


class LSTMPooling(nn.Module):
    def __init__(self, backbone_model, hidden_lstm_size, dropout_rate, bidirectional, is_lstm=True):
        super(LSTMPooling, self).__init__()

        self.num_hidden_layers = AutoConfig.from_pretrained(backbone_model).num_hidden_layers
        self.hidden_size = AutoConfig.from_pretrained(backbone_model).hidden_size
        self.hidden_lstm_size = hidden_lstm_size
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.is_lstm = is_lstm
        self.output_dim = self.hidden_size * 2 if self.bidirectional else self.hidden_size

        if self.is_lstm:
            self.lstm = nn.LSTM(
                self.hidden_size,
                self.hidden_lstm_size,
                bidirectional=self.bidirectional,
                batch_first=True
            )
        else:
            self.lstm = nn.GRU(
                self.hidden_size,
                self.hidden_lstm_size,
                bidirectional=self.bidirectional,
                batch_first=True
            )

        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, inputs, backbone_outputs):
        all_hidden_states = get_all_hidden_states(backbone_outputs)
        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze() for layer_i in range(1, self.num_hidden_layers + 1)], dim=-1)
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out, _ = self.lstm(hidden_states, None)
        out = self.dropout(out[:, -1, :])
        return out


class MeanMaxPooling(nn.Module):
    def __init__(self, backbone_model):
        super(MeanMaxPooling, self).__init__()
        self.output_dim = AutoConfig.from_pretrained(backbone_model).hidden_size * 2

    def forward(self, inputs, backbone_outputs):
        last_hidden_state = get_last_hidden_state(backbone_outputs)
        mean_pooling_embeddings = torch.mean(last_hidden_state, 1)
        _, max_pooling_embeddings = torch.max(last_hidden_state, 1)
        mean_max_embeddings = torch.cat((mean_pooling_embeddings, max_pooling_embeddings), 1)
        return mean_max_embeddings


class MeanPooling(nn.Module):
    def __init__(self, backbone_model):
        super(MeanPooling, self).__init__()
        self.output_dim = AutoConfig.from_pretrained(backbone_model).hidden_size

    def forward(self, inputs, backbone_outputs):
        attention_mask = get_attention_mask(inputs)
        last_hidden_state = get_last_hidden_state(backbone_outputs)

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class WeightedLayerPooling(nn.Module):
    def __init__(self, backbone_model, layer_start, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.num_hidden_layers = AutoConfig.from_pretrained(backbone_model).num_hidden_layers
        self.layer_start = layer_start
        self.layer_weights = layer_weights if layer_weights is not None else nn.Parameter(torch.tensor([1] * (self.num_hidden_layers + 1 - self.layer_start), dtype=torch.float))
        self.output_dim = AutoConfig.from_pretrained(backbone_model).hidden_size

    def forward(self, inputs, backbone_outputs):
        all_hidden_states = get_all_hidden_states(backbone_outputs)
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average[:, 0]


def get_pooling_layer():
    if CFG.pooling_type == "mean_pooling":
        return MeanPooling(backbone_model=CFG.backbone_model)

    elif CFG.pooling_type == "meanmax_pooling":
        return MeanMaxPooling(backbone_model=CFG.backbone_model)

    elif CFG.pooling_type == "weighted_layer_pooling":
        return WeightedLayerPooling(
            backbone_model=CFG.backbone_model,
            layer_start=CFG.layer_start,
            layer_weights=None
        )

    elif CFG.pooling_type == "attention_pooling":
        return AttentionPooling(
            backbone_model=CFG.backbone_model,
            hiddendim_fc=CFG.hiddendim_fc,
            dropout=CFG.dropout
        )

    elif CFG.pooling_type == "concat_pooling":
        return ConcatPooling(
            backbone_model=CFG.backbone_model,
            num_pooling_layers=CFG.num_pooling_layers
        )

    elif CFG.pooling_type == "lstm_pooling":
        return LSTMPooling(
            backbone_model=CFG.backbone_model,
            hidden_lstm_size=CFG.hidden_lstm_size,
            dropout_rate=CFG.dropout_rate,
            bidirectional=CFG.bidirectional,
            is_lstm=True
        )

    elif CFG.pooling_type == "gru_pooling":
        return LSTMPooling(
            backbone_model=CFG.backbone_model,
            hidden_lstm_size=CFG.hidden_lstm_size,
            dropout_rate=CFG.dropout_rate,
            bidirectional=CFG.bidirectional,
            is_lstm=False
        )

    else:
        raise ValueError(f"Invalid Pooling Type: {CFG.pooling_type}")
