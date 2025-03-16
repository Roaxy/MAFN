import torch
import torch.nn as nn
import torch.nn.functional as F

class GRURNNModuleA(nn.Module):
    def __init__(self, rnn_dim, dropout, return_seq):
        super(GRURNNModuleA, self).__init__()
        self.return_seq = return_seq
        self.gru = nn.GRU(rnn_dim, rnn_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, word_embs):
        if self.dropout:
            word_embs = self.dropout(word_embs)
        gru_out, _ = self.gru(word_embs)
        merged_out = gru_out[:, :, :gru_out.size(2) // 2] + gru_out[:, :, gru_out.size(2) // 2:]

        # 只在不返回序列时取最后一个时间步
        if not self.return_seq:
            merged_out = merged_out[:, -1, :]

        return merged_out

class GRURNNModuleS(nn.Module):
    def __init__(self, rnn_dim, dropout, return_seq):
        super(GRURNNModuleS, self).__init__()
        self.return_seq = return_seq
        self.gru = nn.GRU(rnn_dim, rnn_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, word_embs):
        if self.dropout:
            word_embs = self.dropout(word_embs)
        gru_out, _ = self.gru(word_embs)

        if not self.return_seq:
            gru_out = gru_out[:, -1, :]

        return gru_out

class NLPModel(nn.Module):
    def __init__(self, rnn_dim, bidirectional=True, dropout=0.1, lang_att=True, return_raw=False, return_seq=True):
        super(NLPModel, self).__init__()
        self.bidirectional = bidirectional
        self.lang_att = lang_att
        self.return_raw = return_raw
        if bidirectional:
            self.rnn_module = GRURNNModuleA(rnn_dim, dropout, return_seq)
        else:
            self.rnn_module = GRURNNModuleS(rnn_dim, dropout, return_seq)

        if lang_att:
            self.attention_dense = nn.Linear(rnn_dim, rnn_dim)
            self.attention_dropout = nn.Dropout(0.1)

    def forward(self, q_input):
        rnn_out = self.rnn_module(q_input)  # 这里输出应该为 (batchsize, seq_length, features)

        if self.lang_att:
            attention_weights = torch.tanh(self.attention_dense(rnn_out))
            attention_weights = self.attention_dropout(attention_weights)
            attention_weights = F.softmax(attention_weights, dim=1)

            weighted_rnn_out = rnn_out * attention_weights
            rnn_sum = torch.sum(weighted_rnn_out, dim=1)  # (batchsize, features)

            if self.return_raw:
                return rnn_sum, rnn_out  # (batchsize, features), (batchsize, seq_length, features)
            else:
                return rnn_out
        else:
            return rnn_out  # (batchsize, seq_length, features)