from collections import OrderedDict
import sys
import torch
from torch import nn
from torch.nn import functional as F
from bert.modeling_bert import BertModel
from lib.clip import build_model
from lib.language_backbone import NLPModel
from lib.lang_encoder import RNNEncoder

class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers=1, bidirectional=False):
        super(SimpleLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, 
                            batch_first=True, bidirectional=bidirectional)

    def forward(self, input_labels,effective_lengths):
        self.embedding = nn.Embedding(effective_lengths.max().item(), 512).to(input_labels.device)
        embedded = self.embedding(input_labels)
        output, (hidden, cell) = self.lstm(embedded)
        return output, hidden


def load_weights(model, load_path):
    dict_trained = torch.load(load_path)['model']
    dict_new = model.state_dict().copy()
    for key in dict_new.keys():
        if key in dict_trained.keys():
            dict_new[key] = dict_trained[key]
    model.load_state_dict(dict_new)
    del dict_new
    del dict_trained
    torch.cuda.empty_cache()
    print('load weights from {}'.format(load_path))
    return model


class _LAVTSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier):
        super(_LAVTSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x, l_feats, l_mask):
        input_shape = x.shape[-2:]
        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features

        x = self.classifier(x_c4, x_c3, x_c2, x_c1)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        return x


class LAVT(_LAVTSimpleDecode):
    pass


###############################################
# LAVT One: put BERT inside the overall model #
###############################################
class _LAVTOneSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier, args):
        super(_LAVTOneSimpleDecode, self).__init__()
        # clip_model = torch.jit.load(args.clip_pretrain,
        #                     map_location="cpu").eval()
        # self.clip = build_model(clip_model.state_dict(), args.word_len).float()
        # self.bgru=NLPModel(rnn_dim=512, bidirectional=True, dropout=0.1, lang_att=True, return_raw=False)
        # self.lstm=SimpleLSTM(vocab_size=20, 
        #                      hidden_size=512, 
        #                      n_layers=1, 
        #                      bidirectional=True)
        # self.rnn_encoder = RNNEncoder(vocab_size=20,
        #                         word_embedding_size=512,
        #                         word_vec_size=512,
        #                         hidden_size=512,
        #                         bidirectional=1>0,
        #                         input_dropout_p=0.5,
        #                         dropout_p=0.2,
        #                         n_layers=1,
        #                         rnn_type='lstm',
        #                         variable_lengths=1>0)
        
        self.backbone = backbone
        self.classifier = classifier
        self.text_encoder = BertModel.from_pretrained(args.ck_bert)
        self.text_encoder.pooler = None
        # self.embedding = nn.Embedding(1000, 512)
    def forward(self, x, text, l_mask):
        input_shape = x.shape[-2:]
        ### language inference ###
        # _,state=self.clip.encode_text(text) #(1,20,512) clip_text_encode
        #text(1,20)
        
        # text=text.unsqueeze(-1).expand(-1, -1, 512).float()
        # l_feats=self.bgru(text) #(1,20,512) clip_text_encode 
        
        l_feats = self.text_encoder(text, attention_mask=l_mask)[0]  # (6, 10, 768)
        
        
        # effective_lengths = l_mask.sum(dim=1)  # (batch_size,)
        # text = text[:, :effective_lengths]
        # l_feats=self.rnn_encoder(text,effective_lengths)[0]
        
        # l_feats=self.lstm(text,effective_lengths)[0]
        #l_feats(1,768,20)
        l_feats = l_feats.permute(0, 2, 1)  # (B, 768, N_l)
        l_mask = l_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)
        ##########################
        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4  = features   # e.g. x_c1:[B, 128, 120, 120], x_c2:[B, 256, 60, 60], x_c3:[B, 512, 30, 30], x_c4:[B, 1024, 15, 15]
        x = self.classifier(x_c4, x_c3, x_c2, x_c1)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        return x


class LAVTOne(_LAVTOneSimpleDecode):  #change
    pass
