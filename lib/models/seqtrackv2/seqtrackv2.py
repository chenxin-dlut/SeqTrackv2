"""
SeqTrack v2 Model
"""
import torch
import math
from torch import nn
import torch.nn.functional as F

from lib.utils.misc import NestedTensor

from .language_model import build_bert
from .encoder import build_encoder
from .decoder import build_decoder
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.utils.pos_embed import get_sinusoid_encoding_table, get_2d_sincos_pos_embed


class SEQTRACKV2(nn.Module):
    """ This is the base class for SeqTrackV2"""
    def __init__(self, language_interface_extractor, encoder, decoder, hidden_dim,
                 bins=1000, feature_type='x', num_frames=1, num_template=1,
                 seq_format = 'xywh', text_pooling='mean'):
        """ Initializes the model.
        Parameters:
            encoder: torch module of the encoder to be used. See encoder.py
            decoder: torch module of the decoder architecture. See decoder.py
        """
        super().__init__()
        self.language_interface_extractor = language_interface_extractor
        self.language_interface_proj = nn.Linear(language_interface_extractor.num_channels, encoder.num_channels)
        self.text_pooling = text_pooling

        self.encoder = encoder
        self.num_patch_x = self.encoder.body.num_patches_search
        self.num_patch_z = self.encoder.body.num_patches_template
        self.side_fx = int(math.sqrt(self.num_patch_x))
        self.side_fz = int(math.sqrt(self.num_patch_z))
        self.hidden_dim = hidden_dim
        self.bottleneck = nn.Linear(encoder.num_channels, hidden_dim) # the bottleneck layer, which aligns the dimmension of encoder and decoder
        self.decoder = decoder
        self.vocab_embed = MLP(hidden_dim, hidden_dim, bins+2, 3)

        self.seq_format = seq_format
        self.bins = bins

        self.num_frames = num_frames
        self.num_template = num_template
        self.feature_type = feature_type

        # Different type of visual features for decoder.
        # Since we only use one search image for now, the 'x' is same with 'x_last' here.
        if self.feature_type == 'x':
            num_patches = self.num_patch_x * self.num_frames
        elif self.feature_type == 'xz':
            num_patches = self.num_patch_x * self.num_frames + self.num_patch_z * self.num_template
        elif self.feature_type == 'token':
            num_patches = 1
        else:
            raise ValueError('illegal feature type')

        # position embeding for the decocder
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        pos_embed = get_sinusoid_encoding_table(num_patches, self.pos_embed.shape[-1], cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))



    def forward(self, template_list=None, search_list=None, text_data=None, text_src=None,
                xz=None, seq=None, mode="encoder"):
        """
        image_list: list of template and search images, template images should precede search images
        xz: feature from encoder
        seq: input sequence of the decoder
        mode: encoder or decoder.
        """
        if mode == "language":
            return self.forward_text(text_data)
        elif mode == "encoder":
            return self.forward_encoder(template_list, search_list, text_src, seq)
        elif mode == "decoder":
            return self.forward_decoder(xz, seq)
        else:
            raise ValueError

    def forward_encoder(self, template_list, search_list, text_src, seq):
        # Forward the encoder
        # TODO convert seq to instruct tokens
        seq = seq[:,0] - self.decoder.embedding.vocab_size
        xz = self.encoder(template_list, search_list, text_src, seq)
        return xz

    def forward_decoder(self, xz, sequence):

        xz_mem = xz[-1]
        B, _, _ = xz_mem.shape

        # get different type of visual features for decoder.
        if self.feature_type == 'x': # get features of all search images
            dec_mem = xz_mem[:,0:self.num_patch_x * self.num_frames]
        elif self.feature_type == 'xz': # get all features of search and template images
            dec_mem = xz_mem
        elif self.feature_type == 'token': # get an average feature vector of search and template images.
            dec_mem = xz_mem.mean(1).unsqueeze(1)
        else:
            raise ValueError('illegal feature type')

        # align the dimensions of the encoder and decoder
        if dec_mem.shape[-1] != self.hidden_dim:
            dec_mem = self.bottleneck(dec_mem)  #[B,NL,D]
        dec_mem = dec_mem.permute(1,0,2)  #[NL,B,D]

        out = self.decoder(dec_mem, self.pos_embed.permute(1,0,2).expand(-1,B,-1), sequence)
        out = self.vocab_embed(out) # embeddings --> likelihood of words

        return out

    def inference_encoder(self, template_list, search_list, text_src, multi_modal_vision, seq):
        # Forward the encoder
        if not multi_modal_vision:
            xz = self.encoder.forward_rgb(template_list, search_list)
        else:
            seq = seq[:, 0] - self.decoder.embedding.vocab_size
            xz = self.encoder(template_list, search_list, text_src, seq)
        return xz


    def inference_decoder(self, xz, sequence, window=None):
        # Forward the decoder
        xz_mem = xz[-1]
        B, _, _ = xz_mem.shape

        # get different type of visual features for decoder.
        if self.feature_type == 'x':
            dec_mem = xz_mem[:,0:self.num_patch_x]
        elif self.feature_type == 'xz':
            dec_mem = xz_mem
        elif self.feature_type == 'token':
            dec_mem = xz_mem.mean(1).unsqueeze(1)
        else:
            raise ValueError('illegal feature type')

        if dec_mem.shape[-1] != self.hidden_dim:
            dec_mem = self.bottleneck(dec_mem)  #[B,NL,D]
        dec_mem = dec_mem.permute(1,0,2)  #[NL,B,D]

        out = self.decoder.inference(dec_mem,
                                    self.pos_embed.permute(1,0,2).expand(-1,B,-1),
                                    sequence, self.vocab_embed,
                                    window, self.seq_format)

        return out

    def forward_text(self, text_data: NestedTensor):
        # language bert
        text_fea = self.language_interface_extractor(text_data)
        text_src, text_mask = text_fea.decompose()  # seq_len * b * HIDDEN_DIM , seq_len * b
        text_src = self.language_interface_proj(text_src)
        if self.text_pooling == 'mean':
            text_src = torch.mean(text_src, 1)
        elif self.text_pooling == 'max':
            text_src, _ = torch.max(text_src, 1)
        elif self.text_pooling == None:
            text_src = text_src
        else:
            raise ValueError('Wrong pooling type for text features')
        return text_src


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def build_seqtrackv2(cfg):
    encoder = build_encoder(cfg)
    decoder = build_decoder(cfg)
    language_extractor = build_bert(cfg)
    model = SEQTRACKV2(
        language_extractor,
        encoder,
        decoder,
        hidden_dim=cfg.MODEL.HIDDEN_DIM,
        bins = cfg.MODEL.BINS,
        feature_type = cfg.MODEL.FEATURE_TYPE,
        num_frames = cfg.DATA.SEARCH.NUMBER,
        num_template = cfg.DATA.TEMPLATE.NUMBER,
        seq_format = cfg.DATA.SEQ_FORMAT,
        text_pooling = cfg.MODEL.LANGUAGE.POOLING
    )

    if cfg.TRAIN.TYPE in ["peft", "fft", "pret"]:
        load_pretrained(model, cfg.TRAIN.PRETRAINED_PATH)

    return model

def load_pretrained(model, pretrained_path, strict=False):

    seqtrackv1 = torch.load(pretrained_path, map_location="cpu")
    state_dict = seqtrackv1['net']
    pos_st = state_dict['encoder.body.pos_embed']
    pos_s = pos_st[:,:(pos_st.size(1) // 2)]
    pos_t = pos_st[:,(pos_st.size(1) // 2):]
    state_dict['encoder.body.pos_embed_search'] = pos_s
    state_dict['encoder.body.pos_embed_template'] = pos_t
    state_dict['encoder.body.patch_embed_interface.proj.weight'] = state_dict['encoder.body.patch_embed.proj.weight']
    state_dict['encoder.body.patch_embed_interface.proj.bias'] = state_dict['encoder.body.patch_embed.proj.bias']
    state_dict['decoder.embedding.prompt_embeddings.weight'] = model.state_dict()['decoder.embedding.prompt_embeddings.weight']
    state_dict['decoder.embedding.prompt_embeddings.weight'][:] = state_dict['decoder.embedding.word_embeddings.weight'][-1]
    del state_dict['encoder.body.pos_embed']
    model.load_state_dict(state_dict, strict=strict)
