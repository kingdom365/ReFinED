from typing import List, Tuple, Optional
import torch
from torch import nn, Tensor
from refined.data_types.base_types import Span 

from refined.doc_preprocessing.preprocessor import Preprocessor

class CrossAttention(nn.Module):
    def __init__(
        self,
        preprocessor: Preprocessor,
        mention_dim=768,
        output_dim=300,
        hidden_dim=512,
        dropout=0.1,
        add_hidden=False,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.hidden_layer = nn.Linear(mention_dim, hidden_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8)
        self.cross_attention = nn.TransformerDecoder(self.decoder_layer, num_layers=6)
        if add_hidden:
            self.mention_projection = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.mention_projection = nn.Linear(mention_dim, hidden_dim)

    @classmethod
    def get_embeddings_(cls, seq, token_acc, entity_mask):
        pass

    def forward(
        self,
        mention_embeddings: Tensor,
        batch_split_mention_embeddings: List[Tensor],
        batch_mention_interval: List[List[Tuple]],
        seq_embeddings: List[Tensor],
        spans: List[List[Span]],
        candidate_desc: Tensor,
        candidate_entity_targets: Optional[Tensor] = None,
        candidate_desc_emb: Optional[Tensor] = None,
    ):
        """
            mention_embeddings: (num_ents, 768) 
            batch_split_mention_embeddings: (bs, batch_num_ents, 768)
            batch_mention_interval: (bs, batch_num_ents, 2[start, end])
            seq_embeddings: (bs, (seq_len, 768))
            spans: (bs, (num_ents))
            candidate_desc: (num_ents, num_cands, ?[desc_len])
            candidate_desc_emb: (num_ents, num_cands, hidden_dim)
        """
        targets = candidate_entity_targets.argmax(dim=1)
        no_cand_index = torch.zeros_like(targets)
        no_cand_index.fill_(
            scores.size(-1) - 1
        )
        # make NOTA (no_cand_score) the correct answer when the correct entity has no description
        targets = torch.where(
            scores[torch.arange(scores.size(0)), targets] != mask_value, targets, no_cand_index
        )
        for batch_num in range(len(seq_embeddings)):
            target_entity = candidate_entity_targets[batch_num]
            # (1, seq_len, 768) ==> (cands, seq_len, 768)
            input_seqs = seq_embeddings[batch_num].repeat(candidate_desc.shape[0], 1, 1)
            # (1, num_ents, 512)
            out_seqs = self.cross_attention(candidate_desc[batch_num, :, :].unsqueeze(0) , input_seqs)
            enhance_mention_embs = []
            for span in spans[batch_num]:
                out_seqs[:, span.start : span.start + span.ln, :]
                pass
            if candidate_entity_targets is not None:
                # train
                pass
            else:
                # inference
                pass   
            pass
        
        
        pass
                

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
