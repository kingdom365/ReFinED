from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from refined.data_types.base_types import Span 

from refined.doc_preprocessing.preprocessor import Preprocessor

class CrossAttention(nn.Module):
    def __init__(
        self,
        preprocessor: Preprocessor,
        mention_dim=768,
        output_dim=300,
        hidden_dim=300,
        dropout=0.1,
        add_hidden=False,
        description_encoder=None,
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
        self.description_encoder = description_encoder

    @classmethod
    def get_embeddings_(cls, seq, token_acc, entity_mask):
        pass

    def forward(
        self,
        mention_embeddings: Tensor,
        batch_split_mention_embeddings: List[Tensor],
        batch_mention_interval: List[List[Tuple]],
        seq_embeddings: List[Tensor],
        spans: List[Span],
        candidate_desc: Tensor,
        candidate_entity_targets: Optional[Tensor] = None,
        candidate_desc_emb: Optional[Tensor] = None,
    ):
        """
            mention_embeddings: (num_ents, 768) 
            batch_split_mention_embeddings: (bs, (batch_num_ents, 768))
            batch_mention_interval: (bs, batch_num_ents, 2[start, end])
            seq_embeddings: (bs, (seq_len, 768))
            spans: (all_num_ents)
            candidate_desc: (num_ents, num_cands, ?[desc_len])
            candidate_desc_emb: (num_ents, num_cands, hidden_dim)
        """
        # num_ents in batch
        batches = []
        for m in range(len(batch_split_mention_embeddings)):
            batches.append(m.shape[0])

        if self.add_hidden:
            mention_embeddings = self.mention_projection(
                self.dropout(F.relu(self.hidden_layer(mention_embeddings)))
            )
        else:
            mention_embeddings = self.mention_projection(mention_embeddings)

        if candidate_desc_emb is None:
            candidate_entity_embeddings, candidate_seq_embeddings = self.description_encoder(
                candidate_desc
            )  # (num_ents, num_cands, output_dim)
        else:
            candidate_entity_embeddings = candidate_desc_emb  # (num_ents, num_cands, output_dim)
            _, candidate_seq_embeddings = self.description_encoder(
                candidate_desc
            )


        # =========================================================
        # compute sim scores, but only use it to get no_cand_index to update targets
        scores = (candidate_entity_embeddings @ mention_embeddings.unsqueeze(-1)).squeeze(
            -1
        )  # dot product
        # scores.shape = (num_ents, num_cands)
        assert self.description_encoder == None, "cross_attention : must input description_encoder module!"
        mask_value = -100  # very large number may have been overflowing cause -inf loss
        if candidate_desc_emb is not None:
            # assumes candidate_desc_emb is initialised to all 0s
            multiplication_mask = (candidate_desc_emb[:, :, 0] != 0)
            addition_mask = (candidate_desc_emb[:, :, 0] == 0) * mask_value
        else:
            multiplication_mask = (
                candidate_desc[:, :, 0] != self.description_encoder.tokenizer.pad_token_id
            )
            addition_mask = (
                candidate_desc[:, :, 0] == self.description_encoder.tokenizer.pad_token_id
            ) * mask_value
        scores = (scores * multiplication_mask) + addition_mask
        # =========================================================

        if candidate_entity_targets is not None:
            targets = candidate_entity_targets.argmax(dim=1)
            no_cand_index = torch.zeros_like(targets)
            no_cand_index.fill_(
                scores.size(-1) - 1
            )
            # make NOTA (no_cand_score) the correct answer when the correct entity has no description
            targets = torch.where(
                scores[torch.arange(scores.size(0)), targets] != mask_value, targets, no_cand_index
            ) 


        no_cand_score = torch.zeros((scores.size(0), 1), device=scores.device)
        scores = torch.cat([scores, no_cand_score], dim=-1)

        # candidate_desc : (num_ents, num_cands, 1)
        # seq_embeddings : (bs, seq_len, 768)
        last_start = 1
        loss = 0.0
        for batch_num in range(len(seq_embeddings)):
            # in-batch-sample: cands
            # input_seqs = seq_embeddings[batch_num, :, :].unsqueeze(0).repeat(candidate_desc.shape[0], 1, 1)
            # (batch_num_ents, max_cands, 32, output_dim)
            batch_candidate_seq_embeddings = candidate_seq_embeddings[(last_start - 1):(last_start - 1) + batches[batch_num], :, :, :]
            for ent in range(batch_candidate_seq_embeddings.shape[0]):
                # for every ent
                # (1, seq_len, 768) ==> (cands, seq_len, 768)
                input_seqs = seq_embeddings[batch_num, :, :].unsqueeze(0).repeat(candidate_desc.shape[0], 1, 1)
                # (cands, seq_len, hidden_size)
                out_seqs = self.cross_attention(input_seqs, batch_candidate_seq_embeddings[ent, :, :, :])                
                ent_span = spans[(last_start - 1):(last_start - 1) + batches[batch_num]][ent]
                # (cands, mention_start:mention_end, hidden_size)
                enhance_mention_embs = out_seqs[:, ent_span.start:ent.span_start + ent_span.ln, :]
                # loss = -sim(m, target_e) + sim(m, negative_e)
                for cand in range(batch_candidate_seq_embeddings.shape[1]):
                    avgpool_mention_emb = enhance_mention_embs[cand, :, :].mean(dim=0)
                    scores[(last_start - 1) + ent, cand] = (candidate_entity_embeddings[(last_start - 1):(last_start - 1) + batch_num, cand, :] @ avgpool_mention_emb)

            if candidate_entity_targets is not None: 
                # train
                for ent in range(batch_candidate_seq_embeddings.shape[0]):
                    for cand in range(batch_candidate_seq_embeddings.shape[1]):
                        if cand != targets[ent].item():
                            loss += scores[ent, cand]
                        else:
                            loss += -scores[ent, cand]
        
        if candidate_entity_targets is not None:
            return loss, F.softmax(scores, dim=-1)
        else:
            return None, F.softmax(scores, dim=-1)

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
