from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from refined.data_types.base_types import Span 

from refined.doc_preprocessing.preprocessor import Preprocessor
from refined.model_components.description_encoder import DescriptionEncoder

class CrossAttention(nn.Module):
    def __init__(
        self,
        preprocessor: Preprocessor,
        mention_dim=768,
        output_dim=300,
        hidden_dim=1000,
        dropout=0.1,
        add_hidden=False,
    ):
        super().__init__()
        self.add_hidden = add_hidden
        self.dropout = nn.Dropout(dropout)
        self.hidden_layer = nn.Linear(mention_dim, hidden_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=output_dim, nhead=4)
        self.cross_attention = nn.TransformerDecoder(self.decoder_layer, num_layers=1)
        if add_hidden:
            self.mention_projection = nn.Linear(hidden_dim, output_dim)
        else:
            self.mention_projection = nn.Linear(mention_dim, output_dim)
        self.description_encoder = DescriptionEncoder(
            output_dim=output_dim, ff_chunk_size=4, preprocessor=preprocessor
        )

    def get_parameters_to_scale(self) -> List[nn.Parameter]:
        """
        Gets parameters that can be scaled during training. Usually top/last layers.
        :return: list of parameters
        """
        mention_projection_params = list(self.mention_projection.parameters())
        hidden_layer_projection_params = list(self.hidden_layer.parameters())
        description_encoder_projection_params = list(
            self.description_encoder.projection.parameters()
        )
        description_encoder_hidden_params = list(self.description_encoder.hidden_layer.parameters())
        description_encoder_params = list(self.description_encoder.transformer.parameters())
        cross_attention_params = list(self.cross_attention.parameters())
        return (
            mention_projection_params
            + description_encoder_projection_params
            + description_encoder_params
            + hidden_layer_projection_params
            + description_encoder_hidden_params
            + cross_attention_params
        )

    def get_parameters_not_to_scale(self) -> List[nn.Parameter]:
        """
        Gets parameters that can be scaled during training. Usually top/last layers.
        :return: list of parameters
        """
        return []

    def forward(
        self,
        mention_embeddings: Tensor,
        batches_num_ents: List,
        seq_embeddings: Tensor,
        spans: List[List[Span]],
        candidate_desc: Tensor,
        candidate_entity_targets: Optional[Tensor] = None,
        candidate_desc_emb: Optional[Tensor] = None,
    ):
        """
            mention_embeddings: (num_ents, 768) 
            batches_num_ents: (bs, num_ents)
            seq_embeddings: (bs, (seq_len, 768))
            spans: (all_num_ents)
            candidate_desc: (num_ents, num_cands, ?[desc_len])
            candidate_desc_emb: (num_ents, num_cands, hidden_dim)
        """
        if self.add_hidden:
            mention_embeddings = self.mention_projection(
                self.dropout(F.relu(self.hidden_layer(mention_embeddings)))
            )
            seq_embeddings = self.mention_projection(
                self.dropout(F.relu(self.hidden_layer(seq_embeddings)))
            )
        else:
            mention_embeddings = self.mention_projection(mention_embeddings)
            seq_embeddings = self.mention_projection(seq_embeddings)
            # (bs, seq_len, output_dim)

        if candidate_desc_emb is None:
            candidate_entity_embeddings, candidate_seq_embeddings = self.description_encoder(
                candidate_desc
            )  # (num_ents, num_cands, output_dim)
        else:
            candidate_entity_embeddings = candidate_desc_emb  # (num_ents, num_cands, output_dim)
            _, candidate_seq_embeddings = self.description_encoder(
                candidate_desc
            )
            # (num_ents, num_cands, desc_seq_len, output_dim)


        # =========================================================
        # compute sim scores, but only use it to get no_cand_index to update targets
        scores = (candidate_entity_embeddings @ mention_embeddings.unsqueeze(-1)).squeeze(
            -1
        )  # dot product
        # scores.shape = (num_ents, num_cands)
        assert self.description_encoder != None, "cross_attention : must input description_encoder module!"
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
        no_cand_score = torch.zeros((scores.size(0), 1), device=scores.device)
        scores = torch.cat([scores, no_cand_score], dim=-1)
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
        
        # candidate_desc : (num_ents, num_cands, 1)
        # seq_embeddings : (bs, seq_len, 768)
        last_start = 0
        loss = 0.0
        for batch_num in range(len(batches_num_ents)):
            # in-batch-sample: cands
            # input_seqs = seq_embeddings[batch_num, :, :].unsqueeze(0).repeat(candidate_desc.shape[0], 1, 1)
            # (batch_num_ents, max_cands, 32, output_dim)
            # print('candidate_seq_embeddings : ', candidate_seq_embeddings.shape)
            # example内包含一个输入seq，一个seq包含多个mention，每个mention对应自己的多个cands
            batch_candidate_seq_embeddings = candidate_seq_embeddings[last_start:last_start + batches_num_ents[batch_num], :, :, :]
            # (bs, seq_len, output_dim) ==> (1, seq_len, output_dim)
            input_seqs = seq_embeddings[batch_num, :, :].unsqueeze(0)
            rolled_input_seqs = input_seqs.permute(1, 0, 2).contiguous()
            # print('batch ', batch_num, ': ents = ', batches_num_ents[batch_num])
            # print('batch_candidate_seq_embeddings : ', batch_candidate_seq_embeddings.shape)
            # print('seq mention num : ', batches_num_ents[batch_num])
            for ent in range(batches_num_ents[batch_num]):
                ent_span = spans[batch_num][ent]
                if ent_span.ln == 0:
                    continue
                # print('ent is ', ent)
                # for every ent
                # print('seq_embeddings : ', seq_embeddings.shape)
                # print('input_seqs : ', rolled_input_seqs.shape)
                # print('batch_candidate_seq_embeddings : ', batch_candidate_seq_embeddings[ent, :, :, :].shape)
                # (seq_len, bs, dim)
                # rolled_candidate_seq_embeddings = batch_candidate_seq_embeddings[ent, :, :, :].permute(1, 0, 2).contiguous()
                if candidate_entity_targets is None or targets[last_start + ent] >= 30:
                    # only has 'gold entity' matters
                    continue
                # (batch_num_ents, cands, 32, output_dim) ==> (1, 32, output_dim) ==> (32, 1, output_dim)
                gold_entity_seq_embeddings = batch_candidate_seq_embeddings[ent, targets[last_start + ent], :, :].unsqueeze(0).permute(1, 0, 2).contiguous()
                # print('gold_entity_seq_embeddings : ', gold_entity_seq_embeddings.shape)
                out_seqs = self.cross_attention(rolled_input_seqs, gold_entity_seq_embeddings)
                # (seq_len, 1, output_dim) ==> (1, seq_len, output_dim)
                out_seqs = out_seqs.permute(1, 0, 2).contiguous()
                # (cands, mention_start:mention_end, hidden_size)
                # (1, mention_len, output_dim)
                enhance_mention_embs = out_seqs[:, ent_span.start:ent_span.start + ent_span.ln, :]
                # print('enhance_mention_embs : ', enhance_mention_embs.shape)
                if enhance_mention_embs.shape[1] == 0:
                    continue
                avgpool_mention_emb = enhance_mention_embs[0, :, :].mean(dim=0)
                # scores[last_start + ent, targets[last_start + ent]] = (batch_can)
                # print('enhance_mention_embs:', enhance_mention_embs.shape)
                # loss = -sim(m, target_e) + sim(m, negative_e)
                # print('cands num : ', batch_candidate_seq_embeddings.shape[1])
                # print('avgpool_mention_emb : ', avgpool_mention_emb)
                for cand in range(batch_candidate_seq_embeddings.shape[1]):
                    # avgpool_mention_emb = enhance_mention_embs[cand, :, :].mean(dim=0)
                    # avgpool_cand_emb = batch_candidate_seq_embeddings[last_start + ent, cand, :, :].mean(dim=0)
                    # print('avgpool_mention_emb : ', avgpool_mention_emb.shape)
                    # print('split batch_candidate_seq_embeddings : ', batch_candidate_seq_embeddings[last_start + ent:, cand, 0, :].shape)
                    # scores[last_start + ent, cand] = (batch_candidate_seq_embeddings[last_start + ent, cand, 0, :] @ avgpool_mention_emb)
                    # print('dot product : ', (avgpool_cand_emb @ avgpool_mention_emb))
                    # print('avgpool_cand_emb : ', avgpool_cand_emb)
                    scores[last_start + ent, cand] = (batch_candidate_seq_embeddings[last_start + ent, cand, 0, :] @ avgpool_mention_emb)
                del out_seqs
                del enhance_mention_embs
                torch.cuda.empty_cache()

            last_start += batches_num_ents[batch_num]
            del input_seqs
            del rolled_input_seqs
            del batch_candidate_seq_embeddings
            torch.cuda.empty_cache()

        if candidate_entity_targets is not None:
            loss = F.cross_entropy(scores, targets)
            # print('cross_attention loss : ', loss.item())
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