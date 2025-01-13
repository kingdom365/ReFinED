from typing import List
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from refined.doc_preprocessing.preprocessor import Preprocessor
from refined.model_components.description_encoder import DescriptionEncoder
from refined.data_types.base_types import Span

class CrossEncoder(nn.Module):
    def __init__(
        self,
        preprocessor: Preprocessor,
        hidden_dim=1000,
        output_dim=300,
        mention_dim=768,
        add_hidden=False,
    ):
        super().__init__()
        self.hidden_layer = nn.Linear(mention_dim, hidden_dim)
        self.cross_encoder = DescriptionEncoder(
            output_dim=output_dim, ff_chunk_size=4, preprocessor=preprocessor
        )
        self.cross_encoder_pad_token_id = self.cross_encoder.tokenizer.pad_token_id
        # self.fusion_decoder_layer = nn.TransformerEncoderLayer(
        #     d_model=output_dim, nhead=4, dim_forward=128
        # )
        # self.fusion_decoder = nn.TransformerEncoder(
        #     self.fusion_decoder_layer, num_layers=2
        # )
        if add_hidden:
            self.down_sample_linear = nn.Linear(hidden_dim, output_dim)
        else:
            self.down_sample_linear = nn.Linear(mention_dim, output_dim)
        self.add_hidden = add_hidden
    
    def get_parameters_to_scale(self) -> List[nn.Parameter]:
        cross_encoder_params = list(self.cross_encoder.parameters())
        # fusion_decoder_params = list(self.fusion_decoder.parameters())
        down_sample_linear = list(self.down_sample_linear.parameters())
        return (
            cross_encoder_params +
            down_sample_linear
        )

    def get_parameters_not_to_scale(self) -> List[nn.Parameter]:
        return []

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

    def forward(self, 
        ctx_tokens: Tensor,
        mention_embeddings: Tensor,
        cand_desc: Tensor,
        candidate_desc_emb: Tensor,
        candidate_entity_targets: Tensor,
        batches_spans: List[List[Span]],
    ):
        # (bs, seq_len) ==> (num_ents, seq_len)
        fuse_token_tn = torch.tensor(self.cross_encoder_pad_token_id).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, cand_desc.shape[1], 1)
        batches_ctx_tokens = []
        last_batch_ents = 0
        for batch_num in range(len(batches_spans)):
            batch_ents = len(batches_spans[batch_num])
            span_list = batches_spans[batch_num]
            # batches_ctx_tokens.append(ctx_tokens[batch_num, :].unsqueeze(0).repeat(batch_ents, 1))
            for idx, span in enumerate(span_list):
                # (1, num_cands, 1)
                # ctx_tokens ==> (1, num_cands, seq_len)
                # fuse_token_tn (1, num_cands, 1)
                # cand_desc ==> (1, num_cands, desc_len=32)
                span_len = span.ln
                lctx_len = (128 - span_len) // 2
                rctx_len = 128 - span_len - lctx_len
                lctx_trunc_start = max(0, span.start - lctx_len)
                r_supply_len = 0 if lctx_len - (span.start - lctx_trunc_start) <= 0 else lctx_len - (span.start - lctx_trunc_start)
                rctx_trunc_end = min(span.start + span.ln + rctx_len, ctx_tokens[batch_num, :].size(-1))
                l_supply_len = 0 if rctx_len - (ctx_tokens[batch_num, :].size(-1) - (span.start + span_len)) <= 0 else rctx_len - (ctx_tokens[batch_num, :].size(-1) - (span.start + span_len))

                m_c_fusion_seq = torch.cat([ctx_tokens[batch_num, lctx_trunc_start - l_supply_len:rctx_trunc_end + r_supply_len].unsqueeze(0).repeat(cand_desc.shape[1], 1).unsqueeze(0), 
                                            fuse_token_tn, cand_desc[idx + last_batch_ents, :, :].unsqueeze(0)], dim=-1)
                # print('m_c_fusion_seq : ', m_c_fusion_seq.shape)
                batches_ctx_tokens.append(m_c_fusion_seq)
            last_batch_ents = batch_ents

        ctx_max_len = 161
        padded_batches_ctx_tokens = []
        for tn in batches_ctx_tokens:
            if tn.size(-1) < ctx_max_len:
                padded_tn = torch.cat([tn, fuse_token_tn.repeat(1, 1, (ctx_max_len - tn.size(-1)))], dim=-1)
            else:
                padded_tn = tn
            padded_batches_ctx_tokens.append(padded_tn)
        padded_batches_ctx_tokens = torch.cat(padded_batches_ctx_tokens, dim=0)

        if self.add_hidden:
            mention_embeddings = self.down_sample_linear(F.relu(self.hidden_layer(mention_embeddings)))
        else:
            mention_embeddings = self.down_sample_linear(mention_embeddings)

        # 1. input into cross-encoder, (num_ents, num_cands, cat_len) ==> (num_ents, num_cands, output_dim)
        print('padded_batches_ctx_tokens : ', padded_batches_ctx_tokens.shape)
        print('cand_desc : ', cand_desc.shape)
        ctx_cand_interactions = self.cross_encoder(padded_batches_ctx_tokens)
        # 2. use interactions tns instead of desc tns, compute sim scores 
        scores = (ctx_cand_interactions @ mention_embeddings.unsqueeze(-1)).squeeze(-1)        
        mask_value = -100  # very large number may have been overflowing cause -inf loss
        if candidate_desc_emb is not None:
            # assumes candidate_desc_emb is initialised to all 0s
            multiplication_mask = (candidate_desc_emb[:, :, 0] != 0)
            addition_mask = (candidate_desc_emb[:, :, 0] == 0) * mask_value
        else:
            multiplication_mask = (
                cand_desc[:, :, 0] != self.cross_encoder.tokenizer.pad_token_id
            )
            addition_mask = (
                cand_desc[:, :, 0] == self.cross_encoder.tokenizer.pad_token_id
            ) * mask_value
        scores = (scores * multiplication_mask) + addition_mask

        no_cand_score = torch.zeros((scores.size(0), 1), device=scores.device)
        scores = torch.cat([scores, no_cand_score], dim=-1)
        
        if candidate_entity_targets is not None:
            # (num_ents)
            targets = candidate_entity_targets.argmax(-1)
            no_cand_index = torch.zeros_like(targets)
            # fill with 31 - 1 ==> 30, normal targets are 0-29
            no_cand_index.fill_(
                scores.size(-1) - 1
            )
            targets = torch.where(
                scores[torch.arange(scores.size(0)), targets] != mask_value, targets, no_cand_index
            )            
            loss = F.cross_entropy(scores, targets)
            return loss, F.softmax(scores, dim=-1)
        else:
            return None, F.softmax(scores, dim=-1)