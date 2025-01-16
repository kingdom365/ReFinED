from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from refined.doc_preprocessing.preprocessor import Preprocessor
from refined.model_components.description_encoder import DescriptionEncoder

import random

class EDLayer(nn.Module):
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
        self.dropout = nn.Dropout(dropout)
        self.hidden_layer = nn.Linear(mention_dim, hidden_dim)
        if add_hidden:
            self.mention_projection = nn.Linear(hidden_dim, output_dim)
        else:
            self.mention_projection = nn.Linear(mention_dim, output_dim)
        self.init_weights()
        self.description_encoder = DescriptionEncoder(
            output_dim=output_dim, ff_chunk_size=4, preprocessor=preprocessor
        )
        self.add_hidden = add_hidden

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
        return (
            mention_projection_params
            + description_encoder_projection_params
            + description_encoder_params
            + hidden_layer_projection_params
            + description_encoder_hidden_params
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
        candidate_desc: Tensor,
        candidate_pem_values: Tensor,
        candidate_classes: Tensor,
        candidate_entity_targets: Optional[Tensor] = None,
        candidate_desc_emb: Optional[Tensor] = None,
    ):
        # print('candidate_desc : ', candidate_desc.shape)
        # print('candidate_pem_values : ', candidate_pem_values.shape)
        # print('candidate_classes : ', candidate_classes.shape)
        if self.add_hidden:
            mention_embeddings = self.mention_projection(
                self.dropout(F.relu(self.hidden_layer(mention_embeddings)))
            )
        else:
            mention_embeddings = self.mention_projection(mention_embeddings)

        if candidate_desc_emb is None:
            candidate_entity_embeddings = self.description_encoder(
                candidate_desc
            )  # (num_ents, num_cands, output_dim)
        else:
            candidate_entity_embeddings = candidate_desc_emb  # (num_ents, num_cands, output_dim)

        scores = (candidate_entity_embeddings @ mention_embeddings.unsqueeze(-1)).squeeze(
            -1
        )  # dot product
        # scores.shape = (num_ents, num_cands)

        # mask out pad candidates (and candidates without descriptions) before calculating loss
        # multiplying by 0 kills the gradients, loss should also ignore it
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

        # add: fine-grained ranking, use cross-encoder
        ret_cands_scores = []
        ret_cands_targets = []
        ret_cands_desc = []
        ret_cands_desc_emb = []
        ret_cands_pem_values = []
        ret_cands_classes = []
        ret_cands_idx_mp = dict()   # map cand_tn idx ==> original idx, red_cands_idx_mp[ent_num] = [coarse_id1, coarse_id2, ...]
        if candidate_entity_targets is not None:
            # handles case when gold entity is masked should set target index to ignore it from loss
            targets = candidate_entity_targets.argmax(dim=1)
            no_cand_index = torch.zeros_like(targets)
            no_cand_index.fill_(
                scores.size(-1) - 1
            )
            # make NOTA (no_cand_score) the correct answer when the correct entity has no description
            targets = torch.where(
                scores[torch.arange(scores.size(0)), targets] != mask_value, targets, no_cand_index
            )
            
            for i in range(scores.size(0)):
                if targets[i].item() != scores.size(-1) - 1:
                    # gold not NIL
                    gold_entity_idx = targets[i]
                    # print('gold_entity_idx : ', gold_entity_idx)
                    target_idx = None
                    # select top-5 entity to fine-grained re-rank
                    fine_grained_set = torch.topk(scores[i, :-1], 5)
                    fine_grained_indices = fine_grained_set.indices
                    # print('fine_grained_indices : ', fine_grained_indices, '; fine_grained_values : ', fine_grained_set.values)
                    ret_cands_idx_mp[i] = []
                    for idx in range(fine_grained_indices.size(0)):
                        if fine_grained_indices[idx].item() == gold_entity_idx.item():
                            gold_entity_idx = None
                            target_idx = idx    # target_idx is target index in ret_cands_scores
                            # print('target_idx : ', target_idx)
                            break
                    if gold_entity_idx is not None:
                        # gold not include in top-5, hard positive sample
                        # print('gold not include in top-5')
                        ret_cands_targets.append(torch.tensor(4).to(scores.device).unsqueeze(0))
                        ret_cands_scores.append(torch.cat([fine_grained_set.values[:-1], scores[i, targets[i]].unsqueeze(0)], dim=-1).unsqueeze(0))
                        ret_cands_desc.append(torch.cat([candidate_desc[i, fine_grained_indices, :][:-1, :], candidate_desc[i, targets[i], :].unsqueeze(0)], dim=0).unsqueeze(0))
                        if candidate_desc_emb is not None:
                            ret_cands_desc_emb.append(torch.cat([candidate_desc_emb[i, fine_grained_indices, :][:-1, :], candidate_desc_emb[i, targets[i], :].unsqueeze(0)], dim=0).unsqueeze(0))
                        ret_cands_pem_values.append(torch.cat([candidate_pem_values[i, fine_grained_indices][:-1], candidate_pem_values[i, targets[i]].unsqueeze(0)], dim=0).unsqueeze(0))
                        ret_cands_classes.append(torch.cat([candidate_classes[i, fine_grained_indices, :][:-1, :], candidate_classes[i, targets[i], :].unsqueeze(0)], dim=0).unsqueeze(0))
                        for j in range(fine_grained_indices.size(0)):
                            if j != 4:
                                ret_cands_idx_mp[i].append(fine_grained_indices[j].item()) 
                            else:
                                ret_cands_idx_mp[i].append(targets[i])
                    else:
                        # gold include in top-5
                        # print('gold include in top-5')
                        ret_cands_targets.append(torch.tensor(target_idx).to(scores.device).unsqueeze(0))
                        ret_cands_scores.append(fine_grained_set.values.unsqueeze(0))
                        ret_cands_desc.append(candidate_desc[i, fine_grained_indices, :].unsqueeze(0))
                        if candidate_desc_emb is not None:
                            ret_cands_desc_emb.append(candidate_desc_emb[i, fine_grained_indices, :].unsqueeze(0))
                        ret_cands_pem_values.append(candidate_pem_values[i, fine_grained_indices].unsqueeze(0))
                        ret_cands_classes.append(candidate_classes[i, fine_grained_indices, :].unsqueeze(0))
                        for j in range(fine_grained_indices.size(0)):
                            ret_cands_idx_mp[i].append(fine_grained_indices[j].item())
                else:
                    # gold is NIL, hard positive sample
                    fine_grained_set = torch.topk(scores[i, :-1], 5)
                    fine_grained_indices = fine_grained_set.indices
                    gold_nil = None
                    trunc_idx = None
                    indices_set = set()
                    for j in range(fine_grained_indices.size(0)):
                        if fine_grained_indices[j].item() == scores.size(-1) - 1:
                            gold_nil = fine_grained_indices[j]
                            trunc_idx = j
                        indices_set.add(fine_grained_indices[j])
                    ret_cands_targets.append(torch.tensor(5).to(scores.device).unsqueeze(0))
                    if gold_nil is not None:
                        # NIL in cands, remove it, and random resample from 0-29 except indices
                        correct_indices = [num for num in range(30) if num not in indices_set]
                        random_sample_cand_idx = random.choice(correct_indices)
                        left_fine_indices = torch.cat([fine_grained_indices[:trunc_idx], fine_grained_indices[trunc_idx + 1:]], dim=-1)
                        left_fine_values = torch.cat([fine_grained_set.values[:trunc_idx], fine_grained_set.values[trunc_idx + 1:]], dim=-1)
                        ret_cands_scores.append(torch.cat([left_fine_values, scores[i, random_sample_cand_idx].unsqueeze(0)], dim=-1).unsqueeze(0))
                        ret_cands_desc.append(torch.cat([cand_desc[i, left_fine_indices, :], cand_desc[i, random_sample_cand_idx, :]], dim=0).unsqueeze(0))
                        if candidate_desc_emb is not None:
                            ret_cands_desc_emb.append(torch.cat([candidate_desc_emb[i, left_fine_indices, :], candidate_desc_emb[i, random_sample_cand_idx, :]], dim=0).unsqueeze(0))
                        ret_cands_pem_values.append(torch.cat([candidate_pem_values[i, left_fine_indices], candidate_pem_values[i, random_sample_cand_idx, :]], dim=0).unsqueeze(0))
                        ret_cands_classes.append(torch.cat([candidate_classes[i, left_fine_indices, :], candidate_classes[i, random_sample_cand_idx, :]], dim=0).unsqueeze(0))
                        ret_cands_idx_mp[i] = []
                        for j in range(fine_grained_indices.size(0)):
                            if j != 4:
                                ret_cands_idx_mp[i].append(fine_grained_indices[j].item())
                            else:
                                ret_cands_idx_mp[i].append(random_sample_cand_idx)
                    else:
                        # NIL not in cands
                        ret_cands_scores.append(fine_grained_indices.unsqueeze(0))
                        ret_cands_desc.append(candidate_desc[i, fine_grained_indices, :].unsqueeze(0))
                        if candidate_desc_emb is not None:
                            ret_cands_desc_emb.append(candidate_desc_emb[i, fine_grained_indices, :].unsqueeze(0))
                        ret_cands_pem_values.append(candidate_pem_values[i, fine_grained_indices].unsqueeze(0))
                        ret_cands_classes.append(candidate_classes[i, fine_grained_indices, :].unsqueeze(0))
                        ret_cands_idx_mp[i] = []
                        for j in range(fine_grained_indices.size(0)):
                            ret_cands_idx_mp[i].append(fine_grained_indices[j].item())
                # (5 --> 30, NIL)
                ret_cands_idx_mp[i].append(30)            
            # (num_ents)
            ret_cands_targets = torch.cat(ret_cands_targets, dim=0)
            # (num_ents, num_rerank_cands=5)
            ret_cands_scores = torch.cat(ret_cands_scores, dim=0)
            # ret_cands_scores = torch.cat([ret_cands_scores, torch.zeros(ret_cands_scores.size(0), 1).to(ret_cands_scores.device)], dim=-1)
            # (num_ents, num_rereank_cands, 32)
            ret_cands_desc = torch.cat(ret_cands_desc, dim=0)
            if len(ret_cands_desc_emb) > 0:
                # (num_ents, num_rerank_cands, 768)
                ret_cands_desc_emb = torch.cat(ret_cands_desc_emb, dim=0)
            else:
                ret_cands_desc_emb = None
            ret_cands_pem_values = torch.cat(ret_cands_pem_values, dim=0)
            ret_cands_classes = torch.cat(ret_cands_classes, dim=0)
            # loss = 0.0
            # for i in range(scores.size(0)):
            #     loss += -torch.log(torch.exp(scores[i, targets[i]]))
            #     if targets[i].item() > 0 and targets[i].item() < 30:
            #         loss += torch.log(torch.exp(torch.cat([scores[i, :targets[i]], scores[i, targets[i] + 1:]], dim=-1)).sum(dim=-1))
            #     elif targets[i].item() == 0:
            #         loss += torch.log(torch.exp(scores[i, 1:]).sum(dim=-1))
            #     else:
            #         loss += torch.log(torch.exp(scores[i, :-1]).sum(dim=-1))
            loss = F.cross_entropy(F.softmax(scores, dim=-1), targets)

            # Changed this loss Nov 17 2022 (have not trained model with this yet)
            # loss = F.cross_entropy(scores, targets, ignore_index=scores.size(-1) - 1)
            # if all targets are ignore_index value then loss is nan in torch 1.11
            # https://github.com/pytorch/pytorch/issues/75181
            # how should loss be calculated when:
            # Q1 - if gold candidate is not in candidates
            # Answer - should make NOTA (no_cand_score) the correct answer
            #          because none of the provided descriptions match the gold entity
            # Q2 - if gold candidate has no description
            # Answer - should make NOTA (no_cand_score) the correct answer
            #          because none of the provided descriptions match the gold entity
            # print('coarse loss : ', loss)
            return loss, F.softmax(scores, dim=-1), (ret_cands_targets, ret_cands_scores, ret_cands_desc, ret_cands_desc_emb, ret_cands_pem_values, ret_cands_classes, ret_cands_idx_mp)
        else:
            # (num_ents, num_cands) ==> (num_ents, 5)
            fine_grained_set = torch.topk(scores[:, :-1], 5)
            ret_cands_scores = fine_grained_set.values
            fine_grained_indices = fine_grained_set.indices
            # print('ret_cands_scores : ', ret_cands_scores.shape)
            for i in range(scores.size(0)):
                ent_cands_desc = candidate_desc[i, fine_grained_set.indices[i, :], :].unsqueeze(0)
                ret_cands_desc.append(ent_cands_desc)
                if candidate_desc_emb is not None:
                    ent_cands_desc_emb = candidate_desc_emb[i, fine_grained_indices[i, :], :].unsqueeze(0)
                    ret_cands_desc_emb.append(ent_cands_desc_emb)
                ent_cands_pem_values = candidate_pem_values[i, fine_grained_indices[i, :]].unsqueeze(0)
                ret_cands_pem_values.append(ent_cands_pem_values)
                ent_cands_classes = candidate_classes[i, fine_grained_indices[i, :], :].unsqueeze(0)
                ret_cands_classes.append(ent_cands_classes)
                ret_cands_idx_mp[i] = [fine_grained_indices[i, j].item() for j in range(5)]
                ret_cands_idx_mp[i].append(30)
            ret_cands_desc = torch.cat(ret_cands_desc, dim=0)
            ret_cands_pem_values = torch.cat(ret_cands_pem_values, dim=0)
            ret_cands_classes = torch.cat(ret_cands_classes, dim=0)
            if len(ret_cands_desc_emb) > 0:
                ret_cands_desc_emb = torch.cat(ret_cands_desc_emb, dim=0)
            else:
                ret_cands_desc_emb = None
            return None, F.softmax(scores, dim=-1), (None, ret_cands_scores, ret_cands_desc, ret_cands_desc_emb, ret_cands_pem_values, ret_cands_classes, ret_cands_idx_mp)  # output (num_ents, num_cands + 1)

    def init_weights(self):
        """Initialize weights for all member variables with type nn.Module"""
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
