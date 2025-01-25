from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from refined.doc_preprocessing.preprocessor import Preprocessor
from refined.model_components.description_encoder import DescriptionEncoder
from refined.data_types.base_types import Span

from info_nce import InfoNCE, info_nce
from transformers import AutoModel, AutoTokenizer

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
        # self.combine_projection = nn.Linear(2 * output_dim, output_dim)
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

    def calc_info_nce_loss(self, targets, mention_embs, candidate_entity_embs, device, temperature=0.07):
        query = mention_embs    # [mention_num, emb_size]
        loss_func = InfoNCE(temperature=0.07, negative_mode='unpaired')
        info_nce_loss = 0.0
        # num_mentions = query.shape[0]
        num_mentions = targets.shape[0]
        num_cands = candidate_entity_embs.shape[1]
        for i in range(num_mentions):
            if targets[i].item() >= 30:
                # no positives
                # negatives = candidate_entity_embs[i,:,:]
                # for j in range(negatives.shape[0]):
                #     # print("query : ", query[i, :].shape)
                #     # print("negatives : ", negatives[j,:].shape)
                #     euclid_loss += torch.dist(query[i, :], negatives[j,:])
                continue
            else:
                positives = candidate_entity_embs[i, targets[i].item() - 1, :].unsqueeze(0)
                negatives = candidate_entity_embs[i, :, :].squeeze(0)
                neg_mask = torch.ones_like(negatives, dtype=torch.bool)
                neg_mask[targets[i].item() - 1, :] = False 
                negatives = negatives[neg_mask].view(num_cands - 1, -1)
                # print('query shape : ', query.shape)
                # print('positives shape : ', positives.shape)
                # print('negatives shape : ', negatives.shape)
                mention_loss = loss_func(query[i, :].unsqueeze(0), positives, negatives)
                info_nce_loss += mention_loss
           
        return info_nce_loss 

    def forward(
        self,
        mention_embeddings: Tensor,
        batch_tokens: Tensor,
        batches_spans: List[List[Span]],
        batch_split_mention_embeddings: List[Tensor],
        contextualised_embedding: Tensor,
        candidate_desc: Tensor,
        candidate_entity_targets: Optional[Tensor] = None,
        candidate_desc_emb: Optional[Tensor] = None,
    ):
        # print('candidate_entity_target', candidate_entity_targets.shape)
        # print(candidate_entity_targets)
        # (mention, definition)
        # print('ctx_emb : ', contextualised_embedding.shape)
        # (num_ents, 768)
        ctx_emb = contextualised_embedding
        # combine ctx_emb with mention_embs
        # 1. for every mention embedding, mul with len of mention
        i = 0
        new_mention_embeddings = []
        for sample_num in range(len(batches_spans)):
            for span in batches_spans[sample_num]:
                new_mention_embedding = mention_embeddings[i, :] * span.ln
                new_mention_embedding = (ctx_emb[i, :] + new_mention_embedding) / (span.ln + 1)
                new_mention_embeddings.append(new_mention_embedding.unsqueeze(0))
                i += 1
        # (num_ents, 768)
        mention_embeddings = torch.cat(new_mention_embeddings, dim=0)
        ret_mention_embeddings = mention_embeddings
        
        if self.add_hidden:
            # mention:ctx
            # (num_ents, 768)
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
            # When ED, freeze desc decoder, only use candidate_entity_embs
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
            loss = F.cross_entropy(scores, targets)

            # print('mention_embs : ', mention_embeddings.shape)
            # ctx_embs = ctx_emb.repeat(mention_embeddings.shape[0], 1)
            # info_nce_loss = self.calc_info_nce_loss(targets, ctx_embs, candidate_entity_embeddings, scores.device)

            '''
            # for ctx
            # ctx_embs.shape (bs, 300)
            info_nce_loss = 0.0
            # expand ctx_embs for every batch
            batch_split_ctx_embs = []
            # print('batch_split_mention : ', len(batch_split_mention_embeddings))
            for batch_num in range(ctx_emb.shape[0]):
                # (num_ents, 300) num_ents = sum(all_batch_num_ents)
                # print('batch_split_mention : ', batch_split_mention_embeddings[batch_num].shape)
                batch_split_ctx_embs.append(
                    ctx_emb[batch_num, :].unsqueeze(0).repeat(batch_split_mention_embeddings[batch_num].shape[0], 1)
                )
            # (num_ents_in_all_batches, 300)
            final_ctx_embs = torch.cat(batch_split_ctx_embs, dim=0)
            ctx_with_mentions = (final_ctx_embs + mention_embeddings) / 2
            # print('final_ctx_embs : ', final_ctx_embs.shape)
            # print('mention_embeddings : ', mention_embeddings.shape)
            info_nce_loss = self.calc_info_nce_loss(targets, ctx_with_mentions, candidate_entity_embeddings, scores.device)
            '''

            info_nce_loss = self.calc_info_nce_loss(targets, mention_embeddings, candidate_entity_embeddings, scores.device)

            # print('ctx_embs : ', ctx_embs.shape)
            # info_nce_loss = self.calc_info_nce_loss(targets, mention_embeddings, candidate_entity_embeddings, scores.device)
            beta = 0.01
            total_loss = (1.0 - beta) * loss + beta * info_nce_loss
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
            return total_loss, F.softmax(scores, dim=-1), ret_mention_embeddings
        else:
            return None, F.softmax(scores, dim=-1), ret_mention_embeddings  # output (num_ents, num_cands + 1)

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