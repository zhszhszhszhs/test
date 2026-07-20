import torch
import pickle
import numpy as np
import torch_sparse
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F

from config.configurator import configs
from models.general_cf.lightgcn import BaseModel
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class DCCF_int_nogaa(BaseModel):
    def __init__(self, data_handler):
        super(DCCF_int_nogaa, self).__init__(data_handler)
        rows = data_handler.trn_mat.tocoo().row
        cols = data_handler.trn_mat.tocoo().col
        new_rows = np.concatenate([rows, cols + self.user_num], axis=0)
        new_cols = np.concatenate([cols + self.user_num, rows], axis=0)
        plain_adj = sp.coo_matrix((np.ones(len(new_rows)), (new_rows, new_cols)), shape=[self.user_num + self.item_num, self.user_num + self.item_num]).tocsr().tocoo()
        self.all_h_list = list(plain_adj.row)
        self.all_t_list = list(plain_adj.col)
        self.A_in_shape = plain_adj.shape
        self.A_indices = torch.tensor([self.all_h_list, self.all_t_list], dtype=torch.long).cuda()
        self.D_indices = torch.tensor([list(range(self.user_num + self.item_num)), list(range(self.user_num + self.item_num))], dtype=torch.long).cuda()
        self.all_h_list = torch.LongTensor(self.all_h_list).cuda()
        self.all_t_list = torch.LongTensor(self.all_t_list).cuda()
        self.G_indices, self.G_values = self._cal_sparse_adj()

        # hyper-parameter
        self.collaborative_intent_num = configs['model']['collaborative_intent_num']
        self.collaborative_layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']
        self.cl_weight = self.hyper_config['cl_weight']
        self.cl_temperature = self.hyper_config['cl_temperature']
        self.kd_weight = self.hyper_config['kd_weight']
        self.kd_temperature = self.hyper_config['kd_temperature']
        self.kd_int_weight = self.hyper_config['kd_int_weight']
        self.kd_int_temperature = self.hyper_config['kd_int_temperature']
        self.semantic_lightgcn_layer_num = configs['model']['semantic_lightgcn_layer_num']
        self.semantic_topm = configs['model']['semantic_topm']
        self.semantic_message_weight = configs['model']['semantic_message_weight']
        self.semantic_bpr_weight = configs['model']['semantic_bpr_weight']
        self.semantic_score_weight = configs['model']['semantic_score_weight']

        # model parameters
        self.user_embeds = nn.Embedding(self.user_num, self.embedding_size)
        self.item_embeds = nn.Embedding(self.item_num, self.embedding_size)
        self.user_collaborative_intents = torch.nn.Parameter(init(torch.empty(self.embedding_size, self.collaborative_intent_num)), requires_grad=True)
        self.item_collaborative_intents = torch.nn.Parameter(init(torch.empty(self.embedding_size, self.collaborative_intent_num)), requires_grad=True)

        # train/test
        self.is_training = True
        self.final_embeds = False
        self.final_semantic_embeds = None

        # intent information
        self.user_semantic_embeds = torch.tensor(configs['user_semantic_embeds']).float().cuda()
        self.item_semantic_embeds = torch.tensor(configs['item_semantic_embeds']).float().cuda()
        self.semantic_intent_prototypes = torch.tensor(configs['semantic_intent_prototypes']).float().cuda()
        self.semantic_adapter = nn.Linear(self.user_semantic_embeds.shape[1], self.embedding_size, bias=False)
        self.intent_kd_mlp = nn.Sequential(
            nn.Linear(self.user_semantic_embeds.shape[1], (self.user_semantic_embeds.shape[1] + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.user_semantic_embeds.shape[1] + self.embedding_size) // 2, self.embedding_size)
        )
        self.representation_kd_mlp = nn.Sequential(
            nn.Linear(self.user_semantic_embeds.shape[1], (self.user_semantic_embeds.shape[1] + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.user_semantic_embeds.shape[1] + self.embedding_size) // 2, self.embedding_size)
        )
        self._init_weight()

    def _init_weight(self):
        for m in self.intent_kd_mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)
        for m in self.representation_kd_mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)
        init(self.semantic_adapter.weight)
        init(self.user_embeds.weight)
        init(self.item_embeds.weight)

    def _cal_sparse_adj(self):
        A_values = torch.ones(size=(len(self.all_h_list), 1)).view(-1).cuda()
        A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list, value=A_values, sparse_sizes=self.A_in_shape).cuda()
        D_values = A_tensor.sum(dim=1).pow(-0.5)
        G_indices, G_values = torch_sparse.spspmm(self.D_indices, D_values, self.A_indices, A_values, self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])
        G_indices, G_values = torch_sparse.spspmm(G_indices, G_values, self.D_indices, D_values, self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])
        return G_indices, G_values

    def _adaptive_mask(self, head_embeddings, tail_embeddings):
        head_embeddings = torch.nn.functional.normalize(head_embeddings)
        tail_embeddings = torch.nn.functional.normalize(tail_embeddings)
        edge_alpha = (torch.sum(head_embeddings * tail_embeddings, dim=1).view(-1) + 1) / 2
        A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list, value=edge_alpha, sparse_sizes=self.A_in_shape).cuda()
        D_scores_inv = A_tensor.sum(dim=1).pow(-1).nan_to_num(0, 0, 0).view(-1)
        G_indices = torch.stack([self.all_h_list, self.all_t_list], dim=0)
        G_values = D_scores_inv[self.all_h_list] * edge_alpha
        return G_indices, G_values

    def forward(self):
        if not self.is_training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:], None, None, None

        collaborative_states = [torch.concat([self.user_embeds.weight, self.item_embeds.weight], dim=0)]
        graph_layer_embeds, global_intent_layer_embeds, local_intent_layer_embeds = [], [], []

        for i in range(0, self.collaborative_layer_num):
            # Graph-based Message Passing
            graph_embeds = torch_sparse.spmm(self.G_indices, self.G_values, self.A_in_shape[0], self.A_in_shape[1], collaborative_states[i])

            # Intent-aware Information Aggregation
            user_embeds, item_embeds = torch.split(collaborative_states[i], [self.user_num, self.item_num], 0)
            user_global_intents = torch.softmax(user_embeds @ self.user_collaborative_intents, dim=1) @ self.user_collaborative_intents.T
            item_global_intents = torch.softmax(item_embeds @ self.item_collaborative_intents, dim=1) @ self.item_collaborative_intents.T
            global_intent_embeds = torch.concat([user_global_intents, item_global_intents], dim=0)

            # Adaptive Augmentation
            head_intent_embeds = torch.index_select(global_intent_embeds, 0, self.all_h_list)
            tail_intent_embeds = torch.index_select(global_intent_embeds, 0, self.all_t_list)
            intent_graph_indices, intent_graph_values = self._adaptive_mask(head_intent_embeds, tail_intent_embeds)
            local_intent_embeds = torch_sparse.spmm(intent_graph_indices, intent_graph_values, self.A_in_shape[0], self.A_in_shape[1], collaborative_states[i])

            # Aggregation
            graph_layer_embeds.append(graph_embeds)
            global_intent_layer_embeds.append(global_intent_embeds)
            local_intent_layer_embeds.append(local_intent_embeds)
            collaborative_states.append(graph_embeds + global_intent_embeds + local_intent_embeds + collaborative_states[i])

        collaborative_embeds = torch.stack(collaborative_states, dim=1)
        collaborative_embeds = torch.sum(collaborative_embeds, dim=1, keepdim=False)
        user_embeds, item_embeds = torch.split(collaborative_embeds, [self.user_num, self.item_num], 0)
        self.final_embeds = collaborative_embeds
        return user_embeds, item_embeds, graph_layer_embeds, global_intent_layer_embeds, local_intent_layer_embeds

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds

    def _semantic_attention(self, node_embeds, semantic_intent_embeds):
        scores = node_embeds @ semantic_intent_embeds.T / np.sqrt(node_embeds.shape[1])
        if self.training:
            noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-10) + 1e-10)
            scores = scores + noise
        topm_values, topm_indices = torch.topk(scores, self.semantic_topm, dim=1)
        values = torch.softmax(topm_values, dim=1).reshape(-1)
        rows = torch.arange(scores.shape[0], device=scores.device).unsqueeze(1)
        rows = rows.expand(-1, self.semantic_topm).reshape(-1)
        return torch.sparse_coo_tensor(
            torch.stack([rows, topm_indices.reshape(-1)]), values, scores.shape
        ).coalesce()

    def _semantic_forward(self):
        if not self.is_training and self.final_semantic_embeds is not None:
            return torch.split(self.final_semantic_embeds, [self.user_num, self.item_num], 0)

        semantic_users = self.semantic_adapter(self.user_semantic_embeds.detach())
        semantic_items = self.semantic_adapter(self.item_semantic_embeds.detach())
        semantic_states = torch.concat([semantic_users, semantic_items], dim=0)
        semantic_layer_states = [semantic_states]
        for _ in range(self.semantic_lightgcn_layer_num):
            semantic_states = torch_sparse.spmm(
                self.G_indices, self.G_values,
                self.A_in_shape[0], self.A_in_shape[1], semantic_states
            )
            semantic_layer_states.append(semantic_states)
        semantic_users, semantic_items = torch.split(
            torch.stack(semantic_layer_states, dim=1).mean(dim=1),
            [self.user_num, self.item_num], 0
        )

        projected_semantic_intents = self.semantic_adapter(self.semantic_intent_prototypes.detach())
        user_intent_attention = self._semantic_attention(semantic_users, projected_semantic_intents)
        item_intent_attention = self._semantic_attention(semantic_items, projected_semantic_intents)
        user_intent_degree = torch.sparse.sum(user_intent_attention, dim=0).to_dense().clamp_min(1e-6).unsqueeze(1)
        item_intent_degree = torch.sparse.sum(item_intent_attention, dim=0).to_dense().clamp_min(1e-6).unsqueeze(1)
        intents_from_users = torch.sparse.mm(user_intent_attention.t(), semantic_users) / user_intent_degree
        intents_from_items = torch.sparse.mm(item_intent_attention.t(), semantic_items) / item_intent_degree
        dynamic_semantic_intents = (intents_from_users + intents_from_items) / 2
        semantic_users = semantic_users + self.semantic_message_weight * 0.5 * torch.sparse.mm(user_intent_attention, dynamic_semantic_intents)
        semantic_items = semantic_items + self.semantic_message_weight * 0.5 * torch.sparse.mm(item_intent_attention, dynamic_semantic_intents)
        semantic_users = F.normalize(semantic_users, p=2, dim=1)
        semantic_items = F.normalize(semantic_items, p=2, dim=1)

        if not self.is_training:
            self.final_semantic_embeds = torch.concat([semantic_users, semantic_items], dim=0)
        return semantic_users, semantic_items

    def _cal_cl_loss(self, users, items, graph_layer_embeds, global_intent_layer_embeds, local_intent_layer_embeds):
        users = torch.unique(users)
        items = torch.unique(items)
        cl_loss = 0.0
        for i in range(len(graph_layer_embeds)):
            user_graph_embeds, item_graph_embeds = torch.split(graph_layer_embeds[i], [self.user_num, self.item_num], 0)
            user_global_intents, item_global_intents = torch.split(global_intent_layer_embeds[i], [self.user_num, self.item_num], 0)
            user_local_intents, item_local_intents = torch.split(local_intent_layer_embeds[i], [self.user_num, self.item_num], 0)

            user_graph_embeds = user_graph_embeds[users]
            user_global_intents = user_global_intents[users]
            user_local_intents = user_local_intents[users]

            item_graph_embeds = item_graph_embeds[items]
            item_global_intents = item_global_intents[items]
            item_local_intents = item_local_intents[items]

            cl_loss += cal_infonce_loss(user_graph_embeds, user_global_intents, user_global_intents, self.cl_temperature) / user_graph_embeds.shape[0]
            cl_loss += cal_infonce_loss(user_graph_embeds, user_local_intents, user_local_intents, self.cl_temperature) / user_graph_embeds.shape[0]
            cl_loss += cal_infonce_loss(item_graph_embeds, item_global_intents, item_global_intents, self.cl_temperature) / user_graph_embeds.shape[0]
            cl_loss += cal_infonce_loss(item_graph_embeds, item_local_intents, item_local_intents, self.cl_temperature) / user_graph_embeds.shape[0]
        return cl_loss

    def cal_loss(self, batch_data):
        self.is_training = True
        self.final_semantic_embeds = None
        collaborative_user_embeds, collaborative_item_embeds, graph_layer_embeds, global_intent_layer_embeds, local_intent_layer_embeds = self.forward()
        semantic_user_embeds, semantic_item_embeds = self._semantic_forward()
        ancs, poss, negs = batch_data
        anc_collaborative_embeds = collaborative_user_embeds[ancs]
        pos_collaborative_embeds = collaborative_item_embeds[poss]
        neg_collaborative_embeds = collaborative_item_embeds[negs]
        bpr_loss = cal_bpr_loss(anc_collaborative_embeds, pos_collaborative_embeds, neg_collaborative_embeds) / anc_collaborative_embeds.shape[0]
        anc_semantic_embeds = semantic_user_embeds[ancs]
        pos_semantic_embeds = semantic_item_embeds[poss]
        neg_semantic_embeds = semantic_item_embeds[negs]
        semantic_bpr_loss = cal_bpr_loss(anc_semantic_embeds, pos_semantic_embeds, neg_semantic_embeds) / anc_semantic_embeds.shape[0]
        semantic_bpr_loss *= self.semantic_bpr_weight
        reg_loss = self.reg_weight * reg_params(self)
        cl_loss = self.cl_weight * self._cal_cl_loss(ancs, poss, graph_layer_embeds, global_intent_layer_embeds, local_intent_layer_embeds)

        user_kd_semantics = self.representation_kd_mlp(self.user_semantic_embeds)
        item_kd_semantics = self.representation_kd_mlp(self.item_semantic_embeds)
        anc_kd_semantics, pos_kd_semantics, neg_kd_semantics = self._pick_embeds(user_kd_semantics, item_kd_semantics, batch_data)
        kd_loss = cal_infonce_loss(anc_collaborative_embeds, anc_kd_semantics, user_kd_semantics, self.kd_temperature) + \
                        cal_infonce_loss(pos_collaborative_embeds, pos_kd_semantics, pos_kd_semantics, self.kd_temperature) + \
                        cal_infonce_loss(neg_collaborative_embeds, neg_kd_semantics, neg_kd_semantics, self.kd_temperature)
        kd_loss /= anc_collaborative_embeds.shape[0]
        kd_loss *= self.kd_weight

        mean_local_intent_embeds = torch.mean(torch.stack(local_intent_layer_embeds), dim=0)
        user_local_intents, item_local_intents = torch.split(mean_local_intent_embeds, [self.user_num, self.item_num], 0)
        anc_local_intents, pos_local_intents, neg_local_intents = self._pick_embeds(user_local_intents, item_local_intents, batch_data)
        user_intent_kd_semantics = self.intent_kd_mlp(self.user_semantic_embeds)
        item_intent_kd_semantics = self.intent_kd_mlp(self.item_semantic_embeds)
        anc_intent_kd_semantics, pos_intent_kd_semantics, neg_intent_kd_semantics = self._pick_embeds(user_intent_kd_semantics, item_intent_kd_semantics, batch_data)
        kd_int_loss = cal_infonce_loss(anc_local_intents, anc_intent_kd_semantics, user_intent_kd_semantics, self.kd_int_temperature) + \
                      cal_infonce_loss(pos_local_intents, pos_intent_kd_semantics, pos_intent_kd_semantics, self.kd_int_temperature) + \
                      cal_infonce_loss(neg_local_intents, neg_intent_kd_semantics, neg_intent_kd_semantics, self.kd_int_temperature)
        kd_int_loss /= anc_local_intents.shape[0]
        kd_int_loss *= self.kd_int_weight

        loss = bpr_loss + semantic_bpr_loss + reg_loss + cl_loss + kd_loss + kd_int_loss
        losses = {'bpr_loss': bpr_loss, 'semantic_bpr_loss': semantic_bpr_loss, 'reg_loss': reg_loss, 'cl_loss': cl_loss, 'kd_loss': kd_loss, 'kd_int_loss': kd_int_loss}
        return loss, losses

    def full_predict(self, batch_data):
        collaborative_user_embeds, collaborative_item_embeds, _, _, _ = self.forward()
        self.is_training = False
        semantic_user_embeds, semantic_item_embeds = self._semantic_forward()
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        batch_collaborative_users = collaborative_user_embeds[pck_users]
        collaborative_scores = batch_collaborative_users @ collaborative_item_embeds.T
        semantic_scores = semantic_user_embeds[pck_users] @ semantic_item_embeds.T
        full_preds = collaborative_scores + self.semantic_score_weight * semantic_scores
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
