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

        # prepare adjacency matrix for DCCF
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
        self.intent_num = configs['model']['intent_num']
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']
        self.cl_weight = self.hyper_config['cl_weight']
        self.cl_temperature = self.hyper_config['cl_temperature']
        self.kd_weight = self.hyper_config['kd_weight']
        self.kd_temperature = self.hyper_config['kd_temperature']
        self.kd_int_weight = self.hyper_config['kd_int_weight']
        self.kd_int_temperature = self.hyper_config['kd_int_temperature']
        self.llm_lightgcn_layers = configs['model']['llm_lightgcn_layers']
        self.llm_topk = configs['model']['llm_topk']
        self.llm_intent_weight = configs['model']['llm_intent_weight']
        self.llm_bpr_weight = configs['model']['llm_bpr_weight']
        self.llm_score_weight = configs['model']['llm_score_weight']

        # model parameters
        self.user_embeds = nn.Embedding(self.user_num, self.embedding_size)
        self.item_embeds = nn.Embedding(self.item_num, self.embedding_size)
        self.user_intent = torch.nn.Parameter(init(torch.empty(self.embedding_size, self.intent_num)), requires_grad=True)
        self.item_intent = torch.nn.Parameter(init(torch.empty(self.embedding_size, self.intent_num)), requires_grad=True)

        # train/test
        self.is_training = True
        self.final_embeds = False
        self.final_llm_embeds = None

        # intent information
        self.usrint_embeds = torch.tensor(configs['usrint_embeds']).float().cuda()
        self.itmint_embeds = torch.tensor(configs['itmint_embeds']).float().cuda()
        self.llm_intent_prototypes = torch.tensor(configs['llm_intent_prototypes']).float().cuda()
        self.llm_adapter = nn.Linear(self.usrint_embeds.shape[1], self.embedding_size, bias=False)
        self.int_mlp = nn.Sequential(
            nn.Linear(self.usrint_embeds.shape[1], (self.usrint_embeds.shape[1] + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.usrint_embeds.shape[1] + self.embedding_size) // 2, self.embedding_size)
        )
        self.kd_mlp = nn.Sequential(
            nn.Linear(self.usrint_embeds.shape[1], (self.usrint_embeds.shape[1] + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.usrint_embeds.shape[1] + self.embedding_size) // 2, self.embedding_size)
        )
        # self.int_mlp = MoE(
        #     input_size=self.usrint_embeds.shape[1],
        #     output_size=self.embedding_size,
        #     num_experts=4,  # Number of experts
        #     top_k=2  # Number of top experts to use
        # )
        self._init_weight()

    def _init_weight(self):
        for m in self.int_mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)
        for m in self.kd_mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)
        init(self.llm_adapter.weight)
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

        all_embeds = [torch.concat([self.user_embeds.weight, self.item_embeds.weight], dim=0)]
        gnn_embeds, int_embeds, iaa_embeds = [], [], []

        for i in range(0, self.layer_num):
            # Graph-based Message Passing
            gnn_layer_embeds = torch_sparse.spmm(self.G_indices, self.G_values, self.A_in_shape[0], self.A_in_shape[1], all_embeds[i])

            # Intent-aware Information Aggregation
            u_embeds, i_embeds = torch.split(all_embeds[i], [self.user_num, self.item_num], 0)
            u_int_embeds = torch.softmax(u_embeds @ self.user_intent, dim=1) @ self.user_intent.T
            i_int_embeds = torch.softmax(i_embeds @ self.item_intent, dim=1) @ self.item_intent.T
            int_layer_embeds = torch.concat([u_int_embeds, i_int_embeds], dim=0)

            # Adaptive Augmentation
            int_head_embeds = torch.index_select(int_layer_embeds, 0, self.all_h_list)
            int_tail_embeds = torch.index_select(int_layer_embeds, 0, self.all_t_list)
            G_inten_indices, G_inten_values = self._adaptive_mask(int_head_embeds, int_tail_embeds)
            iaa_layer_embeds = torch_sparse.spmm(G_inten_indices, G_inten_values, self.A_in_shape[0], self.A_in_shape[1], all_embeds[i])

            # Aggregation
            gnn_embeds.append(gnn_layer_embeds)
            int_embeds.append(int_layer_embeds)
            iaa_embeds.append(iaa_layer_embeds)
            all_embeds.append(gnn_layer_embeds + int_layer_embeds + iaa_layer_embeds + all_embeds[i])

        all_embeds = torch.stack(all_embeds, dim=1)
        all_embeds = torch.sum(all_embeds, dim=1, keepdim=False)
        user_embeds, item_embeds = torch.split(all_embeds, [self.user_num, self.item_num], 0)
        self.final_embeds = all_embeds
        return user_embeds, item_embeds, gnn_embeds, int_embeds, iaa_embeds

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds

    def _llm_attention(self, node_embeds, intent_embeds):
        scores = node_embeds @ intent_embeds.T / np.sqrt(node_embeds.shape[1])
        if self.training:
            noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-10) + 1e-10)
            scores = scores + noise
        topk_values, topk_indices = torch.topk(scores, self.llm_topk, dim=1)
        values = torch.softmax(topk_values, dim=1).reshape(-1)
        rows = torch.arange(scores.shape[0], device=scores.device).unsqueeze(1)
        rows = rows.expand(-1, self.llm_topk).reshape(-1)
        return torch.sparse_coo_tensor(
            torch.stack([rows, topk_indices.reshape(-1)]), values, scores.shape
        ).coalesce()

    def _llm_forward(self):
        if not self.is_training and self.final_llm_embeds is not None:
            return torch.split(self.final_llm_embeds, [self.user_num, self.item_num], 0)

        llm_users = self.llm_adapter(self.usrint_embeds.detach())
        llm_items = self.llm_adapter(self.itmint_embeds.detach())
        all_embeds = torch.concat([llm_users, llm_items], dim=0)
        layer_embeds = [all_embeds]
        for _ in range(self.llm_lightgcn_layers):
            all_embeds = torch_sparse.spmm(
                self.G_indices, self.G_values,
                self.A_in_shape[0], self.A_in_shape[1], all_embeds
            )
            layer_embeds.append(all_embeds)
        llm_users, llm_items = torch.split(
            torch.stack(layer_embeds, dim=1).mean(dim=1),
            [self.user_num, self.item_num], 0
        )

        intent_embeds = self.llm_adapter(self.llm_intent_prototypes.detach())
        user_attention = self._llm_attention(llm_users, intent_embeds)
        item_attention = self._llm_attention(llm_items, intent_embeds)
        user_degree = torch.sparse.sum(user_attention, dim=0).to_dense().clamp_min(1e-6).unsqueeze(1)
        item_degree = torch.sparse.sum(item_attention, dim=0).to_dense().clamp_min(1e-6).unsqueeze(1)
        user_intents = torch.sparse.mm(user_attention.t(), llm_users) / user_degree
        item_intents = torch.sparse.mm(item_attention.t(), llm_items) / item_degree
        updated_intents = (user_intents + item_intents) / 2
        llm_users = llm_users + self.llm_intent_weight * 0.5 * torch.sparse.mm(user_attention, updated_intents)
        llm_items = llm_items + self.llm_intent_weight * 0.5 * torch.sparse.mm(item_attention, updated_intents)
        llm_users = F.normalize(llm_users, p=2, dim=1)
        llm_items = F.normalize(llm_items, p=2, dim=1)

        if not self.is_training:
            self.final_llm_embeds = torch.concat([llm_users, llm_items], dim=0)
        return llm_users, llm_items

    def _cal_cl_loss(self, users, items, gnn_emb, int_emb, iaa_emb):
        users = torch.unique(users)
        items = torch.unique(items) # different from original SSLRec, remove negative items
        cl_loss = 0.0
        for i in range(len(gnn_emb)):
            u_gnn_embs, i_gnn_embs = torch.split(gnn_emb[i], [self.user_num, self.item_num], 0)
            u_int_embs, i_int_embs = torch.split(int_emb[i], [self.user_num, self.item_num], 0)
            u_iaa_embs, i_iaa_embs = torch.split(iaa_emb[i], [self.user_num, self.item_num], 0)

            u_gnn_embs = u_gnn_embs[users]
            u_int_embs = u_int_embs[users]
            u_iaa_embs = u_iaa_embs[users]

            i_gnn_embs = i_gnn_embs[items]
            i_int_embs = i_int_embs[items]
            i_iaa_embs = i_iaa_embs[items]

            cl_loss += cal_infonce_loss(u_gnn_embs, u_int_embs, u_int_embs, self.cl_temperature) / u_gnn_embs.shape[0]
            cl_loss += cal_infonce_loss(u_gnn_embs, u_iaa_embs, u_iaa_embs, self.cl_temperature) / u_gnn_embs.shape[0]
            cl_loss += cal_infonce_loss(i_gnn_embs, i_int_embs, i_int_embs, self.cl_temperature) / u_gnn_embs.shape[0]
            cl_loss += cal_infonce_loss(i_gnn_embs, i_iaa_embs, i_iaa_embs, self.cl_temperature) / u_gnn_embs.shape[0]
        return cl_loss

    def cal_loss(self, batch_data):
        self.is_training = True
        self.final_llm_embeds = None
        user_embeds, item_embeds, gnn_embeds, int_embeds, iaa_embeds = self.forward()
        llm_user_embeds, llm_item_embeds = self._llm_forward()
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        llm_anc_embeds = llm_user_embeds[ancs]
        llm_pos_embeds = llm_item_embeds[poss]
        llm_neg_embeds = llm_item_embeds[negs]
        llm_bpr_loss = cal_bpr_loss(llm_anc_embeds, llm_pos_embeds, llm_neg_embeds) / llm_anc_embeds.shape[0]
        llm_bpr_loss *= self.llm_bpr_weight
        reg_loss = self.reg_weight * reg_params(self)
        cl_loss = self.cl_weight * self._cal_cl_loss(ancs, poss, gnn_embeds, int_embeds, iaa_embeds)

        # kd_loss
        user_intent_embeds = self.kd_mlp(self.usrint_embeds)
        item_intent_embeds = self.kd_mlp(self.itmint_embeds)
        ancprf_embeds, posprf_embeds, negprf_embeds = self._pick_embeds(user_intent_embeds, item_intent_embeds, batch_data)
        kd_loss = cal_infonce_loss(anc_embeds, ancprf_embeds, user_intent_embeds, self.kd_temperature) + \
                        cal_infonce_loss(pos_embeds, posprf_embeds, posprf_embeds, self.kd_temperature) + \
                        cal_infonce_loss(neg_embeds, negprf_embeds, negprf_embeds, self.kd_temperature)
        kd_loss /= anc_embeds.shape[0]
        kd_loss *= self.kd_weight

        # kd_int_loss
        int_embeds = torch.mean(torch.stack(iaa_embeds), dim=0)
        user_int_embeds, item_int_embeds = torch.split(int_embeds, [self.user_num, self.item_num], 0)
        anc_int_embeds, pos_int_embeds, neg_int_embeds = self._pick_embeds(user_int_embeds, item_int_embeds, batch_data)
        usrint_embeds = self.int_mlp(self.usrint_embeds)
        itmint_embeds = self.int_mlp(self.itmint_embeds)
        ancint_embeds, posint_embeds, negint_embeds = self._pick_embeds(usrint_embeds, itmint_embeds, batch_data)
        kd_int_loss = cal_infonce_loss(anc_int_embeds, ancint_embeds, usrint_embeds, self.kd_int_temperature) + \
                      cal_infonce_loss(pos_int_embeds, posint_embeds, posint_embeds, self.kd_int_temperature) + \
                      cal_infonce_loss(neg_int_embeds, negint_embeds, negint_embeds, self.kd_int_temperature)
        kd_int_loss /= anc_int_embeds.shape[0]
        kd_int_loss *= self.kd_int_weight

        loss = bpr_loss + llm_bpr_loss + reg_loss + cl_loss + kd_loss + kd_int_loss
        losses = {'bpr_loss': bpr_loss, 'llm_bpr_loss': llm_bpr_loss, 'reg_loss': reg_loss, 'cl_loss': cl_loss, 'kd_loss': kd_loss, 'kd_int_loss': kd_int_loss}
        return loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds, _, _, _ = self.forward()
        self.is_training = False
        llm_user_embeds, llm_item_embeds = self._llm_forward()
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        collaborative_scores = pck_user_embeds @ item_embeds.T
        llm_scores = llm_user_embeds[pck_users] @ llm_item_embeds.T
        full_preds = collaborative_scores + self.llm_score_weight * llm_scores
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
