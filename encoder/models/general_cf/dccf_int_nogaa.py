import torch
import pickle
import numpy as np
import torch_sparse
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F

from config.configurator import configs
from models.aug_utils import NodeMask
from models.general_cf.lightgcn import BaseModel
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss, ssl_con_loss

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
        self.llm_weight = self.hyper_config['llm_weight']
        self.mask_ratio = self.hyper_config['mask_ratio']
        self.recon_weight = self.hyper_config['recon_weight']
        self.re_temperature = self.hyper_config['re_temperature']

        # model parameters
        self.user_embeds = nn.Embedding(self.user_num, self.embedding_size)
        self.item_embeds = nn.Embedding(self.item_num, self.embedding_size)
        self.user_intent = torch.nn.Parameter(init(torch.empty(self.embedding_size, self.intent_num)), requires_grad=True)
        self.item_intent = torch.nn.Parameter(init(torch.empty(self.embedding_size, self.intent_num)), requires_grad=True)

        # Cache only clean evaluation embeddings; training views are masked.
        self.final_embeds = None

        # intent information
        self.usrint_embeds = torch.tensor(configs['usrint_embeds']).float().cuda()
        self.itmint_embeds = torch.tensor(configs['itmint_embeds']).float().cuda()
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
        self.llm_mlp = nn.Sequential(
            nn.Linear(self.usrint_embeds.shape[1], (self.usrint_embeds.shape[1] + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.usrint_embeds.shape[1] + self.embedding_size) // 2, self.embedding_size)
        )
        self.llm_masker = NodeMask(self.mask_ratio, self.embedding_size)
        self.llm_recon_mlp = nn.Sequential(
            nn.Linear(self.embedding_size, (self.usrint_embeds.shape[1] + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.usrint_embeds.shape[1] + self.embedding_size) // 2, self.usrint_embeds.shape[1])
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
        for m in self.llm_mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)
        for m in self.llm_recon_mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)
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
        if self.training:
            self.final_embeds = None
        elif self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:], None, None, None, None, None

        all_embeds = [torch.concat([self.user_embeds.weight, self.item_embeds.weight], dim=0)]
        llm_init_embeds = torch.concat([
            self.llm_mlp(self.usrint_embeds),
            self.llm_mlp(self.itmint_embeds)
        ], dim=0)
        llm_seeds = None
        if self.training:
            llm_init_embeds, llm_seeds = self.llm_masker(llm_init_embeds)
        llm_embeds = [llm_init_embeds]
        gnn_embeds, int_embeds, iaa_embeds = [], [], []

        for i in range(0, self.layer_num):
            # Graph-based Message Passing
            gnn_layer_embeds = torch_sparse.spmm(self.G_indices, self.G_values, self.A_in_shape[0], self.A_in_shape[1], all_embeds[i])

            # Intent-aware Information Aggregation
            u_embeds, i_embeds = torch.split(all_embeds[i], [self.user_num, self.item_num], 0)
            u_int_embeds = torch.softmax(u_embeds @ self.user_intent, dim=1) @ self.user_intent.T
            i_int_embeds = torch.softmax(i_embeds @ self.item_intent, dim=1) @ self.item_intent.T
            int_layer_embeds = torch.concat([u_int_embeds, i_int_embeds], dim=0)

            # Intent-aware Adaptive Augmentation
            int_head_embeds = torch.index_select(int_layer_embeds, 0, self.all_h_list)
            int_tail_embeds = torch.index_select(int_layer_embeds, 0, self.all_t_list)
            G_inten_indices, G_inten_values = self._adaptive_mask(int_head_embeds, int_tail_embeds)
            iaa_layer_embeds = torch_sparse.spmm(G_inten_indices, G_inten_values, self.A_in_shape[0], self.A_in_shape[1], all_embeds[i])

            # Propagate the masked LLM intent view over the same IAA graph,
            # rather than over the complete normalized user-item graph.
            llm_layer_embeds = torch_sparse.spmm(
                G_inten_indices,
                G_inten_values,
                self.A_in_shape[0],
                self.A_in_shape[1],
                llm_embeds[i]
            )

            # Aggregation
            gnn_embeds.append(gnn_layer_embeds)
            int_embeds.append(int_layer_embeds)
            iaa_embeds.append(iaa_layer_embeds)
            all_embeds.append(gnn_layer_embeds + int_layer_embeds + iaa_layer_embeds + all_embeds[i])
            llm_embeds.append(llm_layer_embeds)

        all_embeds = torch.stack(all_embeds, dim=1)
        all_embeds = torch.sum(all_embeds, dim=1, keepdim=False)
        llm_embeds = torch.stack(llm_embeds, dim=1)
        llm_embeds = torch.sum(llm_embeds, dim=1, keepdim=False)
        cf_fusion_embeds = F.normalize(all_embeds, p=2, dim=-1)
        llm_fusion_embeds = F.normalize(llm_embeds, p=2, dim=-1)
        all_embeds = cf_fusion_embeds + self.llm_weight * llm_fusion_embeds
        user_embeds, item_embeds = torch.split(all_embeds, [self.user_num, self.item_num], 0)
        if not self.training:
            self.final_embeds = all_embeds
        return user_embeds, item_embeds, gnn_embeds, int_embeds, iaa_embeds, llm_embeds, llm_seeds

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds

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

    def _reconstruction(self, llm_embeds, seeds):
        reconstructed_embeds = self.llm_recon_mlp(llm_embeds[seeds])
        user_seed_mask = seeds < self.user_num
        target_embeds = self.usrint_embeds.new_empty(
            (seeds.shape[0], self.usrint_embeds.shape[1])
        )
        target_embeds[user_seed_mask] = self.usrint_embeds[seeds[user_seed_mask]]
        item_seeds = seeds[~user_seed_mask] - self.user_num
        target_embeds[~user_seed_mask] = self.itmint_embeds[item_seeds]
        return ssl_con_loss(reconstructed_embeds, target_embeds, self.re_temperature)

    def cal_loss(self, batch_data):
        user_embeds, item_embeds, gnn_embeds, int_embeds, iaa_embeds, llm_embeds, llm_seeds = self.forward()
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
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

        # Reconstruct the original LLM intent/profile vectors only for masked nodes.
        recon_loss = self.recon_weight * self._reconstruction(llm_embeds, llm_seeds)

        loss = bpr_loss + reg_loss + cl_loss + kd_loss + kd_int_loss + recon_loss
        losses = {
            'bpr_loss': bpr_loss,
            'reg_loss': reg_loss,
            'cl_loss': cl_loss,
            'kd_loss': kd_loss,
            'kd_int_loss': kd_int_loss,
            'recon_loss': recon_loss
        }
        return loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds, _, _, _, _, _ = self.forward()
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
