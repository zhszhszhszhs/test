import torch as t
from torch import nn
import torch.nn.functional as F

from config.configurator import configs
from models.base_model import BaseModel
from models.loss_utils import cal_bpr_loss, reg_params

init = nn.init.xavier_uniform_


class HCCF(BaseModel):
    def __init__(self, data_handler):
        super(HCCF, self).__init__(data_handler)
        self.adj = data_handler.torch_adj

        # hyper-parameters
        self.layer_num = self._get_hyper_param('layer_num')
        self.reg_weight = self._get_hyper_param('reg_weight')
        self.cl_weight = self._get_hyper_param('cl_weight')
        self.temperature = self._get_hyper_param('temperature')
        self.keep_rate = self._get_hyper_param('keep_rate')
        self.hyper_num = self._get_hyper_param('hyper_num')

        # model parameters, following HCCF/torchVersion/Model.py
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        self.user_hyper = nn.Parameter(init(t.empty(self.embedding_size, self.hyper_num)))
        self.item_hyper = nn.Parameter(init(t.empty(self.embedding_size, self.hyper_num)))

        self.gcn_layer = GCNLayer()
        self.hgnn_layer = HGNNLayer()
        self.edge_dropper = SpAdjDropEdge()

        self.final_embeds = None
        self.is_training = False

    def _get_hyper_param(self, name):
        if name in self.hyper_config:
            return self.hyper_config[name]
        return configs['model'][name]

    def forward(self, adj=None, keep_rate=1.0):
        if adj is None:
            adj = self.adj
        if not self.is_training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:], None, None

        embeds = t.concat([self.user_embeds, self.item_embeds], dim=0)
        latents = [embeds]
        gcn_latents = []
        hyper_latents = []
        user_hyper_adj = self.user_embeds @ self.user_hyper
        item_hyper_adj = self.item_embeds @ self.item_hyper

        for _ in range(self.layer_num):
            dropped_adj = self.edge_dropper(adj, keep_rate)
            gcn_embeds = self.gcn_layer(dropped_adj, latents[-1])

            dropped_user_hyper = F.dropout(
                user_hyper_adj,
                p=1 - keep_rate,
                training=self.is_training and keep_rate < 1.0,
            )
            dropped_item_hyper = F.dropout(
                item_hyper_adj,
                p=1 - keep_rate,
                training=self.is_training and keep_rate < 1.0,
            )
            hyper_user_embeds = self.hgnn_layer(dropped_user_hyper, latents[-1][:self.user_num])
            hyper_item_embeds = self.hgnn_layer(dropped_item_hyper, latents[-1][self.user_num:])
            hyper_embeds = t.concat([hyper_user_embeds, hyper_item_embeds], dim=0)

            gcn_latents.append(gcn_embeds)
            hyper_latents.append(hyper_embeds)
            latents.append(gcn_embeds + hyper_embeds)

        embeds = sum(latents)
        self.final_embeds = embeds
        return embeds[:self.user_num], embeds[self.user_num:], gcn_latents, hyper_latents

    def _cal_cl_loss(self, users, items, gcn_latents, hyper_latents):
        cl_loss = 0.0
        users = t.unique(users)
        items = t.unique(items)
        for layer_idx in range(self.layer_num):
            gcn_embeds = gcn_latents[layer_idx].detach()
            hyper_embeds = hyper_latents[layer_idx]
            cl_loss += self._contrast_loss(
                gcn_embeds[:self.user_num],
                hyper_embeds[:self.user_num],
                users,
            )
            cl_loss += self._contrast_loss(
                gcn_embeds[self.user_num:],
                hyper_embeds[self.user_num:],
                items,
            )
        return cl_loss

    def _contrast_loss(self, embeds1, embeds2, nodes):
        embeds1 = F.normalize(embeds1 + 1e-8, p=2, dim=1)
        embeds2 = F.normalize(embeds2 + 1e-8, p=2, dim=1)
        picked_embeds1 = embeds1[nodes]
        picked_embeds2 = embeds2[nodes]
        numerator = t.exp(t.sum(picked_embeds1 * picked_embeds2, dim=-1) / self.temperature)
        denominator = t.exp(picked_embeds1 @ picked_embeds2.T / self.temperature).sum(-1) + 1e-8
        return -t.log(numerator / denominator + 1e-8).mean()

    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds, gcn_latents, hyper_latents = self.forward(self.adj, self.keep_rate)
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]

        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        reg_loss = self.reg_weight * reg_params(self)
        cl_loss = self.cl_weight * self._cal_cl_loss(ancs, poss, gcn_latents, hyper_latents)
        loss = bpr_loss + reg_loss + cl_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'cl_loss': cl_loss}
        return loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds, _, _ = self.forward(self.adj, 1.0)
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds


class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        return t.spmm(adj, embeds)


class HGNNLayer(nn.Module):
    def __init__(self):
        super(HGNNLayer, self).__init__()

    def forward(self, adj, embeds):
        latents = adj.T @ embeds
        return adj @ latents


class SpAdjDropEdge(nn.Module):
    def __init__(self):
        super(SpAdjDropEdge, self).__init__()

    def forward(self, adj, keep_rate):
        if keep_rate == 1.0:
            return adj
        vals = adj._values()
        idxs = adj._indices()
        edge_num = vals.size()
        mask = (t.rand(edge_num, device=vals.device) + keep_rate).floor().type(t.bool)
        new_vals = vals[mask] / keep_rate
        new_idxs = idxs[:, mask]
        return t.sparse_coo_tensor(new_idxs, new_vals, adj.shape, device=vals.device).coalesce()
