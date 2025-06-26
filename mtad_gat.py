import torch
import torch.nn as nn

from STTNS.STTNs import STTNSNet
from modules import (
    GRULayer,
    Forecasting_Model
)


class MTAD_GAT(nn.Module):
    """ MTAD-GAT model class.

    :param n_features: 输入特征数
    :param window_size: 输入序列的长度
    :param out_dim: 要输出的特征数
    :param kernel_size: 用于一维卷积的内核大小
    :param feat_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in feat-oriented GAT layer
    :param time_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in time-oriented GAT layer
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param gru_n_layers: number of layers in the GRU layer
    :param gru_hid_dim: hidden dimension in the GRU layer
    :param forecast_n_layers: number of layers in the FC-based Forecasting Model
    :param forecast_hid_dim: hidden dimension in the FC-based Forecasting Model
    :param recon_n_layers: number of layers in the GRU-based Reconstruction Model
    :param recon_hid_dim: hidden dimension in the GRU-based Reconstruction Model
    :param dropout: dropout rate
    :param alpha: negative slope used in the leaky rely activation function

    """

    def __init__(
            self,
            n_features,
            window_size,
            out_dim,
            k,
            graph,
            embed_size,
            in_channels,
            gru_n_layers=1,
            gru_hid_dim=150,
            forecast_n_layers=1,
            forecast_hid_dim=150,
            dropout=0.2
    ):
        super(MTAD_GAT, self).__init__()
        self.sttn = STTNSNet(n_features, k, in_channels, embed_size, window_size, dropout=dropout, graph=graph)
        self.gru = GRULayer(2 * n_features, gru_hid_dim, gru_n_layers, dropout)
        self.forecasting_model = Forecasting_Model(gru_hid_dim, forecast_hid_dim, out_dim, forecast_n_layers, dropout)
        self.feature_idx = torch.arange(n_features).cuda()
        self.temporal_idx = torch.arange(window_size).cuda()
       # self.linear1 = nn.Linear(2*n_features*window_size, gru_hid_dim)#加入线性层移除GRU

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        b, n, k = x.shape

        data = x.unsqueeze(1).permute(0, 1, 3, 2)  # (b, 1, k, n)
        h = self.sttn(data)

        h_cat = torch.cat([x, h.permute(0, 2, 1)], dim=2)  # (b, n, 3k)

        _, h_end = self.gru(h_cat)#移除GRU包括下面一行
        # h_cat1 = h_cat.view(x.shape[0], -1)#线性层代替GRU
        #h_end = self.linear1(h_cat1)#线性层代替GRU
        h_end = h_end.view(x.shape[0], -1)  # Hidden state for last timestamp      batch_size * gru_hid_dim #

        predictions = self.forecasting_model(h_end)

        return predictions
