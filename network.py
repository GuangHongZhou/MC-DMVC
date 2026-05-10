import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize

class FeatureFilter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        importance = self.gate(z)
        return z * importance

class CrossViewFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)

        # AoA gating
        self.fc_g = nn.Linear(dim * 2, dim)
        self.fc_a = nn.Linear(dim * 2, dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.view_weight_net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1)
        )

    def forward(self, zs):
        """
        zs: list of [N, d] from different views
        """
        N, d = zs[0].shape

        z_stack = torch.stack(zs, dim=1)
        Q = self.query_proj(z_stack)
        K = self.key_proj(z_stack)
        Vv = self.value_proj(z_stack)

        # self-attention across views (view-wise)
        attn = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) / (d ** 0.5), dim=-1)
        attn_out = torch.bmm(attn, Vv)

        # combine attention output with original feature
        combined = torch.cat([attn_out, z_stack], dim=-1)
        g = self.sigmoid(self.fc_g(combined))
        h = self.tanh(self.fc_a(combined))

        fused_per_view = g * h

        view_scores = self.view_weight_net(fused_per_view).squeeze(-1)
        view_weights = F.softmax(view_scores, dim=1)

        fused = torch.sum(fused_per_view * view_weights.unsqueeze(-1), dim=1)
        return fused


class FeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )
    def forward(self, x):
        return self.decoder(x)

class MCMVC(nn.Module):
    def __init__(self, view, input_size, low_feature_dim, high_feature_dim, device):
        super(MCMVC, self).__init__()
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], low_feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], low_feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.Specific_view = nn.Sequential(
            nn.Linear(low_feature_dim, high_feature_dim),
        )
        
        self.view = view

        self.Common_view = nn.Sequential(
            nn.Linear(low_feature_dim, high_feature_dim),
        )
        self.gcvr = CrossViewFusion(low_feature_dim)

        self.feature_filter = FeatureFilter(low_feature_dim)


    def forward(self, xs):
        xrs = []
        zs = []
        hs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = normalize(self.Specific_view(z), dim=1)
            xr = self.decoders[v](z)
            hs.append(h)
            zs.append(z)
            xrs.append(xr)
        return xrs, zs, hs

    def MCMVC(self, xs):
        zs = []
        Alist = []
        for v in range(self.view):
            x = xs[v]
            A = self.computeA(F.normalize(x))
            Alist.append(A)
            z = self.encoders[v](x)
            z= self.feature_filter(z)
            zs.append(z)

        fused_z = self.gcvr(zs)  # [N, hidden_dim]

        z0 = F.normalize(self.Common_view(fused_z), dim=1)
        A_avg = torch.mean(torch.stack(Alist), dim=0)

        return z0, A_avg

    def computeA(self, x, k=10, sigma=1.0):
            N = x.shape[0]
            device = x.device

            x = F.normalize(x, p=2, dim=1)
            dis2 = (-2 * x.mm(x.t())) + torch.sum(x ** 2, dim=1, keepdim=True) + torch.sum(x ** 2, dim=1, keepdim=True).t()
            if sigma is None:
                sigma = torch.median(dis2).sqrt().item()
            topk_idx = torch.topk(-dis2, k=k + 1, dim=1).indices
            A = torch.zeros(N, N, device=device)
            row_idx = torch.arange(N, device=device).unsqueeze(1).repeat(1, k + 1)
            mask = topk_idx != row_idx
            neighbor_dis2 = dis2[row_idx[mask], topk_idx[mask]]
            A[row_idx[mask], topk_idx[mask]] = torch.exp(-neighbor_dis2 / (2 * sigma ** 2))

            return A
