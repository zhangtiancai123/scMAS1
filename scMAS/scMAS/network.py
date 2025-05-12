import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils import *
import math, os

import numpy as np
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import confusion_matrix
import math
from sklearn.metrics.pairwise import cosine_similarity
from community import community_louvain

import torch
import torch.nn.functional as F


from sklearn.neighbors import kneighbors_graph
import igraph as ig

import scipy 
import scipy.sparse as sp

# The NBLoss class implements the negative binomial regression loss function, 
# which is used to fit discrete count data.
class NBLoss(nn.Module):
    def __init__(self):
        super(NBLoss, self).__init__()

    def forward(self, x, mean, disp, scale_factor=1.0):
        # x: Input data, i.e., the actual observed values (e.g., gene expression counts).
        # mean: The mean predicted by the model.
        # disp: Dispersion parameter, controlling the variance of the data.
        # scale_factor: Scaling factor used to adjust the predicted mean, 
        # typically related to batch size.
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor

        # t1: Compute the logarithmic part of the gamma function, 
        # with eps added for numerical stability.
        # t2: Compute the remaining part of the loss, including logarithmic operations.
        # result: The final negative binomial regression loss, averaged via torch.mean 
        # to return a scalar value as the final loss.
        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
        t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
        result = t1 + t2

        result = torch.mean(result)
        return result


# The ZINBLoss class implements the zero-inflated negative binomial (ZINB) regression loss function, 
# suitable for sparse count data (e.g., scRNA-seq data) where zeros may be overrepresented.
class ZINBLoss(nn.Module):
    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
        # pi: Zero-inflation coefficient, representing the probability of the 
        # overinflated zeros in the data.
        # ridge_lambda: L2 regularization coefficient, controlling the penalty term 
        # for zero-inflation probability to prevent overfitting.
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor

        # nb_final: Compute the standard negative binomial loss.
        # nb_case and zero_case: Handle different cases for observed zeros and non-zeros.
        # result: Return the averaged final loss value.
        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
        t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0-pi+eps)
        zero_nb = torch.pow(disp/(disp+mean+eps), disp)
        zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)
        
        if ridge_lambda > 0:
            ridge = ridge_lambda*torch.square(pi)
            result += ridge
        
        result = torch.mean(result)
        return result


# The MeanAct class implements an activation function that transforms the input x 
# into positive values via the exponential function and limits their range.
class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    # torch.clamp: Clamp the output to ensure it stays within the specified range, 
    # preventing numerical overflow.
    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


# The DispAct class defines an activation function that processes the input x using 
# the softplus function and limits it to a specified range.
class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    # softplus: A smooth version of the ReLU function that ensures positive outputs 
    # without sudden cutoffs like ReLU.
    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)



# scRNA encoder class based on masked attention mechanism
class ScRNAEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=8, dropout=0.1):   # num_heads=8
        super(ScRNAEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout

        # Multihead Attention Layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout
        )
        # self.fc = nn.Linear(input_dim, hidden_dim)
        self.fc = nn.Linear(
            input_dim, hidden_dim, dtype=torch.float32  # Uniformly specified as float32 type, which can be adjusted according to actual conditions
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask=None):
        # Input x is of shape (seq_len, batch_size, input_dim)
        # Convert input to hidden dimension using linear layer
        x = self.fc(x)
        
        # Attention layer expects (seq_len, batch_size, hidden_dim)
        attn_output, _ = self.attention(x, x, x, key_padding_mask=mask)
        attn_output = self.norm(attn_output + x)  # Residual connection
        return attn_output


# ATAC encoder class based on multi-head attention mechanism
class ATACEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=8, dropout=0.1):  # num_heads=8
        super(ATACEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout

        # Multihead Attention Layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout
        )
        # self.fc = nn.Linear(input_dim, hidden_dim)
        self.fc = nn.Linear(
            input_dim, hidden_dim, dtype=torch.float32  # Uniformly specified as float32 type, which can be adjusted according to actual conditions
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask=None):
        # Input x is of shape (seq_len, batch_size, input_dim)
        # Convert input to hidden dimension using linear layer
        x = self.fc(x)
        
        # Attention layer expects (seq_len, batch_size, hidden_dim)
        attn_output, _ = self.attention(x, x, x, key_padding_mask=mask)
        attn_output = self.norm(attn_output + x)  # Residual connection
        return attn_output


def NormalNwt(layers, type, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        net.append(nn.BatchNorm1d(layers[i], affine=True))
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "selu":
            net.append(nn.SELU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
        elif activation == "elu":
            net.append(nn.ELU())
    return nn.Sequential(*net)


def snn_from_data(latent, k):
    knn_mat = kneighbors_graph(latent, k, mode='connectivity', include_self=False)
    snn_mat = knn_mat.dot(knn_mat.T)
    snn_mat.data[:] = snn_mat.data / (k + k - snn_mat.data)
    snn_mat.setdiag(0)
    return snn_mat


def graph_from_snn(snn_mat):
    snn_mat.eliminate_zeros()
    graph = ig.Graph(
        n=snn_mat.shape[0],
        edges=list(zip(*snn_mat.nonzero())),
        edge_attrs={'weight': snn_mat.data},
        directed=True
    )
    return graph


class scMAS(nn.Module):  
    def __init__(
        self, 
        input_dim1, 
        input_dim2,
        encodeLayer1=[],
        encodeLayer2=[],
        decodeLayer1=[], 
        decodeLayer2=[], 
        tau=1., 
        t=10, 
        device="cuda:1",
        activation="elu", 
        sigma1=2.5, 
        sigma2=.1, 
        alpha=1., 
        gamma=1., 
        phi1=0.0001, 
        phi2=0.0001, 
        cutoff=0.5
    ):
        super(scEMC, self).__init__()
        self.tau = tau
        self.input_dim1 = input_dim1 
        self.input_dim2 = input_dim2
        self.cutoff = cutoff
        self.activation = activation
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.alpha = alpha
        self.gamma = gamma
        self.phi1 = phi1
        self.phi2 = phi2
        self.t = t
        self.device = device

        self.encoder1 = ScRNAEncoder(input_dim1, encodeLayer1[-1]) 
        self.encoder2 = ATACEncoder(input_dim2, encodeLayer2[-1])   

        self.decoder1 = NormalNet(decodeLayer1, type="decode", activation=activation)
        self.decoder2 = NormalNwt(decodeLayer2, type="decode", activation=activation)
        self.dec_mean1 = nn.Sequential(nn.Linear(decodeLayer1[-1], input_dim1), MeanAct())
        self.dec_disp1 = nn.Sequential(nn.Linear(decodeLayer1[-1], input_dim1), DispAct())
        self.dec_mean2 = nn.Sequential(nn.Linear(decodeLayer2[-1], input_dim2), MeanAct())
        self.dec_disp2 = nn.Sequential(nn.Linear(decodeLayer2[-1], input_dim2), DispAct())
        self.dec_pi1 = nn.Sequential(nn.Linear(decodeLayer1[-1], input_dim1), nn.Sigmoid())
        self.dec_pi2 = nn.Sequential(nn.Linear(decodeLayer2[-1], input_dim2), nn.Sigmoid())
        
        self.zinb_loss = ZINBLoss()
        self.z_dim = decodeLayer1[0]
        self.trans_enc = nn.TransformerEncoderLayer(
            d_model=2*encodeLayer1[-1], nhead=1, dim_feedforward=256
        )
        self.extract_layers = nn.TransformerEncoder(self.trans_enc, num_layers=1)

        # Added for Louvain clustering
        self.cluster_model = None
        self.epoch = 0

        # Add a new parameter to control the weight of the Euclidean distance loss
        self.lambda_euclidean = 0.00001  # Can be adjusted as needed

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)
        
    def cal_latent(self, z):
        sum_y = torch.sum(torch.square(z), dim=1)
        num = -2.0 * torch.matmul(z, z.t()) + torch.reshape(sum_y, [-1, 1]) + sum_y
        num = num / self.alpha
        num = torch.pow(1.0 + num, -(self.alpha + 1.0) / 2.0)
        zerodiag_num = num - torch.diag(torch.diag(num))
        latent_p = (zerodiag_num.t() / torch.sum(zerodiag_num, dim=1)).t()
        return num, latent_p

    def louvain_loss(self, zbatch):
        G = nx.Graph(zbatch.data.cpu().numpy())
        partition = community_louvain.best_partition(G)
        modularity = community_louvain.modularity(partition, G)
        return modularity

    def target_distribution(self, q):
        p = q**2 / q.sum(0)
        return (p.t() / p.sum(1)).t()

    def euclidean_distance_loss(self, latent, original):  # Calculate Euclidean distance loss
        """
        Calculate the difference between Euclidean distances in the latent space and diffusion distances in the original feature space.
        :param latent: Representation in the latent space (batch_size, latent_dim)
        :param original: Representation in the original feature space (batch_size, original_dim)
        :return: Euclidean distance loss
        """
        # Calculate pairwise Euclidean distances in the latent space
        latent_dist = torch.cdist(latent, latent, p=2)  # p=2 indicates Euclidean distance

        # Calculate pairwise diffusion distances in the original feature space
        original_dist = torch.cdist(original, original, p=2)

        # Calculate the difference between the two
        diff = torch.abs(latent_dist - original_dist)

        # Return the average difference as the loss
        return torch.mean(diff)

    def forward(self, x1, x2):
        # Additive Gaussian noise processing for input data x1 and x2
        x1_1 = x1 + torch.randn_like(x1) * self.sigma1
        x2_1 = x2 + torch.randn_like(x2) * self.sigma2
        
        # Pass the processed input data x1_1 and x2_1 through the two encoders self.encoder1 and self.encoder2 to get encoded features xh1 and xh2
        xh1 = self.encoder1(x1_1)
        xh2 = self.encoder2(x2_1)
        
        # Concatenate the encoded features xh1 and xh2
        h = self.extract_layers(torch.cat((xh1, xh2), 1))
        h = torch.cat([xh1, h], dim=-1)

        # Pass the concatenated feature h through the two decoders self.decoder1 and self.decoder2 to get reconstructed features h1 and h2
        h1 = self.decoder1(h)
        mean1 = self.dec_mean1(h1)
        disp1 = self.dec_disp1(h1)
        pi1 = self.dec_pi1(h1)

        h2 = self.decoder2(h)
        mean2 = self.dec_mean2(h2)
        disp2 = self.dec_disp2(h2)
        pi2 = self.dec_pi2(h2)

        # Calculate MSE loss
        mse_loss1 = F.mse_loss(mean1, x1)
        mse_loss2 = F.mse_loss(mean2, x2)

        # Calculate the expected loss of negative log-likelihood
        nll_loss1 = F.mse_loss(mean1, x1)  # Approximate negative log-likelihood using MSE
        nll_loss2 = F.mse_loss(mean2, x2)  # Approximate negative log-likelihood using MSE
        nll_loss = nll_loss1 + nll_loss2

        # Pass the original input data x1 and x2 through the encoders again to get features x1_2 and x2_2, then concatenate and process
        x1_2 = self.encoder1(x1)
        x2_2 = self.encoder2(x2)
        h00 = self.extract_layers(torch.cat((x1_2, x2_2), 1))
        h00 = torch.cat([x1_2, h00], dim=-1)

        num, lq = self.cal_latent(h00)
        
        # Return MSE loss and negative log-likelihood expected loss
        return h00, num, lq, mean1, mean2, disp1, disp2, pi1, pi2, mse_loss1, mse_loss2, nll_loss

    def encodeBatch(self, X1, X2, batch_size=256):
        encoded = []
        self.eval()
        num = X1.shape[0]
        num_batch = int(math.ceil(1.0 * X1.shape[0] / batch_size))
        for batch_idx in range(num_batch):
            x1batch = X1[batch_idx * batch_size : min((batch_idx + 1) * batch_size, num)]
            x2batch = X2[batch_idx * batch_size : min((batch_idx + 1) * batch_size, num)]

            inputs1 = Variable(x1batch)
            inputs2 = Variable(x2batch)
            z, _, _, _, _, _, _, _, _, _, _, _ = self.forward(inputs1, inputs2)
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded

    def kldloss(self, p, q):
        c1 = -torch.sum(p * torch.log(q), dim=-1)
        c2 = -torch.sum(p * torch.log(p), dim=-1)
        return torch.mean(c1 - c2)

    def pretrain_autoencoder(
        self, 
        X1, 
        X_raw1, 
        sf1, 
        X2, 
        X_raw2, 
        sf2, 
        batch_size=256, 
        lr=0.001, 
        epochs=400, 
        ae_save=True, 
        ae_weights='AE_weights.pth.tar'
    ):
        num_batch = int(math.ceil(1.0 * X1.shape[0] / batch_size))
        X2_dense = X2.toarray() if scipy.sparse.issparse(X2) else X2
        X_raw2_dense = X_raw2.toarray() if scipy.sparse.issparse(X_raw2) else X_raw2
        sf2_dense = sf2.toarray() if scipy.sparse.issparse(sf2) else sf2

        print(X1.shape, X_raw1.shape, sf1.shape)
        print(X2_dense.shape, X_raw2_dense.shape, sf2_dense.shape)

        # Filter based on the number of samples in X1
        common_indices = np.arange(min(X1.shape[0], X2_dense.shape[0]))
        X1 = X1[common_indices]
        X_raw1 = X_raw1[common_indices]
        sf1 = sf1[common_indices]
        X2_dense = X2_dense[common_indices]
        X_raw2_dense = X_raw2_dense[common_indices]
        sf2_dense = sf2_dense[common_indices]

        dataset = TensorDataset(
            torch.Tensor(X1), 
            torch.Tensor(X_raw1), 
            torch.Tensor(sf1), 
            torch.Tensor(X2_dense), 
            torch.Tensor(X_raw2_dense), 
            torch.Tensor(sf2_dense)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("Pretraining stage")
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), 
            lr=lr, 
            amsgrad=True
        )
        num = X1.shape[0]
        for epoch in range(epochs):
            loss_val = 0
            recon_loss1_val = 0
            recon_loss2_val = 0
            kl_loss_val = 0
            for batch_idx, (x1_batch, x_raw1_batch, sf1_batch, x2_batch, x_raw2_batch, sf2_batch) in enumerate(dataloader):
                x1_tensor = Variable(x1_batch).to(self.device)
                x_raw1_tensor = Variable(x_raw1_batch).to(self.device)
                sf1_tensor = Variable(sf1_batch).to(self.device)
                x2_tensor = Variable(x2_batch).to(self.device)
                x_raw2_tensor = Variable(x_raw2_batch).to(self.device)
                sf2_tensor = Variable(sf2_batch).to(self.device)
                (zbatch, z_num, lqbatch, mean1_tensor, mean2_tensor, disp1_tensor, 
                 disp2_tensor, pi1_tensor, pi2_tensor, mseloss1_tensor, mseloss2_tensor, 
                 nllloss_tensor) = self.forward(x1_tensor, x2_tensor)
                recon_loss1 = self.zinb_loss(
                    x=x_raw1_tensor, 
                    mean=mean1_tensor, 
                    disp=disp1_tensor, 
                    pi=pi1_tensor, 
                    scale_factor=sf1_tensor
                )
                recon_loss2 = self.zinb_loss(
                    x=x_raw2_tensor, 
                    mean=mean2_tensor, 
                    disp=disp2_tensor, 
                    pi=pi2_tensor, 
                    scale_factor=sf2_tensor
                )
                lpbatch = self.target_distribution(lqbatch)
                lqbatch = lqbatch + torch.diag(torch.diag(z_num))
                lpbatch = lpbatch + torch.diag(torch.diag(z_num))
                kl_loss = self.kldloss(lpbatch, lqbatch) 
                if epoch + 1 >= epochs * self.cutoff:
                    loss = recon_loss1 + recon_loss2 + kl_loss * self.phi1
                else:
                    loss = recon_loss1 + recon_loss2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val += loss.item() * len(x1_batch)
                recon_loss1_val += recon_loss1.item() * len(x1_batch)
                recon_loss2_val += recon_loss2.item() * len(x2_batch)
                if epoch + 1 >= epochs * self.cutoff:
                    kl_loss_val += kl_loss.item() * len(x1_batch)

            loss_val = loss_val / num
            recon_loss1_val = recon_loss1_val / num
            recon_loss2_val = recon_loss2_val / num
            kl_loss_val = kl_loss_val / num
            if epoch % self.t == 0:
                print('Pretrain epoch {}, Total loss:{:.6f}, ZINB loss1:{:.6f}, ZINB loss2:{:.6f}, KL loss:{:.6f}'.format(
                    epoch + 1, loss_val, recon_loss1_val, recon_loss2_val, kl_loss_val
                ))
            torch.cuda.empty_cache()  # Add this line to free up GPU memory

        if ae_save:
            print("Saving autoencoder weights to %s" % ae_weights)
            torch.save({
                'ae_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, ae_weights)

    def save_checkpoint(self, state, index, filename):
        newfilename = os.path.join(filename, 'FTcheckpoint_%d.pth.tar' % index)
        torch.save(state, newfilename)

    def fit(
        self, 
        X1, 
        X_raw1, 
        sf1, 
        X2, 
        X_raw2, 
        sf2, 
        lamclu, 
        lamrec, 
        lammse, 
        lameuc, 
        lamnll, 
        y=None, 
        lr=0.001, 
        batch_size=256, 
        num_epochs=10, 
        update_interval=1, 
        tol=1e-3, 
        save_dir=""
    ):
        '''X: tensor data'''
        print("Clustering stage")
        X1 = torch.tensor(X1).to(self.device)
    
        # Check if X_raw1 is a sparse matrix and convert to dense if so
        if sp.issparse(X_raw1):
            X_raw1 = X_raw1.toarray()
        X_raw1 = torch.tensor(X_raw1).to(self.device)
        
        sf1 = torch.tensor(sf1).to(self.device)
        X2 = torch.tensor(X2).to(self.device)
        
        # Check if X_raw2 is a sparse matrix and convert to dense if so
        if sp.issparse(X_raw2):
            X_raw2 = X_raw2.toarray()
        X_raw2 = torch.tensor(X_raw2).to(self.device)
        
        sf2 = torch.tensor(sf2).to(self.device)
        optimizer = torch.optim.Adadelta(
            filter(lambda p: p.requires_grad, self.parameters()), 
            lr=lr, 
            rho=.95
        )

        print("Initializing clustering with Louvain.")
        Zdata = self.encodeBatch(X1, X2, batch_size=batch_size)
        latent = Zdata.data.cpu().numpy()
        k = 20  
        snn_mat = snn_from_data(latent, k)
        graph = graph_from_snn(snn_mat)
        undirected = graph.copy()
        undirected.to_undirected(combine_edges='sum')
        # Re获取无向图的边权重
        weights = undirected.es['weight']
        res = undirected.community_multilevel(weights=weights, return_levels=False)
        self.y_pred = np.array(res.membership)
        self.y_pred_last = self.y_pred

        if y is not None:
            nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 4)
            ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 4)
            print('Initializing Louvain: ARI= %.4f, NMI= %.4f' % (ari, nmi))

        self.train()
        num = X1.shape[0]
        num_batch = int(math.ceil(1.0 * X1.shape[0] / batch_size))

        final_ami, final_nmi, final_ari, final_epoch = 0, 0, 0, 0

        stop = 0

        Loss_save = []


        for epoch in range(num_epochs):
            if epoch % update_interval == 0:
                Zdata = self.encodeBatch(X1, X2, batch_size=batch_size)
                latent = Zdata.data.cpu().numpy()
                snn_mat = snn_from_data(latent, k)
                graph = graph_from_snn(snn_mat)
                undirected = graph.copy()
                undirected.to_undirected(combine_edges='sum')
                # Re获取无向图的边权重
                weights = undirected.es['weight']
                res = undirected.community_multilevel(weights=weights, return_levels=False)
                self.y_pred = np.array(res.membership)
                if y is not None:
                    final_nmi = nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 4)
                    final_ari = ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 4)
                    pur = np.round(Purity_score(y, self.y_pred), 4)
                    final_epoch = epoch + 1

                    print('Clustering   %d: ARi= %.4f, NMI= %.4f' % (epoch + 1, ari, nmi))

                # Check stop criterion
                delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / num
                self.y_pred_last = self.y_pred

                if epoch > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print("Reach tolerance threshold. Stopping training.")
                    break

            # Train 1 epoch for clustering loss
            loss_val = 0.0
            recon_loss1_val = 0.0
            recon_loss2_val = 0.0
            kl_loss_val = 0.0
            mse_loss_val = 0.0
            euclidean_loss_val = 0.0
            nll_loss_val = 0.0
            for batch_idx in range(num_batch):
                x1_batch = X1[batch_idx * batch_size : min((batch_idx + 1) * batch_size, num)]
                x_raw1_batch = X_raw1[batch_idx * batch_size : min((batch_idx + 1) * batch_size, num)]
                sf1_batch = sf1[batch_idx * batch_size : min((batch_idx + 1) * batch_size, num)]
                x2_batch = X2[batch_idx * batch_size : min((batch_idx + 1) * batch_size, num)]
                x_raw2_batch = X_raw2[batch_idx * batch_size : min((batch_idx + 1) * batch_size, num)]
                sf2_batch = sf2[batch_idx * batch_size : min((batch_idx + 1) * batch_size, num)]

                inputs1 = Variable(x1_batch)
                rawinputs1 = Variable(x_raw1_batch)
                sfinputs1 = Variable(sf1_batch)
                inputs2 = Variable(x2_batch)
                rawinputs2 = Variable(x_raw2_batch)
                sfinputs2 = Variable(sf2_batch)

                (zbatch, z_num, lqbatch, mean1_tensor, mean2_tensor, disp1_tensor, 
                 disp2_tensor, pi1_tensor, pi2_tensor, mse_loss1, mse_loss2, nll_loss) = self.forward(inputs1, inputs2)
                recon_loss1 = self.zinb_loss(
                    x=rawinputs1, 
                    mean=mean1_tensor, 
                    disp=disp1_tensor, 
                    pi=pi1_tensor, 
                    scale_factor=sfinputs1
                )
                recon_loss2 = self.zinb_loss(
                    x=rawinputs2, 
                    mean=mean2_tensor, 
                    disp=disp2_tensor, 
                    pi=pi2_tensor, 
                    scale_factor=sfinputs2
                )
                target2 = self.target_distribution(lqbatch)
                lqbatch = lqbatch + torch.diag(torch.diag(z_num))
                target2 = target2 + torch.diag(torch.diag(z_num))

                kl_loss = self.kldloss(target2, lqbatch)
                lclu = kl_loss * self.phi2
                lrec = recon_loss1 + recon_loss2  # Total reconstruction loss based on ZINB distribution
                lmse = mse_loss1 + mse_loss2

                # Calculate Euclidean distance loss leuc
                leuc = self.euclidean_distance_loss(zbatch, inputs1) + self.euclidean_distance_loss(zbatch, inputs2)

                # Total loss = clustering loss + reconstruction loss + MSE loss + Euclidean distance loss + negative log-likelihood expected loss
                loss = lamclu * lclu + lamrec * lrec + lambda_mse * lmse + lameuc * leuc + lamnll * nll_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                recon_loss1_val += recon_loss1.data * len(inputs1)
                recon_loss2_val += recon_loss2.data * len(inputs2)
                kl_loss_val += kl_loss.data * len(inputs1)
                mse_loss_val += lmse.data * len(inputs1)
                euclidean_loss_val += leuc.data * len(inputs1)
                nll_loss_val += nll_loss.data * len(inputs1)
                loss_val += loss.data * len(inputs1)

            if epoch % self.t == 0:
                print("#Epoch %d: Total: %.6f ZINB Loss1: %.6f ZINB Loss2: %.6f KL Loss: %.6f MSE Loss: %.6f Euclidean Loss: %.6f NLL Loss: %.6f" % (
                    epoch + 1, 
                    loss_val / num, 
                    recon_loss1_val / num, 
                    recon_loss2_val / num, 
                    kl_loss_val / num, 
                    mse_loss_val / num, 
                    euclidean_loss_val / num, 
                    nll_loss_val / num
                ))

            torch.cuda.empty_cache()

            Loss_save.append(loss.item())

        return self.y_pred, final_nmi, final_ari, final_epoch