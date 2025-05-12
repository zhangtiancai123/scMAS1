# scMAS.py
from time import time
import argparse
import os
import numpy as np
import torch
import datetime
import h5py
import scanpy as sc

from network import scMAS
from preprocess import load, readData, normalizeInit, tfidf_transform
from utils import bestMap
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics as metrics  # Explicitly import metrics for clarity


def parse_arguments(data_para):
    parser = argparse.ArgumentParser(description="scEMC")
    parser.add_argument("--n_clusters", default=data_para["K"], type=int)
    parser.add_argument("--lr", default=1, type=float)  # 1
    parser.add_argument(
        "-el1", "--encodeLayer1", nargs="+", default=[256, 64, 32, 8]
    )  # [256, 64, 32, 8]
    parser.add_argument(
        "-el2", "--encodeLayer2", nargs="+", default=[256, 64, 32, 8]
    )  # [256, 64, 32, 8]
    parser.add_argument("-dl1", "--decodeLayer1", nargs="+", default=[24, 64, 256])
    parser.add_argument("-dl2", "--decodeLayer2", nargs="+", default=[24, 20])
    parser.add_argument("--dataset", default=data_para)
    parser.add_argument("--view_dims", default=data_para["n_input"])
    parser.add_argument("--name", type=str, default=data_para[1])
    parser.add_argument(
        "--cutoff",
        default=0.5,
        type=float,
        help="Start to train combined layer after what ratio of epoch",
    )
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--maxiter", default=500, type=int)
    parser.add_argument("--pretrain_epochs", default=400, type=int)
    parser.add_argument(
        "--gamma", default=0.1, type=float, help="coefficient of clustering loss"
    )
    parser.add_argument(
        "--tau", default=1.0, type=float, help="fuzziness of clustering loss"
    )
    parser.add_argument(
        "--phi1",
        default=0.001,
        type=float,
        help="pre coefficient of KL loss",  # 0.001
    )
    parser.add_argument(
        "--phi2",
        default=0.001,
        type=float,
        help="coefficient of KL loss",  # 0.001
    )
    parser.add_argument("--update_interval", default=1, type=int)
    parser.add_argument("--tol", default=0.001, type=float)
    parser.add_argument("--ae_weights", default=None)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--ae_weight_file", default=None)
    parser.add_argument("--resolution", default=0.2, type=float)
    parser.add_argument("--n_neighbors", default=30, type=int)
    parser.add_argument("--embedding_file", action="store_true", default=None)
    parser.add_argument("--prediction_file", action="store_true", default=None)
    parser.add_argument("--sigma1", default=2.5, type=float)
    parser.add_argument("--sigma2", default=1.5, type=float)
    parser.add_argument(
        "--f1", default=2000, type=float, help="Number of mRNA after feature selection"
    )
    parser.add_argument(
        "--f2", default=2000, type=float, help="Number of ADT/ATAC after feature selection"
    )
    parser.add_argument("--run", default=1, type=int)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--lamclu", default=1, type=float)
    parser.add_argument("--lamkl", default=1, type=float)
    parser.add_argument("--lammse", default=0.01, type=float)
    parser.add_argument("--lameuc", default=0.00001, type=float)
    parser.add_argument("--lamnll", default=0.00001, type=float)
    return parser.parse_args()


def prepare_data(
    dataset, size_factors=True, normalize_input=True, logtrans_input=True, modality_type="RNA"
):
    """Prepare and preprocess single-cell data for model input."""
    data = sc.AnnData(np.array(dataset))
    data = readData(data, transpose=False, test_split=False, copy=True)
    if modality_type == "ATAC":
        data = tfidf_transform(data)
    data = normalizeInit(data, size_factors=size_factors, normalize_input=normalize_input, logtrans_input=logtrans_input)
    return data


def main():
    # 1. Data loading and preprocessing:
    print("begin 1--dataload")
    my_data_dic = preprocess.DADI
    for i_d in my_data_dic:
        data_para = my_data_dic[i_d]
    args = parse_arguments(data_para)
    X, Y = preprocess.load(args.dataset)
    labels = Y[0].copy().astype(np.int32)
    adata1 = prepare_data(X[0])
    adata2 = prepare_data(
        X[1], size_factors=False, normalize_input=False, logtrans_input=False, modality_type="ATAC"
    )
    y = labels
    input_size1 = adata1.n_vars
    input_size2 = adata2.n_vars
    encodeLayer1 = list(map(int, args.encodeLayer1))
    encodeLayer2 = list(map(int, args.encodeLayer2))
    decodeLayer1 = list(map(int, args.decodeLayer1))
    decodeLayer2 = list(map(int, args.decodeLayer2))

    # Initialize the scMAS model with specified parameters
    model = scMAS(
        input_dim1=input_size1,
        input_dim2=input_size2,
        tau=args.tau,
        encodeLayer1=encodeLayer1,
        encodeLayer2=encodeLayer2,
        decodeLayer1=decodeLayer1,
        decodeLayer2=decodeLayer2,
        activation="elu",
        sigma1=args.sigma1,
        sigma2=args.sigma2,
        gamma=args.gamma,
        cutoff=args.cutoff,
        phi1=args.phi1,
        phi2=args.phi2,
        device=args.device,
    ).to(args.device)
    
    # Create save directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    t0 = time()

    # 4. Pretrain autoencoder stage
    print("begin 4--pretrain_autoencoder")
    if args.ae_weights is None:
        model.pretrain_autoencoder(
            X1=adata1.X,
            X_raw1=adata1.raw.X,
            sf1=adata1.obs.size_factors,
            X2=adata2.X,
            X_raw2=adata2.raw.X,
            sf2=adata2.obs.size_factors,
            batch_size=args.batch_size,
            epochs=args.pretrain_epochs,
            ae_weights=args.ae_weight_file,
        )
    else:
        if os.path.isfile(args.ae_weights):
            print(f"==> loading checkpoint '{args.ae_weights}'")
            checkpoint = torch.load(args.ae_weights)
            model.load_state_dict(checkpoint["ae_state_dict"])
        else:
            print(f"==> no checkpoint found at '{args.ae_weights}'")
            raise ValueError

    # 5. Generate latent representations
    print("begin 5--GetCluster")  
    latent = model.encodeBatch(
        torch.tensor(adata1.X, dtype=torch.float32).to(args.device),
        torch.tensor(adata2.X.todense(), dtype=torch.float32).to(args.device),
    )
    latent = latent.cpu().numpy()

    # 6. Clustering stage: Use model.fit() to obtain predicted labels
    print("begin 6--fit")
    X2_tensor = torch.tensor(adata2.X.todense(), dtype=torch.float32)
    y_pred, _, _, _ = model.fit(
        X1=adata1.X,
        X_raw1=adata1.raw.X,
        sf1=adata1.obs.size_factors,
        X2=X2_tensor,
        X_raw2=adata2.raw.X,
        sf2=adata2.obs.size_factors,
        y=y,
        batch_size=args.batch_size,
        num_epochs=args.maxiter,
        update_interval=args.update_interval,
        tol=args.tol,
        lr=args.lr,
        save_dir=args.save_dir,
        lamclu=args.lamclu,
        lamkl=args.lamkl,
        lammse=args.lammse,
        lameuc=args.lameuc,
        lamnll=args.lamnll,
    )

    print("Initializing clustering with Louvain.")
    
    # Calculate and display total execution time
    total_seconds = int(time() - t0)
    time_delta = datetime.timedelta(seconds=total_seconds)
    hours = time_delta.seconds // 3600
    minutes = (time_delta.seconds % 3600) // 60
    seconds = time_delta.seconds % 60
    print(f"Total time: {hours}h {minutes}min {seconds}s")

    # Best map alignment for cluster labels
    y_pred_ = utils.bestMap(y, y_pred)
    
    # 7. Evaluation: Compute NMI and ARI for clustering performance
    print("begin 7--NMI_ARI")
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 4)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 4)
    print(f"Final: ARI= {ari:.4f}, NMI= {nmi:.4f}")


if __name__ == "__main__":
    main()