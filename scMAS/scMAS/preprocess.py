from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle, os, numbers
import numpy as np
import scipy as sp
import pandas as pd
import scanpy as sc
import torch
import h5py
import warnings
import scipy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale,LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
import scipy.sparse as sp1

warnings.filterwarnings("ignore")


DADI = dict(
    # BMNC10K 数据集配置
    BMNC10K={
        # 数据集标识
        1: 'PBMC10K',  # 数据集名称
        2: 'd1',       # 数据集版本
        
        # 统计参数
        'N': 9631,      # 样本总数
        'K': 27,        # 类别数量
        'V': 2,         # 特征变量数
        
        # 网络结构参数
        'n_input': [29095, 107194],  # 输入层维度（双模态输入）
        'n_hid': [10, 256],          # 隐藏层维度列表
        'n_output': 64               # 输出层维度
    },
)

# 修改数据集路径
scRNA_path = './datasets/Pbmc10k-ATAC.h5ad'
scATAC_path = './datasets/Pbmc10k-ATAC.h5ad'

def load(dataset_name):
    adata_rna = sc.read_h5ad(scRNA_path)
    adata_atac = sc.read_h5ad(scATAC_path)

    assert np.all(adata_rna.obs_names == adata_atac.obs_names), "cell order is false"

    X_rna = adata_rna.X  #
    X_atac = adata_atac.X  

    mm = MinMaxScaler()
    #deal with scRNA data
    if scipy.sparse.issparse(X_rna):
        X_rna = X_rna.toarray()
    X_rna_std = mm.fit_transform(X_rna)
    #deal with scATAC data
    if scipy.sparse.issparse(X_atac):
        X_atac = X_atac.toarray()
    X_atac_std = mm.fit_transform(X_atac)
    
    labels = adata_rna.obs['cell_type'].values
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    X_rna_tensor = torch.from_numpy(X_rna_std.astype(np.float32))
    X_atac_tensor = torch.from_numpy(X_atac_std.astype(np.float32))
    return [X_rna_tensor, X_atac_tensor], [labels_encoded, labels_encoded]




def tfidf_transform(adata):
    if sp1.issparse(adata.X):
        X_dense = adata.X.toarray() 
    else:
        X_dense = adata.X
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_dense)

    if not sp1.issparse(X_tfidf):
        X_tfidf = sp1.csr_matrix(X_tfidf)
    X_tfidf_normalized = normalize(X_tfidf, norm='l2', axis=1)

    if isinstance(X_tfidf_normalized, sp1.csr_matrix):
        try:
            adata.X = X_tfidf_normalized.tocsc()
        except ValueError:
            adata.X = X_tfidf_normalized.toarray()
    else:
        adata.X = X_tfidf_normalized
    print("TF-IDF transformation completed.")
    return adata


class AnnSequence:
    def __init__(self, matrix, batch_size, sf=None):
        self.matrix = matrix
        if sf is None:
            self.size_factors = np.ones((self.matrix.shape[0], 1),
                                        dtype=np.float32)
        else:
            self.size_factors = sf
        self.batch_size = batch_size

    def __len__(self):
        return len(self.matrix) // self.batch_size

    def __getitem__(self, idx):
        batch = self.matrix[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_sf = self.size_factors[idx*self.batch_size:(idx+1)*self.batch_size]

        # return an (X, Y) pair
        return {'count': batch, 'size_factors': batch_sf}, batch





def readDdata(adata: sc.AnnData | str, transpose: bool = False, test_split: bool = False, copy: bool = False) -> sc.AnnData:
    """
    Load and preprocess single-cell RNA-seq data for autoencoder analysis.
    
    Args:
        adata: AnnData object or path to h5ad file
        transpose: If True, transpose the dataset (cells x genes -> genes x cells)
        test_split: If True, split data into train/test subsets
        copy: If True, create a copy of the AnnData object
        
    Returns:
        Processed AnnData object with 'MAS_split' column in obs
    """
    # Load data if path is provided
    adata = _load_anndata(adata, copy)
    
    # Validate count data
    _validate_counts(adata)
    
    # Transpose if required
    if transpose:
        adata = adata.transpose()
    
    # Add train/test split annotation
    _add_split_annotation(adata, test_split)
    
    # Log processing results
    print(f'### Autoencoder: Successfully preprocessed {adata.n_vars} genes and {adata.n_obs} cells.')
    
    return adata

def _load_anndata(adata: sc.AnnData | str, copy: bool) -> sc.AnnData:
    """Load AnnData from path or create copy if requested."""
    if isinstance(adata, str):
        return sc.read(adata)
    elif isinstance(adata, sc.AnnData):
        return adata.copy() if copy else adata
    else:
        raise NotImplementedError("Input must be AnnData or path to .h5ad file")

def _validate_counts(adata: sc.AnnData) -> None:
    """Validate that AnnData contains unnormalized count data."""
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error
    
    # Check if data is integer counts (only for smaller datasets)
    if adata.X.size < 50e6:
        if sp.sparse.issparse(adata.X):
            non_integer_entries = (adata.X.astype(int) != adata.X).nnz
            assert non_integer_entries == 0, norm_error

def _add_split_annotation(adata: sc.AnnData, test_split: bool) -> None:
    """Add train/test split annotation to adata.obs."""
    if test_split:
        train_idx, test_idx = train_test_split(
            np.arange(adata.n_obs), 
            test_size=0.1, 
            random_state=42
        )
        split_labels = pd.Series(['train'] * adata.n_obs)
        split_labels.iloc[test_idx] = 'test'
        adata.obs['MAS_split'] = split_labels.values
    else:
        adata.obs['MAS_split'] = 'train'
    
    # Convert to categorical for efficiency
    adata.obs['MAS_split'] = adata.obs['MAS_split'].astype('category')


def normalizeInit(adata, filter_min_counts=False, size_factors=True, normalize_input=True, logtrans_input=True):
    if size_factors or normalize_input or logtrans_input:  
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:  
        sc.pp.log1p(adata)
    if normalize_input: 
        sc.pp.scale(adata)

    return adata




def calculate_qc_metrics(adata, modality_type):
    if modality_type == "RNA":
        adata.var['mito'] = adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mito'], percent_top=None, inplace=True)
        adata.obs.rename(columns={'pct_counts_mito': 'pct_mito'}, inplace=True)
        
    elif modality_type == "ATAC":
        sc.pp.calculate_qc_metrics(adata, percent_top=None, inplace=True)
        adata.obs.rename(columns={'n_genes_by_counts': 'n_features'}, inplace=True)
    
    print("QC metrics calculated.")
    return adata

def perform_qc_filtering(adata, modality_type, 
                        min_genes=1, max_genes=5000, 
                        min_counts=1, max_counts=20000,
                        max_pct_mito=20, min_cells=1):
    """
    根据QC指标过滤细胞和基因
    """
    print(f"Performing QC filtering for {modality_type} data...")

    if modality_type == "RNA":
        adata = adata[adata.obs['pct_mito'] < max_pct_mito, :]
        
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_cells(adata, max_genes=max_genes)
    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_cells(adata, max_counts=max_counts)
    
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    print(f"After QC filtering: {adata.n_obs} cells, {adata.n_vars} features remaining.")
    return adata

def select_features(adata, modality_type, 
                   min_features=500, max_features=5000,
                   target_sparsity=0.9, n_top_genes=2000):
    """
    根据稀疏度自动调整特征选择
    """
    print(f"Selecting features for {modality_type} data...")
    
    total_nonzero = adata.X.nnz if scipy.sparse.issparse(adata.X) else np.count_nonzero(adata.X)
    total_elements = adata.n_obs * adata.n_vars
    sparsity = 1 - (total_nonzero / total_elements)

    if modality_type == "RNA":
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat')
        adata = adata[:, adata.var['highly_variable']]
        
    elif modality_type == "ATAC":
        dynamic_features = int(min_features + (max_features - min_features) * (sparsity / target_sparsity))
        dynamic_features = max(min_features, min(max_features, dynamic_features))
        
        feature_counts = np.array(adata.X.sum(axis=0)).flatten()
        top_features = np.argsort(feature_counts)[-dynamic_features:]
        adata = adata[:, top_features]
    
    print(f"Selected {adata.n_vars} features based on sparsity {sparsity:.2f}.")
    return adata
