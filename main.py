import os
import scanpy as sc
import scvi
import numpy as np
import torch
from scCausalVAE.model.scCausalVAE import scCausalVAEModel

scvi.settings.seed = 0

file_path = "/Users/sa3520/Unfinished/CausalInferrence_Drug/Mycode10/testdata/"
adata = sc.read_h5ad(os.path.join(file_path, 'haber_2017.h5ad'))
n_conditions = len(adata.obs['condition'].unique())

# preprocess data
adata.raw = adata
sc.pp.filter_cells(adata, min_genes=100)
sc.pp.filter_genes(adata, min_cells=5)
# The harber_2017 data has been normalized and log_1p transformed, so we don't need to do it again
# sc.pp.normalize_total(adata, target_sum=1e6)
# sc.pp.log1p(adata)
sc.pp.highly_variable_genes(
    adata,
    flavor="seurat_v3",
    n_top_genes=1000,
    layer="count",
    batch_key="condition",
    subset=True,
)

group_key = "condition"
control_key = 'Control'
conditions = ['Control', 'Hpoly.Day10', 'Salmonella']  # Make sure the first label is the control group!!!
group_indices_list = [np.where(adata.obs[group_key] == group)[0] for group in conditions]

scCausalVAEModel.setup_anndata(adata, layer="count", labels_key=group_key,)

condition2int = adata.obs.groupby(group_key, observed=False)['_scvi_labels'].first().to_dict()
control = condition2int[control_key]

# TO DO: add the following code to the scCausalVAEModel class
n_conditions = len(conditions)
model = scCausalVAEModel(
    adata,
    n_conditions=n_conditions,
    control=control,
    n_layers=2,
    n_treat=2,
    n_background_latent=10,
    n_salient_latent=10,
    cls_weight=0.,
    mse_weight=0.0,
    mmd_weight=1,
    norm_weight=0.01,
)

use_gpu = torch.cuda.is_available()
model.train(
    group_indices_list,
    use_gpu=use_gpu,
    max_epochs=40,
)


latent_bg, latent_t = model.get_latent_representation()
expression = model.get_normalized_expression()

adata.obsm['latent_bg'] = latent_bg
adata.obsm['latent_t'] = latent_t

sc.pp.neighbors(adata, use_rep='latent_bg')
sc.tl.umap(adata)

sc.pl.umap(adata, color=['condition', 'cell_type'], save='all_condition_celltype.png')
# sc.pl.umap(adata, color=['condition', 'cell_type'], )

indices_tm = [index for sublist in group_indices_list[1:] for index in sublist]
adata_tm = adata[indices_tm, :]

sc.pp.neighbors(adata_tm, use_rep='latent_t')
sc.tl.umap(adata_tm)

sc.pl.umap(adata_tm, color=['condition', 'cell_type'], save='treatment_condition_celltype.png')
print('Done!')
