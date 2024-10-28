import logging
from typing import List, Optional, Sequence, Union, Dict

import numpy as np
import torch
from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)
from scvi.dataloaders import AnnDataLoader
from scvi.model._utils import _init_library_size
from scvi.model.base import BaseModelClass
from scvi.distributions import ZeroInflatedNegativeBinomial

from scCausalVI.model.base.training_mixin import scCausalVITrainingMixin
from scCausalVI.module.scCausalVI import scCausalVIModule

logger = logging.getLogger(__name__)


class scCausalVIModel(scCausalVITrainingMixin, BaseModelClass):
    """
    Model class for scCausalVAE.
    Args:
        adata: AnnData object with `.raw` attribute containing count data.
        n_salient_latent: Dimensionality of the salient latent space.
        n_background_latent: Dimensionality of the background latent space.
        use_observed_lib_size: Whether to use the observed library size as a covariate.
        mmd_weight: Weight for the MMD loss.
        use_mmd: Whether to use the MMD loss.
        gammas: Weights for the background loss.
    """

    def __init__(
            self,
            adata: AnnData,
            control: int,
            n_conditions: int,
            n_batch: int = 0,
            n_treat: int = 0,
            n_hidden: int = 128,
            n_background_latent: int = 10,
            n_salient_latent: int = 10,
            n_layers: int = 2,
            dropout_rate: float = 0.1,
            use_observed_lib_size: bool = True,
            disentangle: bool = False,
            bg_gan: bool = True,
            use_mmd: bool = True,
            mmd_weight: float = 1.0,
            cls_weight: float = 1.0,
            mse_weight: float = 1.0,
            norm_weight: float = 1.0,
            gammas: Optional[np.ndarray] = None,
            gan_weight: float = 1.0,
    ) -> None:

        super(scCausalVIModel, self).__init__(adata)
        # self.summary_stats from BaseModelClass gives info about anndata dimensions
        # and other tensor info.
        n_batch = self.summary_stats.n_batch
        if use_observed_lib_size:
            library_log_means, library_log_vars = None, None
        else:
            library_log_means, library_log_vars = _init_library_size(self.adata_manager, n_batch)

        if use_mmd:
            if gammas is None:
                gammas = torch.FloatTensor([10 ** x for x in range(-6, 7, 1)])

        self.module = scCausalVIModule(
            n_input=self.summary_stats["n_vars"],
            n_conditions=n_conditions,
            control=control,
            n_batch=n_batch,
            n_treat=n_treat,
            n_hidden=n_hidden,
            n_background_latent=n_background_latent,
            n_salient_latent=n_salient_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            use_observed_lib_size=use_observed_lib_size,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            disentangle=disentangle,
            # bg_gan=bg_gan,
            use_mmd=use_mmd,
            mmd_weight=mmd_weight,
            cls_weight=cls_weight,
            mse_weight=mse_weight,
            norm_weight=norm_weight,
            gammas=gammas,
            # gan_weight=gan_weight,
        )
        self._model_summary_string = "scCausalVAE."
        # Necessary line to get params to be used for saving and loading.
        self.init_params_ = self._get_init_params(locals())
        logger.info("The model has been initialized.")


    @classmethod
    # @setup_anndata_dsp.dedent
    def setup_anndata(
            cls,
            adata: AnnData,
            layer: Optional[str] = None,
            batch_key: Optional[str] = None,
            labels_key: Optional[str] = None,
            size_factor_key: Optional[str] = None,
            categorical_covariate_keys: Optional[List[str]] = None,
            continuous_covariate_keys: Optional[List[str]] = None,
            **kwargs,
    ):
        """
        Set up AnnData instance for scCausalVAE model

        Args:
            adata: AnnData object with `.raw` attribute containing count data.
            layer: Key for `.layers` or `.raw` where counts are stored.
            batch_key: Key for batch information in `adata.obs`.
            labels_key: Key for label information in `adata.obs`.
            size_factor_key: Key for size factor information in `adata.obs`.
            categorical_covariate_keys: Keys for categorical covariates in `adata.obs`.
            continuous_covariate_keys: Keys for continuous covariates in `adata.obs`.
        """

        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False
            ),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.no_grad()
    def get_latent_representation(
            self,
            adata: Optional[AnnData] = None,
            indices: Optional[Sequence[int]] = None,
            give_mean: bool = True,
            batch_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the background or salient latent representation for each cell.

        Args:
        ----
        adata: AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the AnnData object
            used to initialize the model.

        indices: Indices of cells in adata to use. If `None`, all cells are used.

        give_mean: Give mean of distribution or sample from it.

        batch_size: Mini-batch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
            A numpy array with shape `(n_cells, n_latent)`.
        """

        adata = self._validate_anndata(adata)
        data_loader = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            data_loader_class=AnnDataLoader,
        )

        latent_bg = []
        latent_t = []

        for tensors in data_loader:
            x = tensors[REGISTRY_KEYS.X_KEY]
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
            label_index = tensors[REGISTRY_KEYS.LABELS_KEY]
            unique_labels = label_index.unique()

            latent_bg_tensor = torch.zeros([x.shape[0], self.module.n_background_latent])
            latent_t_tensor = torch.zeros([x.shape[0], self.module.n_salient_latent])

            for label in unique_labels:
                mask = (label_index == label).squeeze().cpu()
                x_label = x[mask]
                batch_index_label = batch_index[mask]
                condition_label_ = label_index[mask]

                if label.item() == self.module.control:  # assuming label 0 corresponds to 'control'
                    src = 'control'
                    outputs = self.module._generic_inference(
                        x=x_label,
                        batch_index=batch_index_label,
                        src=src,
                        condition_label=condition_label_
                    )
                    z_bg_label = outputs['z_bg']
                    latent_bg_tensor[mask] = outputs['qbg_m'].detach().cpu() if give_mean else z_bg_label.detach().cpu()

                else:
                    src = 'treatment'
                    x_label_ = torch.log(x_label + 1)

                    # Specify the treatment encoder for treatment data.
                    # use (label - 1)-th encoder for treatment label since treatment encoder are labelled from 0
                    # while treatment labels are labelled from 1 (label 0 implies control data)
                    treat_index = int(label - 1)
                    bg_encoder = self.module.treatment_background_encoder[treat_index]
                    salient_encoder = self.module.treatment_salient_encoder[treat_index]
                    qbg_m_label, qbg_v_label, z_bg_label = bg_encoder(x_label_, batch_index_label)
                    qt_m_label, qt_v_label, z_t_label = salient_encoder(z_bg_label)
                    # z_t_label = z_t_label*self.module.scaling_factor
                    attention_weights = torch.softmax(self.module.attention(z_t_label), dim=-1)
                    z_t_label = attention_weights * z_t_label
                    latent_bg_tensor[mask] = qbg_m_label.detach().cpu() if give_mean else z_bg_label.detach().cpu()
                    latent_t_tensor[mask] = qt_m_label.detach().cpu() if give_mean else z_t_label.detach().cpu()

            latent_bg.append(latent_bg_tensor)
            latent_t.append(latent_t_tensor)

        latent_bg = torch.cat(latent_bg, dim=0).numpy()
        latent_t = torch.cat(latent_t, dim=0).numpy()

        return latent_bg, latent_t

    @torch.no_grad()
    def get_latent_representation_counterfactual(
            self,
            condition2int: Dict,
            source_condition: str,
            target_condition: str,
            adata: Optional[AnnData] = None,
            indices: Optional[Sequence[int]] = None,
            give_mean: bool = True,
            batch_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the background or salient latent representation for each cell.

        Args:
        ----
        adata: AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the AnnData object
            used to initialize the model.

        indices: Indices of cells in adata to use. If `None`, all cells are used.

        give_mean: Give mean of distribution or sample from it.

        batch_size: Mini-batch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
            A numpy array with shape `(n_cells, n_latent)`.
        """

        if source_condition == target_condition:
            raise ValueError(f"source condition and target condition should be different.")

        adata = self._validate_anndata(adata)
        data_loader = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            data_loader_class=AnnDataLoader,
        )

        latent_bg = []
        latent_t = []

        for tensors in data_loader:
            x = tensors[REGISTRY_KEYS.X_KEY]
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
            label_index = tensors[REGISTRY_KEYS.LABELS_KEY]
            unique_labels = label_index.unique()

            latent_bg_tensor = torch.zeros([x.shape[0], self.module.n_background_latent])
            latent_t_tensor = torch.zeros([x.shape[0], self.module.n_salient_latent])

            for label in unique_labels:
                mask = (label_index == label).squeeze().cpu()
                x_label = x[mask]
                batch_index_label = batch_index[mask]
                condition_label_sub = label_index[mask]

                if label.item() == self.module.control:  # assuming label 0 corresponds to 'control'
                    src = 'control'
                    outputs = self.module._generic_inference(
                        x=x_label,
                        batch_index=batch_index_label,
                        src=src,
                        condition_label=condition_label_sub
                    )
                    z_bg_label = outputs['z_bg']
                    latent_bg_tensor[mask] = outputs['qbg_m'].detach().cpu() if give_mean else z_bg_label.detach().cpu()

                    if condition2int[source_condition] == self.module.control:
                        salient_encoder = self.module.treatment_salient_encoder[condition2int[target_condition] - 1]
                        qt_m_label, qt_v_label, z_t_label = salient_encoder(z_bg_label)
                        attention_weights = torch.softmax(self.module.attention(z_t_label), dim=-1)
                        z_t_label = attention_weights * z_t_label
                        latent_t_tensor[mask] = z_t_label

                else:
                    src = 'treatment'
                    x_label_ = torch.log(x_label + 1)

                    # Specify the treatment encoder for treatment data.
                    # use (label - 1)-th encoder for treatment label since treatment encoder are labelled from 0
                    # while treatment labels are labelled from 1 (label 0 implies control data)
                    treat_index = int(label - 1)
                    bg_encoder = self.module.treatment_background_encoder[treat_index]
                    qbg_m_label, qbg_v_label, z_bg_label = bg_encoder(x_label_, batch_index_label)
                    latent_bg_tensor[mask] = qbg_m_label.detach().cpu() if give_mean else z_bg_label.detach().cpu()

                    if condition2int[source_condition] == label.item():
                        # If target condition is control, then treatment effect embedding is zero
                        # If not, then input the background embedding to corresponding treatment effect network of
                        # target condition and generate counterfactual latent treatment effect embeddings
                        if condition2int[target_condition] != self.module.control:
                            salient_encoder = self.module.treatment_salient_encoder[condition2int[target_condition] - 1]
                            qt_m_label, qt_v_label, z_t_label = salient_encoder(z_bg_label)
                            attention_weights = torch.softmax(self.module.attention(z_t_label), dim=-1)
                            z_t_label = attention_weights * z_t_label
                            latent_t_tensor[mask] = qt_m_label.detach().cpu() if give_mean else z_t_label.detach().cpu()
                        else:
                            pass
                    else:
                        treat_index = int(label - 1)
                        salient_encoder = self.module.treatment_salient_encoder[treat_index]
                        qt_m_label, qt_v_label, z_t_label = salient_encoder(z_bg_label)
                        # z_t_label = z_t_label*self.module.scaling_factor
                        attention_weights = torch.softmax(self.module.attention(z_t_label), dim=-1)
                        z_t_label = attention_weights * z_t_label
                        latent_t_tensor[mask] = qt_m_label.detach().cpu() if give_mean else z_t_label.detach().cpu()

            latent_bg.append(latent_bg_tensor)
            latent_t.append(latent_t_tensor)

        latent_bg = torch.cat(latent_bg, dim=0).numpy()
        latent_t = torch.cat(latent_t, dim=0).numpy()

        return latent_bg, latent_t

    @torch.no_grad()
    def get_normalized_expression(self,
                                  adata: Optional[AnnData] = None,
                                  indices: Optional[Sequence[int]] = None,
                                  batch_size: Optional[int] = None,
                                  library_size: Union[str, float] = 'latent',
                                  ):
        adata = self._validate_anndata(adata)
        data_loader = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            data_loader_class=AnnDataLoader
        )

        exprs = []
        for tensors in data_loader:
            x = tensors[REGISTRY_KEYS.X_KEY]
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
            label_index = tensors[REGISTRY_KEYS.LABELS_KEY]
            unique_labels = label_index.unique()

            latent_bg_tensor = torch.zeros([x.shape[0], self.module.n_background_latent], device=self.module.device)
            latent_t_tensor = torch.zeros([x.shape[0], self.module.n_salient_latent], device=self.module.device)
            latent_library_tensor = torch.zeros([x.shape[0], 1], device=self.module.device)

            # Compute expression for each condition (label)
            # Ensure the output expression match the input cells by mask operation
            for label in unique_labels:
                mask = (label_index == label).squeeze().cpu()
                x_label = x[mask]
                batch_index_label = batch_index[mask]
                condition_label_sub = label_index[mask]

                if label.item() == self.module.control:
                    src = 'control'
                    infer_out = self.module._generic_inference(
                        x=x_label,
                        batch_index=batch_index_label,
                        src=src,
                        condition_label=condition_label_sub,
                    )
                    latent_bg_tensor[mask] = infer_out['z_bg']
                    latent_library_tensor[mask] = infer_out['library']

                else:
                    src = 'treatment'
                    # The computation is the same as self.module._generic_inference function, but refactor it
                    # for single treatment data instead of the list of all treatment data.
                    x_label_ = torch.log(x_label + 1)
                    treat_index = int(label - 1)
                    bg_encoder = self.module.treatment_background_encoder[treat_index]
                    salient_encoder = self.module.treatment_salient_encoder[treat_index]
                    qbg_m_label, qbg_v_label, z_bg_label = bg_encoder(x_label_, batch_index_label)
                    qt_m_label, qt_v_label, z_t_label = salient_encoder(z_bg_label)
                    attention_weights = torch.softmax(self.module.attention(z_t_label), dim=-1)
                    z_t_label = attention_weights * z_t_label
                    # z_t_label = z_t_label*(torch.exp(self.module.log_scaling_factor)+1)

                    # Compute latent variable for library size
                    if self.module.use_observed_lib_size:
                        library_label = torch.log(x_label.sum(1).unsqueeze(1))
                    else:
                        ql_m_label, ql_v_label, library_label = self.module.l_encoder(x_label_, batch_index_label)

                    latent_bg_tensor[mask] = z_bg_label
                    latent_t_tensor[mask] = z_t_label
                    latent_library_tensor[mask] = library_label

            # Generate expression data
            latent_tensor = torch.cat([latent_bg_tensor, latent_t_tensor], dim=-1)
            px_scale_tensor, px_r_tensor, px_rate_tensor, px_dropout_tensor = self.module.decoder(
                self.module.dispersion,
                latent_tensor,
                latent_library_tensor,
                batch_index
            )

            if library_size == 'latent':
                exprs.append(
                    px_rate_tensor.detach().cpu())  # return expression level data (gene expression values level)
            else:
                exprs.append(px_scale_tensor.detach().cpu())  # return scale data (proportion level of gene expression)

        expression = torch.cat(exprs, dim=0).numpy()
        return expression

    @torch.no_grad()
    def get_count_expression(self,
                             adata: Optional[AnnData] = None,
                             indices: Optional[Sequence[int]] = None,
                             batch_size: Optional[int] = None,
                             ):
        adata = self._validate_anndata(adata)
        data_loader = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            data_loader_class=AnnDataLoader
        )

        exprs = []
        for tensors in data_loader:
            x = tensors[REGISTRY_KEYS.X_KEY]
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
            label_index = tensors[REGISTRY_KEYS.LABELS_KEY]
            unique_labels = label_index.unique()

            latent_bg_tensor = torch.zeros([x.shape[0], self.module.n_background_latent], device=self.module.device)
            latent_t_tensor = torch.zeros([x.shape[0], self.module.n_salient_latent], device=self.module.device)
            latent_library_tensor = torch.zeros([x.shape[0], 1], device=self.module.device)

            # Compute expression for each condition (label)
            # Ensure the output expression match the input cells by mask operation
            for label in unique_labels:
                mask = (label_index == label).squeeze().cpu()
                x_label = x[mask]
                batch_index_label = batch_index[mask]
                condition_label_sub = label_index[mask]

                if label.item() == self.module.control:
                    src = 'control'
                    infer_out = self.module._generic_inference(
                        x=x_label,
                        batch_index=batch_index_label,
                        src=src,
                        condition_label=condition_label_sub,
                    )
                    latent_bg_tensor[mask] = infer_out['z_bg']
                    latent_library_tensor[mask] = infer_out['library']

                else:
                    src = 'treatment'
                    # The computation is the same as self.module._generic_inference function, but refactor it
                    # for single treatment data instead of the list of all treatment data.
                    x_label_ = torch.log(x_label + 1)
                    treat_index = int(label - 1)
                    bg_encoder = self.module.treatment_background_encoder[treat_index]
                    salient_encoder = self.module.treatment_salient_encoder[treat_index]
                    qbg_m_label, qbg_v_label, z_bg_label = bg_encoder(x_label_, batch_index_label)
                    qt_m_label, qt_v_label, z_t_label = salient_encoder(z_bg_label)
                    attention_weights = torch.softmax(self.module.attention(z_t_label), dim=-1)
                    z_t_label = attention_weights * z_t_label
                    # z_t_label = z_t_label*(torch.exp(self.module.log_scaling_factor)+1)

                    # Compute latent variable for library size
                    if self.module.use_observed_lib_size:
                        library_label = torch.log(x_label.sum(1).unsqueeze(1))
                    else:
                        ql_m_label, ql_v_label, library_label = self.module.l_encoder(x_label_, batch_index_label)

                    latent_bg_tensor[mask] = z_bg_label
                    latent_t_tensor[mask] = z_t_label
                    latent_library_tensor[mask] = library_label

            # Generate expression data
            latent_tensor = torch.cat([latent_bg_tensor, latent_t_tensor], dim=-1)
            px_scale_tensor, px_r_tensor, px_rate_tensor, px_dropout_tensor = self.module.decoder(
                self.module.dispersion,
                latent_tensor,
                latent_library_tensor,
                batch_index
            )
            if px_r_tensor is None:
                px_r_tensor = torch.exp(self.module.px_r)

            count_tensor = ZeroInflatedNegativeBinomial(
                mu=px_rate_tensor,
                theta=px_r_tensor,
                zi_logits=px_dropout_tensor
            ).sample()
            exprs.append(count_tensor.detach().cpu())

        expression = torch.cat(exprs, dim=0).numpy()
        return expression

    @torch.no_grad()
    def get_normalized_expression_counterfactual(self,
                                                 condition2int: Dict,
                                                 source_condition: str,
                                                 target_condition: str,
                                                 adata: Optional[AnnData] = None,
                                                 indices: Optional[Sequence[int]] = None,
                                                 batch_size: Optional[int] = None,
                                                 library_size: Union[str, float] = 'latent',
                                                 ):
        if source_condition == target_condition:
            raise ValueError(f"source condition and target condition should be different.")

        adata = self._validate_anndata(adata)
        data_loader = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            data_loader_class=AnnDataLoader
        )

        exprs = []
        for tensors in data_loader:
            x = tensors[REGISTRY_KEYS.X_KEY]
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
            label_index = tensors[REGISTRY_KEYS.LABELS_KEY]
            unique_labels = label_index.unique()

            latent_bg_tensor = torch.zeros([x.shape[0], self.module.n_background_latent], device=self.module.device)
            latent_t_tensor = torch.zeros([x.shape[0], self.module.n_salient_latent], device=self.module.device)
            latent_library_tensor = torch.zeros([x.shape[0], 1], device=self.module.device)

            # Compute expression for each condition (label)
            # Ensure the output expression match the input cells by mask operation
            for label in unique_labels:
                mask = (label_index == label).squeeze().cpu()
                x_label = x[mask]
                batch_index_label = batch_index[mask]
                condition_label_sub = label_index[mask]

                if label.item() == self.module.control:
                    src = 'control'
                    infer_out = self.module._generic_inference(
                        x=x_label,
                        batch_index=batch_index_label,
                        src=src,
                        condition_label=condition_label_sub,
                    )
                    # latent_bg_tensor[mask] = infer_out['z_bg']
                    latent_bg_tensor[mask] = infer_out['qbg_m']
                    latent_library_tensor[mask] = infer_out['library']

                    # Check whether to predict counterfactual expression data for control data
                    # if so, input the latent background embedding to corresponding treatment effect network of
                    # target condition and generate counterfactual latent treatment effect embeddings
                    if condition2int[source_condition] == self.module.control:
                        salient_encoder = self.module.treatment_salient_encoder[condition2int[target_condition] - 1]
                        qt_m_label, qt_v_label, z_t_label = salient_encoder(infer_out['z_bg'])
                        # qt_m_label, qt_v_label, z_t_label = salient_encoder(infer_out['qbg_m'])
                        attention_weights = torch.softmax(self.module.attention(z_t_label), dim=-1)
                        # attention_weights = torch.softmax(self.module.attention(qt_m_label), dim=-1)
                        z_t_label = attention_weights * z_t_label
                        # z_t_label = attention_weights * qt_m_label
                        latent_t_tensor[mask] = z_t_label

                else:
                    src = 'treatment'
                    # The computation is the same as self.module._generic_inference function, but refactor it
                    # for single treatment data instead of the list of all treatment data.
                    x_label_ = torch.log(x_label + 1)
                    treat_index = int(label - 1)
                    bg_encoder = self.module.treatment_background_encoder[treat_index]
                    qbg_m_label, qbg_v_label, z_bg_label = bg_encoder(x_label_, batch_index_label)
                    latent_bg_tensor[mask] = z_bg_label
                    # latent_bg_tensor[mask] = qbg_m_label

                    if condition2int[source_condition] == label.item():
                        # If target condition is control, then treatment effect embedding is zero
                        # If not, then input the background embedding to corresponding treatment effect network of
                        # target condition and generate counterfactual latent treatment effect embeddings
                        if condition2int[target_condition] != self.module.control:
                            salient_encoder = self.module.treatment_salient_encoder[condition2int[target_condition] - 1]
                            qt_m_label, qt_v_label, z_t_label = salient_encoder(z_bg_label)
                            # qt_m_label, qt_v_label, z_t_label = salient_encoder(qbg_m_label)
                            attention_weights = torch.softmax(self.module.attention(z_t_label), dim=-1)
                            # attention_weights = torch.softmax(self.module.attention(qt_m_label), dim=-1)
                            z_t_label = attention_weights * z_t_label
                            # z_t_label = attention_weights * qt_m_label
                            latent_t_tensor[mask] = z_t_label
                        else:
                            # if target condition is control, then resulting treatment effects are all 0,
                            # no need to change corresponding latent_t_tensor
                            pass
                    else:
                        treat_index = int(label - 1)
                        salient_encoder = self.module.treatment_salient_encoder[treat_index]
                        qt_m_label, qt_v_label, z_t_label = salient_encoder(z_bg_label)
                        # qt_m_label, qt_v_label, z_t_label = salient_encoder(qbg_m_label)
                        attention_weights = torch.softmax(self.module.attention(z_t_label), dim=-1)
                        # attention_weights = torch.softmax(self.module.attention(qbg_m_label), dim=-1)
                        z_t_label = attention_weights * z_t_label
                        # z_t_label = attention_weights * qbg_m_label
                        latent_t_tensor[mask] = z_t_label

                    # Compute latent variable for library size
                    if self.module.use_observed_lib_size:
                        library_label = torch.log(x_label.sum(1).unsqueeze(1))
                    else:
                        ql_m_label, ql_v_label, library_label = self.module.l_encoder(x_label_, batch_index_label)

                    latent_library_tensor[mask] = library_label

            # Generate expression data
            latent_tensor = torch.cat([latent_bg_tensor, latent_t_tensor], dim=-1)
            px_scale_tensor, px_r_tensor, px_rate_tensor, px_dropout_tensor = self.module.decoder(
                self.module.dispersion,
                latent_tensor,
                latent_library_tensor,
                batch_index
            )

            if library_size == 'latent':
                exprs.append(
                    px_rate_tensor.detach().cpu())  # return expression level data (gene expression values level)
            else:
                exprs.append(px_scale_tensor.detach().cpu())  # return scale data (proportion level of gene expression)

        expression = torch.cat(exprs, dim=0).numpy()
        return expression

    @torch.no_grad()
    def get_count_expression_counterfactual(self,
                                            condition2int,
                                            source_condition,
                                            target_condition,
                                            adata=None,
                                            indices=None,
                                            batch_size=None,
                                            ):
        if source_condition == target_condition:
            raise ValueError(f"source condition and target condition should be different.")

        adata = self._validate_anndata(adata)
        data_loader = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            data_loader_class=AnnDataLoader
        )

        exprs = []
        latent = []
        px_rate = []
        px_dropout = []
        for tensors in data_loader:
            x = tensors["X"]
            batch_index = tensors["batch"]
            label_index = tensors["labels"]
            unique_labels = label_index.unique()

            latent_bg_tensor = torch.zeros([x.shape[0], self.module.n_background_latent], device=self.module.device)
            latent_t_tensor = torch.zeros([x.shape[0], self.module.n_salient_latent], device=self.module.device)
            latent_library_tensor = torch.zeros([x.shape[0], 1], device=self.module.device)

            # Compute expression for each condition (label)
            # Ensure the output expression match the input cells by mask operation
            for label in unique_labels:
                mask = (label_index == label).squeeze().cpu()
                x_label = x[mask]
                batch_index_label = batch_index[mask]
                condition_label_sub = label_index[mask]

                if label.item() == self.module.control:
                    src = 'control'
                    infer_out = self.module._generic_inference(
                        x=x_label,
                        batch_index=batch_index_label,
                        src=src,
                        condition_label=condition_label_sub,
                    )
                    # latent_bg_tensor[mask] = infer_out['z_bg']
                    latent_bg_tensor[mask] = infer_out['z_bg']
                    latent_library_tensor[mask] = infer_out['library']

                    # Check whether to predict counterfactual expression data for control data
                    # if so, input the latent background embedding to corresponding treatment effect network of
                    # target condition and generate counterfactual latent treatment effect embeddings
                    if condition2int[source_condition] == self.module.control:
                        salient_encoder = self.module.treatment_salient_encoder[condition2int[target_condition] - 1]
                        # print(f"from {source_condition} to {target_condition}")
                        # print(condition2int[target_condition] - 1)
                        qt_m_label, qt_v_label, z_t_label = salient_encoder(infer_out['z_bg'])
                        # qt_m_label, qt_v_label, z_t_label = salient_encoder(infer_out['qbg_m'])
                        attention_weights = torch.softmax(self.module.attention(z_t_label), dim=-1)
                        # tt1 = z_t_label.detach().cpu()
                        # print(np.abs(tt1).max())
                        # attention_weights = torch.softmax(self.module.attention(qt_m_label), dim=-1)
                        z_t_label = attention_weights * z_t_label
                        # z_t_label = attention_weights * qt_m_label
                        latent_t_tensor[mask] = z_t_label

                else:
                    src = 'treatment'
                    # The computation is the same as self.module._generic_inference function, but refactor it
                    # for single treatment data instead of the list of all treatment data.
                    x_label_ = torch.log(x_label + 1)
                    treat_index = int(label - 1)
                    bg_encoder = self.module.treatment_background_encoder[treat_index]
                    qbg_m_label, qbg_v_label, z_bg_label = bg_encoder(x_label_, batch_index_label)
                    latent_bg_tensor[mask] = z_bg_label
                    # latent_bg_tensor[mask] = qbg_m_label

                    if condition2int[source_condition] == label.item():
                        # If target condition is control, then treatment effect embedding is zero
                        # If not, then input the background embedding to corresponding treatment effect network of
                        # target condition and generate counterfactual latent treatment effect embeddings
                        if condition2int[target_condition] != self.module.control:
                            # print(f"from {source_condition} to {target_condition}")
                            salient_encoder = self.module.treatment_salient_encoder[condition2int[target_condition] - 1]
                            qt_m_label, qt_v_label, z_t_label = salient_encoder(z_bg_label)
                            # qt_m_label, qt_v_label, z_t_label = salient_encoder(qbg_m_label)
                            attention_weights = torch.softmax(self.module.attention(z_t_label), dim=-1)
                            # attention_weights = torch.softmax(self.module.attention(qt_m_label), dim=-1)
                            z_t_label = attention_weights * z_t_label
                            # z_t_label = attention_weights * qt_m_label
                            latent_t_tensor[mask] = z_t_label
                        else:
                            # if target condition is control, then resulting treatment effects are all 0,
                            # no need to change corresponding latent_t_tensor
                            pass
                    else:
                        treat_index = int(label - 1)
                        salient_encoder = self.module.treatment_salient_encoder[treat_index]
                        qt_m_label, qt_v_label, z_t_label = salient_encoder(z_bg_label)
                        # qt_m_label, qt_v_label, z_t_label = salient_encoder(qbg_m_label)
                        attention_weights = torch.softmax(self.module.attention(z_t_label), dim=-1)
                        # attention_weights = torch.softmax(self.module.attention(qbg_m_label), dim=-1)
                        z_t_label = attention_weights * z_t_label
                        # z_t_label = attention_weights * qbg_m_label
                        latent_t_tensor[mask] = z_t_label

                    # Compute latent variable for library size
                    if self.module.use_observed_lib_size:
                        library_label = torch.log(x_label.sum(1).unsqueeze(1))
                    else:
                        ql_m_label, ql_v_label, library_label = self.module.l_encoder(x_label_, batch_index_label)

                    latent_library_tensor[mask] = library_label

            # Generate expression data
            latent_tensor = torch.cat([latent_bg_tensor, latent_t_tensor], dim=-1)
            px_scale_tensor, px_r_tensor, px_rate_tensor, px_dropout_tensor = self.module.decoder(
                self.module.dispersion,
                latent_tensor,
                latent_library_tensor,
                batch_index
            )
            if px_r_tensor is None:
                px_r_tensor = torch.exp(self.module.px_r)

            torch.manual_seed(0)
            count_tensor = ZeroInflatedNegativeBinomial(
                mu=px_rate_tensor,
                theta=px_r_tensor,
                zi_logits=px_dropout_tensor
            ).sample()

            exprs.append(count_tensor.detach().cpu())
            latent.append(latent_tensor.detach().cpu())
            px_rate.append(px_rate_tensor.detach().cpu())
            px_dropout.append(px_dropout_tensor.detach().cpu())

        expression = torch.cat(exprs, dim=0).numpy()
        latent_bg_t = torch.cat(latent, dim=0).numpy()
        px_rate = torch.cat(px_rate, dim=0).numpy()
        px_r = px_r_tensor.detach().cpu().numpy()
        px_dropout = torch.cat(px_dropout, dim=0).numpy()

        return expression, latent_bg_t, px_rate, px_r, px_dropout



