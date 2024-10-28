from typing import Dict, Optional, Tuple, Union, List, Any

import numpy as np
import torch
import torch.nn.functional as F
from scvi._constants import REGISTRY_KEYS
from scvi.distributions import ZeroInflatedNegativeBinomial
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import DecoderSCVI, Encoder, one_hot, FCLayers
from torch import Tensor
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from scCausalVI.data.utils import gram_matrix


class scCausalVIModule(BaseModuleClass):
    """
    PyTorch module for scCausalVI (Variational Inference).

    Args:
    ----
        n_input: Number of input genes.
        n_conditions: Number of conditions.
        n_batch: Number of batches. If 0, no batch effect correction is performed.
        n_hidden: Number of nodes per hidden layer.
        n_background_latent: Dimensionality of the background latent space.
        n_salient_latent: Dimensionality of the salient latent space.
        n_layers: Number of hidden layers used for encoder and decoder NNs.
        dropout_rate: Dropout rate for neural networks.
        use_observed_lib_size: Use observed library size for RNA as scaling factor in
            mean of conditional distribution.
        library_log_means: 1 x n_batch array of means of the log library sizes.
            Parameterize prior on library size if not using observed library size.
        library_log_vars: 1 x n_batch array of variances of the log library sizes.
            Parameterize prior on library size if not using observed library size.
        disentangle: Whether to disentangle the salient and background latent variables.
        use_mmd: Whether to use the maximum mean discrepancy to force background latent
            variables of the control and treatment dataset to follow the same
            distribution.
        mmd_weight: Weight of the mmd loss so the mmd loss has similar scale as the
            other loss terms.
        gammas: Gamma values when `use_mmd` is `True`.
    """

    def sample(self, *args, **kwargs):
        pass

    def __init__(
            self,
            n_input: int,
            n_conditions: int,
            control: int,
            n_batch: int = 0,
            n_treat: int = 0,
            n_hidden: int = 128,
            n_background_latent: int = 10,
            n_salient_latent: int = 10,
            n_layers: int = 1,
            dropout_rate: float = 0.1,
            use_observed_lib_size: bool = True,
            library_log_means: Optional[np.ndarray] = None,
            library_log_vars: Optional[np.ndarray] = None,
            disentangle: bool = False,
            use_mmd: bool = True,
            # bg_gan: bool = True,
            mmd_weight: float = 1.0,
            cls_weight: float = 1.0,
            mse_weight: float = 1.0,
            norm_weight: float = 1.0,
            gammas: Optional[np.ndarray] = None,
            # gan_weight: float = 1.0,
    ) -> None:
        super(scCausalVIModule, self).__init__()
        self.n_input = n_input
        self.n_conditions = n_conditions
        self.control = control
        self.treat_ind = [i for i in range(self.n_conditions) if i != self.control]
        self.n_batch = n_batch
        self.n_treat = n_treat
        self.n_hidden = n_hidden
        self.n_background_latent = n_background_latent
        self.n_salient_latent = n_salient_latent
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.latent_distribution = "normal"
        self.dispersion = "gene"
        self.px_r = torch.nn.Parameter(torch.randn(n_input))
        self.use_observed_lib_size = use_observed_lib_size
        self.disentangle = disentangle
        self.use_mmd = use_mmd
        self.mmd_weight = mmd_weight
        self.cls_weight = cls_weight
        self.mse_weight = mse_weight
        self.norm_weight = norm_weight
        self.gammas = gammas
        # self.bg_gan = bg_gan
        # self.gan_weight = gan_weight

        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )
            self.register_buffer(
                "library_log_means", torch.from_numpy(library_log_means).float()
            )
            self.register_buffer(
                "library_log_vars", torch.from_numpy(library_log_vars).float()
            )

        if use_mmd:
            if gammas is None:
                raise ValueError("If using mmd, must provide gammas.")
            # self.register_buffer("gammas", torch.from_numpy(gammas).float())

        # cat_list = [n_batch]
        cat_list = [n_batch]

        # Background encoder encodes cellular intrinsic heterogeneity
        # of control data. The input dim equals to the number of genes (no condition).
        self.control_background_encoder = Encoder(
            n_input,
            n_background_latent,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=self.latent_distribution,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=False,
            var_activation=None,
        )

        # Each treatment background encoder encodes the cellular intrinsic heterogeneity of treatment data.
        self.treatment_background_encoder = torch.nn.ModuleList(
            [
                Encoder(
                    n_input,
                    n_background_latent,
                    n_cat_list=cat_list,
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    distribution=self.latent_distribution,
                    inject_covariates=True,
                    use_batch_norm=True,
                    use_layer_norm=False,
                    var_activation=None,
                )
                for _ in range(self.n_conditions - 1)
            ]
        )

        # Introduce learnable parameter as scaling factor for treatment effect variable
        # self.log_scaling_factor = torch.nn.Parameter(torch.zeros(1))
        # self.scaling_factor = torch.exp(self.log_scaling_factor) + 1
        # self.log_scaling_factor = torch.nn.Parameter(torch.zeros(1))

        self.attention = torch.nn.Linear(self.n_salient_latent, 1)
        # Initialize parameters in self.treatment_background_encoder using parameters in self.control_background_encoder
        for encoder in self.treatment_background_encoder:
            encoder.load_state_dict(self.control_background_encoder.state_dict())

        # Each treatment_salient_encoder generates its corresponding treatment effect.
        # Goes from the n_background_latent to the n_salient_latent.
        self.treatment_salient_encoder = torch.nn.ModuleList(
            [
                Encoder(
                    n_input=n_background_latent,
                    n_output=n_salient_latent,
                    n_cat_list=None,
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    distribution=self.latent_distribution,
                    inject_covariates=False,
                    use_batch_norm=True,
                    use_layer_norm=False,
                    var_activation=None,
                )
                for _ in range(self.n_conditions - 1)

            ]
        )

        # Initialize the last self.n_treat-1 encoders using the parameters in the first encoder in
        # self.treatment_salient_encoder
        if len(self.treatment_salient_encoder) > 0:
            # Get the state dictionary of the first encoder in the list
            first_encoder_state_dict = self.treatment_salient_encoder[0].state_dict()

            # Iterate over all other encoders in the list and initialize them with the first encoder's parameters
            for i in range(1, len(self.treatment_salient_encoder)):
                self.treatment_salient_encoder[i].load_state_dict(first_encoder_state_dict)

        # Library size encoder.
        self.l_encoder = Encoder(
            n_input,
            n_output=1,
            n_layers=1,
            n_cat_list=cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=False,
            var_activation=None,
        )
        # Decoder from latent variable to distribution parameters in data space.
        n_total_latent = n_background_latent + n_salient_latent
        self.decoder = DecoderSCVI(
            n_total_latent,
            n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=True,
        )

        self.classifier = torch.nn.Sequential(
            FCLayers(
                n_in=n_salient_latent,
                n_out=n_hidden,
                n_layers=n_layers,
                n_hidden=n_hidden,
                inject_covariates=False
            ),
            torch.nn.Linear(n_hidden, self.n_treat),
        )

        # # Discriminator for total correlation loss
        # if self.bg_gan:
        #     self.discriminator = torch.nn.Sequential(
        #         FCLayers(
        #             n_in=n_background_latent,
        #             n_out=n_hidden,
        #             n_layers=n_layers,
        #             n_hidden=n_hidden,
        #             inject_covariates=False
        #         ),
        #         torch.nn.Linear(n_hidden, self.n_treat + 1),
        #     )

    @auto_move_data
    def _compute_local_library_params(
            self, batch_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_batch = self.library_log_means.shape[1]
        local_library_log_means = F.linear(
            one_hot(batch_index, n_batch), self.library_log_means
        )
        local_library_log_vars = F.linear(
            one_hot(batch_index, n_batch), self.library_log_vars
        )
        return local_library_log_means, local_library_log_vars

    # @staticmethod
    # def _get_min_batch_size(concat_tensors: Dict[str, Dict[str, torch.Tensor]]) -> int:
    #     return min(
    #         concat_tensors["control"][REGISTRY_KEYS.X_KEY].shape[0],
    #         concat_tensors["treatment"][REGISTRY_KEYS.X_KEY].shape[0],
    #     )

    # @staticmethod
    # def _reduce_tensors_to_min_batch_size(
    #         tensors: Dict[str, torch.Tensor], min_batch_size: int
    # ) -> None:
    #     for name, tensor in tensors.items():
    #         tensors[name] = tensor[:min_batch_size, :]
    #         print(tensors[name].shape[0])

    # @staticmethod
    # def _get_inference_input_from_concat_tensors(
    #         concat_tensors: Dict[str, Dict[str, torch.Tensor]], index: str
    # ) -> Dict[str, torch.Tensor]:
    #     tensors = concat_tensors[index]
    #     x = tensors[REGISTRY_KEYS.X_KEY]
    #     condition_index = tensors[REGISTRY_KEYS.LABELS_KEY]
    #     batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
    #     input_dict = dict(x=x, condition_index=condition_index, batch_index=batch_index)
    #     return input_dict

    def _get_inference_input(self,
                             tensors,
                             **kwargs
                             ) -> Union[Dict[str, list[str]], Dict[str, Any]]:

        # The tensors[0] is control data and tensors[1:] is the treatment data.
        x = [group[REGISTRY_KEYS.X_KEY] for group in tensors]
        batch_index = [group[REGISTRY_KEYS.BATCH_KEY] for group in tensors]
        condition_label = [group[REGISTRY_KEYS.LABELS_KEY] for group in tensors]

        out = dict(x=x, condition_label=condition_label, batch_index=batch_index,)
        return out

    # @staticmethod
    # def _reshape_tensor_for_samples(tensor: torch.Tensor, n_samples: int):
    #     return tensor.unsqueeze(0).expand((n_samples, tensor.size(0), tensor.size(1)))

    @auto_move_data
    def _generic_inference(
            self,
            x: torch.Tensor,
            batch_index: torch.Tensor,
            src: str,
            condition_label: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # src represents the input data are control data or which treatment data.

        if src == 'control':
            x_ = torch.log(x + 1)

            ql_m, ql_v = None, None
            if self.use_observed_lib_size:
                library = torch.log(x.sum(1).unsqueeze(1))
            else:
                ql_m, ql_v, library_encoded = self.l_encoder(x_, batch_index)
                library = library_encoded

            batch_size = x.size()[0]
            qbg_m, qbg_v, z_bg = self.control_background_encoder(x_, batch_index)
            qt_m = torch.zeros(batch_size, 1)
            qt_v = torch.zeros(batch_size, 1)
            z_t = torch.zeros(batch_size, self.n_salient_latent, device=self.device)

        else:
            qbg_m, qbg_v, z_bg = [], [], []
            qt_m, qt_v, z_t = [], [], []
            library, ql_m, ql_v = [], [], []
            for i in range(self.n_treat):
                label_treat = condition_label[i][0]
                x_treat = x[i]
                batch_index_treat = batch_index[i]
                ind = int(label_treat) - 1
                bg_encoder = self.treatment_background_encoder[ind]
                salient_encoder = self.treatment_salient_encoder[ind]

                x_treat_ = torch.log(x_treat + 1)
                ql_m_treat, ql_v_treat = None, None
                if self.use_observed_lib_size:
                    library_treat = torch.log(x_treat.sum(1).unsqueeze(1))
                else:
                    ql_m_treat, ql_v_treat, library_treat = self.l_encoder(x_treat_, batch_index_treat)
                ql_m.append(ql_m_treat)
                ql_v.append(ql_v_treat)
                library.append(library_treat)

                # batch_size = x_treat.size()[0]
                qbg_m_treat, qbg_v_treat, z_bg_treat = bg_encoder(x_treat_, batch_index_treat)
                qt_m_treat, qt_v_treat, z_t_treat = salient_encoder(z_bg_treat)  # input the sampled values

                qbg_m.append(qbg_m_treat)
                qbg_v.append(qbg_v_treat)
                z_bg.append(z_bg_treat)
                qt_m.append(qt_m_treat)
                qt_v.append(qt_v_treat)
                z_t.append(z_t_treat)

        outputs = {'z_bg': z_bg, 'qbg_m': qbg_m, 'qbg_v': qbg_v,
                   'z_t': z_t, 'qt_m': qt_m, 'qt_v': qt_v,
                   'library': library, 'ql_m': ql_m, 'ql_v': ql_v}
        return outputs

    @auto_move_data
    def inference(
            self,
            x,
            condition_label,
            batch_index,
            n_samples: int = 1,
    ) -> Dict[str, Dict[str, torch.Tensor]]:

        # Inference of control data (the first condition data)
        # x always contain all condition data (control + treatment)
        if len(x) != self.n_conditions:
            raise ValueError(f"Number of conditions is {self.n_conditions}. Number of tensors is {len(x)}.")
        x_control = x[0]
        x_treatment = [x_treat for x_treat in x[1:]]

        cond_control = condition_label[0]
        cond_treatment = [cond for cond in condition_label[1:]]

        batch_control = batch_index[0]
        batch_treatment = [batch for batch in batch_index[1:]]

        control_outputs = self._generic_inference(x_control, batch_control, src='control', condition_label=cond_control)
        treatment_outputs = self._generic_inference(x_treatment, batch_treatment, src='treatment', condition_label=cond_treatment)

        return dict(control=control_outputs, treatment=treatment_outputs)

    # @staticmethod
    # def _get_generative_input_from_concat_tensors(
    #         concat_tensors: Dict[str, Dict[str, torch.Tensor]], index: str
    # ) -> Dict[str, torch.Tensor]:
    #     tensors = concat_tensors[index]
    #     batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
    #     input_dict = dict(batch_index=batch_index)
    #     return input_dict

    # @staticmethod
    # def _get_generative_input_from_inference_outputs(
    #         inference_outputs: Dict[str, Dict[str, torch.Tensor]], data_source: str
    # ) -> Dict[str, torch.Tensor]:
    #     bg_z = inference_outputs[data_source]["bg_z"]
    #     # s = inference_outputs[data_source]["s"]
    #     z_t = inference_outputs[data_source]["z_t"]
    #     library = inference_outputs[data_source]["library"]
    #     # return dict(bg_z=bg_z, s=s, library=library)
    #     return dict(bg_z=bg_z, z_t=z_t, library=library)

    def _get_generative_input(self,
                              tensors,
                              inference_outputs,
                              **kwargs,
                              ):
        control = inference_outputs['control']
        treatment = inference_outputs['treatment']
        output = {}
        for key in ['z_bg', 'z_t', 'library']:
            output[key] = torch.cat([control[key]] + treatment[key], dim=0)

        output['batch_index'] = torch.cat([group[REGISTRY_KEYS.BATCH_KEY] for group in tensors], dim=0)

        return output

    # def _get_generative_input(self,
    #         tensors,
    #         inference_outputs,
    # ) -> Dict[str, Dict[str, torch.Tensor]]:
    #     control_tensor_input = self._get_generative_input_from_concat_tensors(
    #         concat_tensors, "control"
    #     )
    #     treatment_tensor_input = self._get_generative_input_from_concat_tensors(
    #         concat_tensors, "treatment"
    #     )
    #     # Ensure batch sizes are the same.
    #     # min_batch_size = self._get_min_batch_size(concat_tensors)
    #     # self._reduce_tensors_to_min_batch_size(control_tensor_input, min_batch_size)
    #     # self._reduce_tensors_to_min_batch_size(treatment_tensor_input, min_batch_size)
    #
    #     control_inference_outputs = (
    #         self._get_generative_input_from_inference_outputs(
    #             inference_outputs, "control"
    #         )
    #     )
    #     treatment_inference_outputs = self._get_generative_input_from_inference_outputs(
    #         inference_outputs, "treatment"
    #     )
    #     control = {**control_tensor_input, **control_inference_outputs}
    #     treatment = {**treatment_tensor_input, **treatment_inference_outputs}
    #     return dict(control=control, treatment=treatment)

    # @auto_move_data
    # def _generic_generative(
    #         self,
    #         z_bg: torch.Tensor,
    #         # s: torch.Tensor,
    #         z_t: torch.Tensor,
    #         library: torch.Tensor,
    #         batch_index: torch.Tensor,
    # ) -> Dict[str, torch.Tensor]:
    #     # latent = torch.cat([bg_z, s], dim=-1)
    #     latent = torch.cat([z_bg, z_t], dim=-1)
    #     px_scale, px_r, px_rate, px_dropout = self.decoder(
    #         self.dispersion,
    #         latent,
    #         library,
    #         batch_index,
    #     )
    #     px_r = torch.exp(self.px_r)
    #     return dict(
    #         px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout
    #     )

    @auto_move_data
    def generative(
            self,
            z_bg: torch.Tensor,
            z_t: torch.Tensor,
            library: torch.Tensor,
            batch_index: List[int],
    ) -> Dict[str, Dict[str, torch.Tensor]]:

        # z_t = z_t * torch.exp(self.log_scaling_factor)+1
        attention_weights = torch.softmax(self.attention(z_t), dim=-1)
        z_t = attention_weights * z_t
        latent = torch.cat([z_bg, z_t], dim=-1)
        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion,
            latent,
            library,
            batch_index,
        )
        px_r = torch.exp(self.px_r)
        return dict(
            px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout
        )

        # latent_z_shape = control["bg_z"].shape
        # batch_size_dim = 0 if len(latent_z_shape) == 2 else 1
        # control_batch_size = control["bg_z"].shape[batch_size_dim]
        # treatment_batch_size = treatment["bg_z"].shape[batch_size_dim]
        # generative_input = {}
        # for key in ["bg_z", "z_t", "library"]:
        #     generative_input[key] = torch.cat(
        #         [control[key], treatment[key]], dim=batch_size_dim
        #     )
        # generative_input["batch_index"] = torch.cat(
        #     [control["batch_index"], treatment["batch_index"]], dim=0
        # )
        # outputs = self._generic_generative(**generative_input)
        # control_outputs, treatment_outputs = {}, {}
        # for key in ["px_scale", "px_rate", "px_dropout"]:
        #     if outputs[key] is not None:
        #         control_tensor, treatment_tensor = torch.split(
        #             outputs[key],
        #             [control_batch_size, treatment_batch_size],
        #             dim=batch_size_dim,
        #         )
        #     else:
        #         control_tensor, treatment_tensor = None, None
        #     control_outputs[key] = control_tensor
        #     treatment_outputs[key] = treatment_tensor
        # control_outputs["px_r"] = outputs["px_r"]
        # treatment_outputs["px_r"] = outputs["px_r"]
        # return dict(control=control_outputs, treatment=treatment_outputs)

    @staticmethod
    def reconstruction_loss(
            x: torch.Tensor,
            px_rate: torch.Tensor,
            px_r: torch.Tensor,
            px_dropout: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute likelihood loss for zero-inflated negative binomial distribution.

        Args:
        ----
            x: Input data.
            px_rate: Mean of distribution.
            px_r: Inverse dispersion.
            px_dropout: Logits scale of zero inflation probability.

        Returns
        -------
            Negative log likelihood (reconstruction loss) for each data point. If number
            of latent samples == 1, the tensor has shape `(batch_size, )`. If number
            of latent samples > 1, the tensor has shape `(n_samples, batch_size)`.
        """
        if x.shape[0] != px_rate.shape[0]:
            print(f"x.shape[0]= {x.shape[0]} and px_rate.shape[0]= {px_rate.shape[0]}.")
        recon_loss = (
            -ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)
            .log_prob(x)
            .sum(dim=-1)
        )
        return recon_loss

    @staticmethod
    def latent_kl_divergence(
            variational_mean: torch.Tensor,
            variational_var: torch.Tensor,
            prior_mean: torch.Tensor,
            prior_var: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence between a variational posterior and prior Gaussian.
        Args:
        ----
            variational_mean: Mean of the variational posterior Gaussian.
            variational_var: Variance of the variational posterior Gaussian.
            prior_mean: Mean of the prior Gaussian.
            prior_var: Variance of the prior Gaussian.

        Returns
        -------
            KL divergence for each data point. If number of latent samples == 1,
            the tensor has shape `(batch_size, )`. If number of latent
            samples > 1, the tensor has shape `(n_samples, batch_size)`.
        """
        return kl(
            Normal(variational_mean, variational_var.sqrt()),
            Normal(prior_mean, prior_var.sqrt()),
        ).sum(dim=-1)

    def library_kl_divergence(
            self,
            batch_index: torch.Tensor,
            variational_library_mean: torch.Tensor,
            variational_library_var: torch.Tensor,
            library: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence between library size variational posterior and prior.

        Both the variational posterior and prior are Log-Normal.
        Args:
        ----
            batch_index: Batch indices for batch-specific library size mean and
                variance.
            variational_library_mean: Mean of variational Log-Normal.
            variational_library_var: Variance of variational Log-Normal.
            library: Sampled library size.

        Returns
        -------
            KL divergence for each data point. If number of latent samples == 1,
            the tensor has shape `(batch_size, )`. If number of latent
            samples > 1, the tensor has shape `(n_samples, batch_size)`.
        """
        if not self.use_observed_lib_size:
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)

            kl_library = kl(
                Normal(variational_library_mean, variational_library_var.sqrt()),
                Normal(local_library_log_means, local_library_log_vars.sqrt()),
            )
        else:
            kl_library = torch.zeros_like(library)
        return kl_library.sum(dim=-1)

    def mmd_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cost = torch.mean(gram_matrix(x, x, gammas=self.gammas.to(self.device)))
        cost += torch.mean(gram_matrix(y, y, gammas=self.gammas.to(self.device)))
        cost -= 2 * torch.mean(gram_matrix(x, y, gammas=self.gammas.to(self.device)))
        if cost < 0:  # Handle numerical instability.
            return torch.tensor(0)
        return cost

    def _generic_loss(
            self,
            tensors: Union[torch.Tensor, List[torch.Tensor]],
            inference_outputs: Union[Dict, List[Dict]],
            generative_outputs: Union[Dict, List[Dict]],
            src: str,
    ) -> dict[str, Union[Tensor, list[Tensor]]]:

        if src == 'control':
            x = tensors[REGISTRY_KEYS.X_KEY]
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

            qbg_m = inference_outputs['qbg_m']
            qbg_v = inference_outputs['qbg_v']
            library = inference_outputs['library']
            ql_m = inference_outputs["ql_m"]
            ql_v = inference_outputs["ql_v"]
            px_rate = generative_outputs["px_rate"]
            px_r = generative_outputs["px_r"]
            px_dropout = generative_outputs["px_dropout"]

            prior_bg_m = torch.zeros_like(qbg_m)
            prior_bg_v = torch.ones_like(qbg_v)

            if x.shape[0] != px_rate.shape[0]:
                print(f"x.shape[0]= {x.shape[0]} and px_rate.shape[0]= {px_rate.shape[0]}.")
            recon_loss = self.reconstruction_loss(x, px_rate, px_r, px_dropout)
            kl_bg = self.latent_kl_divergence(qbg_m, qbg_v, prior_bg_m, prior_bg_v)
            kl_library = self.library_kl_divergence(batch_index, ql_m, ql_v, library)

        else:
            recon_loss, kl_bg, kl_library = [], [], []
            for tensor, infer_out, gene_out in zip(tensors, inference_outputs, generative_outputs):
                x = tensor[REGISTRY_KEYS.X_KEY]
                batch_index = tensor[REGISTRY_KEYS.BATCH_KEY]

                qbg_m = infer_out['qbg_m']
                qbg_v = infer_out['qbg_v']
                library = infer_out['library']
                ql_m = infer_out["ql_m"]
                ql_v = infer_out["ql_v"]
                px_rate = gene_out["px_rate"]
                px_r = gene_out["px_r"]
                px_dropout = gene_out["px_dropout"]

                prior_bg_m = torch.zeros_like(qbg_m)
                prior_bg_v = torch.ones_like(qbg_v)

                if x.shape[0] != px_rate.shape[0]:
                    print(f"x.shape[0]= {x.shape[0]} and px_rate.shape[0]= {px_rate.shape[0]}.")
                recon_loss.append(self.reconstruction_loss(x, px_rate, px_r, px_dropout))
                kl_bg.append(self.latent_kl_divergence(qbg_m, qbg_v, prior_bg_m, prior_bg_v))
                kl_library.append(self.library_kl_divergence(batch_index, ql_m, ql_v, library))

        return dict(
            recon_loss=recon_loss,
            kl_bg=kl_bg,
            kl_library=kl_library,
        )

    def graph_loss(self,
                   tensor: torch.Tensor,
                   z_bg: torch.Tensor,
                   z_t: torch.Tensor,
                   ) -> torch.Tensor:
        """
        Add graph alignment to constrain treatment effect by network subtraction.
        Parameters
        ----------
            tensor: n_batch by n_feature scRNA-seq count data
            z_bg: background latent variable representing cell type heterogeneities
            z_t: latent variable representing treatment effect

        Returns:
            Norm of difference of distance matrix,
        """
        tensor_ = torch.log(tensor + 1)
        D = torch.cdist(tensor_, tensor_, p=2)
        D_bg = torch.cdist(z_bg, z_bg, p=2)
        D_t = torch.cdist(z_t, z_t, p=2)
        # Align the scale of distance matrix by Z-score normalization
        D_mean = torch.mean(D)
        D_std = torch.std(D)
        D_normalized = (D - D_mean) / (D_std + 0.00001)
        D_bg_mean = torch.mean(D_bg)
        D_bg_std = torch.std(D_bg)
        D_bg_normalized = (D_bg - D_bg_mean) / (D_bg_std + 0.00001)
        D_t_mean = torch.mean(D_t)
        D_t_std = torch.std(D_t)
        D_t_normalized = (D_t - D_t_mean) / (D_t_std + 0.00001)

        loss = torch.norm(torch.abs(D_normalized - D_bg_normalized) - D_t_normalized)
        return loss

    def loss(
            self,
            tensors: List[torch.Tensor],
            inference_outputs: Dict[str, Dict[str, torch.Tensor]],
            generative_outputs: Dict[str, torch.Tensor],
            **loss_args,
    ) -> LossRecorder:
        """
        Compute loss terms for scCausalVI.
        Args:
        ----
            concat_tensors: Tuple of data mini-batch. The first element contains
                control data mini-batch. The second element contains treatment data
                mini-batch.
            inference_outputs: Dictionary of inference step outputs. The keys
                are "control" and "treatment" for the corresponding outputs.
            generative_outputs: Dictionary of generative step outputs. The keys
                are "control" and "treatment" for the corresponding outputs.
            kl_weight: Importance weight for KL divergence of background and salient
                latent variables, relative to KL divergence of library size.

        Returns
        -------
            An scvi.module.base.LossRecorder instance that records the following:
            loss: One-dimensional tensor for overall loss used for optimization.
            reconstruction_loss: Reconstruction loss with shape
                `(n_samples, batch_size)` if number of latent samples > 1, or
                `(batch_size, )` if number of latent samples == 1.
            kl_local: KL divergence term with shape
                `(n_samples, batch_size)` if number of latent samples > 1, or
                `(batch_size, )` if number of latent samples == 1.
            kl_global: One-dimensional tensor for global KL divergence term.
        """
        # The input tensors[0] is the control dataset and the tensors[1:] are the treatment datasets
        tensor_control = tensors[0]
        tensor_treatment = [tensor for tensor in tensors[1:]]
        condition_label = [group[REGISTRY_KEYS.LABELS_KEY] for group in tensors]
        treat_ind_tensors = [int(label[0]) for label in condition_label]
        treat_ind_tensors = treat_ind_tensors[1:]
        condition_label = torch.cat(condition_label, dim=0)
        n_treatment = self.n_conditions - 1
        control_indices = (condition_label == self.control).squeeze(dim=-1)

        treatment_indices = [(condition_label == i).squeeze(dim=-1) for i in treat_ind_tensors]

        inference_outputs_control = inference_outputs['control']

        inference_outputs_treatment = []
        for i in range(n_treatment):
            inference_outputs_treatment.append(
                {key: inference_outputs['treatment'][key][i] for key in inference_outputs['treatment'].keys()}
            )

        generative_outputs_control = {key: value[control_indices] for key, value in generative_outputs.items() if
                                      key != 'px_r'}
        generative_outputs_treatment = []
        for i in range(self.n_conditions - 1):
            generative_outputs_treatment.append(
                {key: value[treatment_indices[i]] for key, value in generative_outputs.items() if key != "px_r"}
            )

        generative_outputs_control['px_r'] = generative_outputs['px_r']
        for i in range(n_treatment):
            generative_outputs_treatment[i]['px_r'] = generative_outputs['px_r']

        if tensor_control[REGISTRY_KEYS.X_KEY].shape[0] != generative_outputs_control['px_rate'].shape[0]:
            print(
                f"x.shape[0]= {tensor_control.shape[0]} and px_rate.shape[0]= {generative_outputs_control['px_rate'].shape[0]}.")

        elbo_loss_control = self._generic_loss(tensor_control, inference_outputs_control,
                                               generative_outputs_control, 'control')
        elbo_loss_treatment = self._generic_loss(tensor_treatment, inference_outputs_treatment,
                                                 generative_outputs_treatment, 'treatment')

        # Compute MMD loss of background latent variables between control data and treatment data
        if self.use_mmd:
            loss_mmd = 0
            for z_bg_treat in [infer_treat['z_bg'] for infer_treat in inference_outputs_treatment]:
                loss_mmd += self.mmd_loss(x=inference_outputs_control['z_bg'],
                                          y=z_bg_treat)
        else:
            loss_mmd = 0

        loss_mmd = self.mmd_weight * loss_mmd

        # # Inappropriate training process! The train process is like GAN updating two network iteratively.
        # if self.bg_gan:
        #     gan_input = torch.cat([inference_outputs_control['z_bg']]+[infer_treat['z_bg'] for infer_treat in
        #                                                                inference_outputs_treatment], dim=0)
        #     gan_output = self.discriminator(gan_input)
        #     loss_gan = F.cross_entropy(gan_output, condition_label.squeeze().long())
        # else:
        #     loss_gan = 0
        # loss_gan = self.gan_weight * loss_gan

        # Use classifier to classify different treatment effects,
        # force salient_encoders to separate treatment effects. This loss only helps to train salient_encoder.
        z_bg_detach = [infer_treat['z_bg'].detach() for infer_treat in inference_outputs_treatment]
        qt_m_clf, qt_v_clf, z_t_clf = [], [], []
        for z_bg_treat, salient_encoder in zip(z_bg_detach, self.treatment_salient_encoder):
            qt_m_treat, qt_v_treat, z_t_treat = salient_encoder(z_bg_treat)
            qt_m_clf.append(qt_m_treat)
            qt_v_clf.append(qt_m_treat)
            z_t_clf.append(z_t_treat)

        pred_label = self.classifier(
            torch.cat(qt_m_clf, dim=0))  # use mean instead of sample values to predict condition labels
        # Normalize the true labels to be [0, n_treatment-1] for F.cross_entropy
        treat_ind_sorted = sorted(self.treat_ind)
        label_mapping = {label: idx for idx, label in enumerate(treat_ind_sorted)}
        true_label = [condition_label[treat_index] for treat_index in treatment_indices]
        normalized_label = torch.tensor(
            [label_mapping[label] for labels in true_label for label in labels.squeeze().tolist()],
            device=self.device)

        loss_clf = 0.
        if self.cls_weight > 0:
            loss_clf = F.cross_entropy(pred_label, normalized_label.long(), reduction='mean')
        loss_clf = self.cls_weight * loss_clf

        # Add MSE loss
        loss_mse = 0.
        if self.mse_weight > 0:
            loss_mse += F.mse_loss(tensor_control[REGISTRY_KEYS.X_KEY], generative_outputs_control['px_rate'],
                                   reduction='mean')
            for i in range(self.n_treat):
                loss_mse += F.mse_loss(tensor_treatment[i][REGISTRY_KEYS.X_KEY],
                                       generative_outputs_treatment[i]['px_rate'],
                                       reduction='mean')
        loss_mse = self.mse_weight * loss_mse

        loss_norm = torch.tensor(0)
        if self.norm_weight > 0:
            loss_norm = []
            for i in range(self.n_treat):
                # loss_norm += torch.norm(inference_outputs_treatment[i]['z_t'], p=2)
                loss_norm_treat = inference_outputs_treatment[i]['z_t'] ** 2
                loss_norm.append(loss_norm_treat.sum(dim=-1))
            loss_norm = torch.cat(loss_norm, dim=0)

            # prior_bg_m = torch.zeros_like(inference_outputs_treatment[i]['qt_m'])
            # prior_bg_v = torch.ones_like(inference_outputs_treatment[i]['qt_v'])
            # loss_norm += torch.sum(self.latent_kl_divergence(inference_outputs_treatment[i]['qt_m'],
            #                                        inference_outputs_treatment[i]['qt_v'],
            #                                        prior_bg_m, prior_bg_v))
            # loss_norm += self.graph_loss(tensor_treatment[i][REGISTRY_KEYS.X_KEY],
            #                              inference_outputs_treatment[i]['z_bg'],
            #                              inference_outputs_treatment[i]['z_t'])

            # logvar = torch.log(inference_outputs_treatment[i]['qt_v'])
            # loss_norm += 0.5 * torch.sum(1 + logvar)
            # print(logvar)

        loss_norm = self.norm_weight * loss_norm

        # loss_clf = 0
        # for i in range(self.n_conditions - 1):
        #     for j in range(self.n_conditions - 1):
        #         if j <= i:
        #             continue
        #         else:
        #             loss_clf += self.mmd_loss(qt_m_clf[i], qt_m_clf[j])
        #
        # loss_clf = -self.cls_weight * loss_clf  # use mmd loss to distinguish treatment effects

        recon_loss = torch.cat([elbo_loss_control['recon_loss']] + elbo_loss_treatment['recon_loss'],
                               dim=0)
        loss_kl_l = torch.cat([elbo_loss_control['kl_library']] + elbo_loss_treatment['kl_library'], dim=0)
        loss_kl_bg = torch.cat([elbo_loss_control['kl_bg']] + elbo_loss_treatment['kl_bg'], dim=0)

        loss = (
                torch.mean(recon_loss) +
                torch.mean(loss_kl_bg) +
                torch.mean(loss_kl_l) +
                torch.mean(loss_norm) +
                loss_mmd + loss_clf + loss_mse
        )

        kl_local = dict(loss_kl_bg=loss_kl_bg,
                        loss_kl_l=loss_kl_l,
                        # loss_norm=loss_norm,
                        loss_mmd=loss_mmd,
                        loss_clf=loss_clf,
                        loss_mse=loss_mse)
        kl_global = torch.tensor(0.0)
        return LossRecorder(loss, recon_loss, kl_local, kl_global)

        # recon_loss = control_losses["recon_loss"] + treatment_losses["recon_loss"]
        # # kl_z = control_losses["kl_z"] + treatment_losses["kl_z"]
        # kl_bg = control_losses['kl_bg'] + treatment_losses['kl_bg']
        # # kl_s = treatment_losses["kl_s"]
        # kl_library = control_losses["kl_library"] + treatment_losses["kl_library"]
        #
        # # loss = (
        # #         torch.sum(recon_loss)
        # #         + torch.sum(kl_z)
        # #         + torch.sum(kl_s)
        # #         + torch.sum(kl_library)
        # # )
        #
        # # COMMENT! Check use torch.mean or sum?
        # loss = (
        #         torch.sum(recon_loss)
        #         + torch.sum(kl_bg)
        #         + torch.sum(kl_library)
        # )
        #
        # if self.disentangle:
        #     # z_tar = inference_outputs["treatment"]["qz_m"]
        #     # s_tar = inference_outputs["treatment"]["qs_m"]
        #     z_tar = inference_outputs["treatment"]["qbg_m"]
        #     # s_tar = inference_outputs["treatment"]["qs_m"]
        #
        #     # If more than one sample, the outputs have dimension
        #     # (n_samples, batch_size, n_latent). Otherwise, the outputs have dimension
        #     # (batch_size, n_latent). We want to make sure that the first dimension
        #     # corresponds to the batch size for total correlation estimation.
        #     if len(z_tar.shape) == 3:
        #         z_tar = z_tar.permute(1, 0, 2)
        #         # s_tar = s_tar.permute(1, 0, 2)
        #     z1, z2 = torch.chunk(z_tar, 2)
        #     # s1, s2 = torch.chunk(s_tar, 2)
        #
        #     # Make sure all tensors have same number of batch samples. This is
        #     # necessary e.g. if we have an odd batch size at the end of an epoch.
        #     size = min(len(z1), len(z2))
        #     # z1, z2, s1, s2 = z1[:size], z2[:size], s1[:size], s2[:size]
        #     z1, z2 = z1[:size], z2[:size]
        #
        #     # q = torch.cat([torch.cat([z1, s1], dim=-1), torch.cat([z2, s2], dim=-1)])
        #     q = torch.cat([z1, z2])
        #     q_bar = torch.cat(
        #         # [torch.cat([z1, s2], dim=-1), torch.cat([z2, s1], dim=-1)]
        #         [z1, z2]
        #     )
        #     q_bar_score = F.sigmoid(self.discriminator(q_bar))
        #     q_score = F.sigmoid(self.discriminator(q))
        #     tc_loss = torch.log(q_score / (1 - q_score))
        #     discriminator_loss = -torch.log(q_score) - torch.log(1 - q_bar_score)
        #     loss += torch.sum(tc_loss) + torch.sum(discriminator_loss)
        #
        # if self.use_mmd:
        #     z_treat = inference_outputs["treatment"]["bg_z"]
        #     z_control = inference_outputs["control"]["bg_z"]
        #     mmd = self.mmd_loss(z_treat, z_control)
        #     loss += self.mmd_weight * torch.sum(mmd)
        #
        # kl_local = dict(
        #     # kl_z=kl_z,
        #     # kl_s=kl_s,
        #     kl_bg=kl_bg,
        #     kl_library=kl_library,
        # )
        # kl_global = torch.tensor(0.0)
        # return LossRecorder(loss, recon_loss, kl_local, kl_global)
