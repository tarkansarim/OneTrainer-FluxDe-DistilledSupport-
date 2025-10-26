from modules.model.FluxModel import FluxModel
from modules.modelSetup.BaseFluxSetup import BaseFluxSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.config.TrainConfig import TrainConfig
from modules.util.flux_block_util import get_block_name_from_parameter
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.torch_util import state_dict_has_prefix
from modules.util.TrainProgress import TrainProgress

import torch


class FluxLoRASetup(
    BaseFluxSetup,
):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super().__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )

    def create_parameters(
            self,
            model: FluxModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()

        self._create_model_part_parameters(parameter_group_collection, "text_encoder_1_lora", model.text_encoder_1_lora, config.text_encoder)
        self._create_model_part_parameters(parameter_group_collection, "text_encoder_2_lora", model.text_encoder_2_lora, config.text_encoder_2)

        if config.train_any_embedding() or config.train_any_output_embedding():
            if config.text_encoder.train_embedding and model.text_encoder_1 is not None:
                self._add_embedding_param_groups(
                    model.all_text_encoder_1_embeddings(), parameter_group_collection, config.embedding_learning_rate,
                    "embeddings_1"
                )

            if config.text_encoder_2.train_embedding and model.text_encoder_2 is not None:
                self._add_embedding_param_groups(
                    model.all_text_encoder_2_embeddings(), parameter_group_collection, config.embedding_learning_rate,
                    "embeddings_2"
                )

        # Transformer: Check if block-wise LRs are enabled
        if config.block_learning_rate_multiplier and config.layer_filter_preset == "blocks":
            self._create_block_wise_parameters(
                parameter_group_collection,
                model.transformer_lora,
                config
            )
        else:
            # Default behavior: single parameter group
            self._create_model_part_parameters(parameter_group_collection, "transformer_lora", model.transformer_lora, config.prior)
        
        return parameter_group_collection
    
    def _create_block_wise_parameters(
            self,
            parameter_group_collection: NamedParameterGroupCollection,
            transformer_lora,
            config: TrainConfig
    ):
        """Create separate parameter groups for each block with individual learning rates"""
        if transformer_lora is None:
            return
        
        base_lr = config.prior.learning_rate if config.prior.learning_rate else config.learning_rate
        block_lr_multipliers = config.block_learning_rate_multiplier
        
        # Group parameters by block
        block_params = {}  # Maps block_name -> list of parameters
        
        # LoRAModuleWrapper stores modules in lora_modules dict
        for name, lora_module in transformer_lora.lora_modules.items():
            block_name = get_block_name_from_parameter(name)
            
            if block_name is None:
                # Parameters that don't belong to a specific block (e.g., norm layers)
                block_name = "other"
            
            if block_name not in block_params:
                block_params[block_name] = []
            
            # Get all parameters from this LoRA module
            block_params[block_name].extend(lora_module.parameters())
        
        # Create parameter groups with individual LRs
        if config.debug_mode:
            print(f"[BlockLR] Using block-wise learning rates with base_lr={base_lr:.8g}")

        for block_name, params in block_params.items():
            multiplier = block_lr_multipliers.get(block_name, 1.0)
            block_lr = base_lr * multiplier
            if config.debug_mode:
                try:
                    param_count = sum(p.numel() for p in params)
                except Exception:
                    param_count = len(list(params))
                print(
                    f"[BlockLR][LoRA] transformer_{block_name}: "
                    f"base_lr={base_lr:.8g}, multiplier={multiplier:.6g}, "
                    f"final_lr={block_lr:.8g}, params={param_count}"
                )
            
            parameter_group_collection.add_group(NamedParameterGroup(
                unique_name=f"transformer_{block_name}",
                parameters=params,
                learning_rate=block_lr
            ))

    def __setup_requires_grad(
            self,
            model: FluxModel,
            config: TrainConfig,
    ):
        self._setup_embeddings_requires_grad(model, config)
        if model.text_encoder_1 is not None:
            model.text_encoder_1.requires_grad_(False)
        if model.text_encoder_2 is not None:
            model.text_encoder_2.requires_grad_(False)
        model.transformer.requires_grad_(False)
        model.vae.requires_grad_(False)

        self._setup_model_part_requires_grad("text_encoder_1_lora", model.text_encoder_1_lora, config.text_encoder, model.train_progress)
        self._setup_model_part_requires_grad("text_encoder_2_lora", model.text_encoder_2_lora, config.text_encoder_2, model.train_progress)
        self._setup_model_part_requires_grad("transformer_lora", model.transformer_lora, config.prior, model.train_progress)

    def setup_model(
            self,
            model: FluxModel,
            config: TrainConfig,
    ):
        create_te1 = config.text_encoder.train or state_dict_has_prefix(model.lora_state_dict, "lora_te1")
        create_te2 = config.text_encoder_2.train or state_dict_has_prefix(model.lora_state_dict, "lora_te2")

        if model.text_encoder_1 is not None:
            model.text_encoder_1_lora = LoRAModuleWrapper(
                model.text_encoder_1, "lora_te1", config
            ) if create_te1 else None

        if model.text_encoder_2 is not None:
            model.text_encoder_2_lora = LoRAModuleWrapper(
                model.text_encoder_2, "lora_te2", config
            ) if create_te2 else None

        model.transformer_lora = LoRAModuleWrapper(
            model.transformer, "lora_transformer", config, config.layer_filter.split(",")
        )

        if model.lora_state_dict:
            if model.text_encoder_1_lora is not None:
                model.text_encoder_1_lora.load_state_dict(model.lora_state_dict)
            if model.text_encoder_2_lora is not None:
                model.text_encoder_2_lora.load_state_dict(model.lora_state_dict)
            model.transformer_lora.load_state_dict(model.lora_state_dict)
            model.lora_state_dict = None

        if model.text_encoder_1_lora is not None:
            model.text_encoder_1_lora.set_dropout(config.dropout_probability)
            model.text_encoder_1_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
            model.text_encoder_1_lora.hook_to_module()

        if model.text_encoder_2_lora is not None:
            model.text_encoder_2_lora.set_dropout(config.dropout_probability)
            model.text_encoder_2_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
            model.text_encoder_2_lora.hook_to_module()

        model.transformer_lora.set_dropout(config.dropout_probability)
        model.transformer_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
        model.transformer_lora.hook_to_module()

        if config.train_any_embedding():
            if model.text_encoder_1 is not None:
                model.text_encoder_1.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())
            if model.text_encoder_2 is not None:
                model.text_encoder_2.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())

        self._remove_added_embeddings_from_tokenizer(model.tokenizer_1)
        self._remove_added_embeddings_from_tokenizer(model.tokenizer_2)
        self._setup_embeddings(model, config)
        self._setup_embedding_wrapper(model, config)
        self.__setup_requires_grad(model, config)

        init_model_parameters(model, self.create_parameters(model, config), self.train_device)

    def setup_train_device(
            self,
            model: FluxModel,
            config: TrainConfig,
    ):
        vae_on_train_device = not config.latent_caching
        text_encoder_1_on_train_device = \
            config.train_text_encoder_or_embedding() \
            or not config.latent_caching

        text_encoder_2_on_train_device = \
            config.train_text_encoder_2_or_embedding() \
            or not config.latent_caching

        model.text_encoder_1_to(self.train_device if text_encoder_1_on_train_device else self.temp_device)
        model.text_encoder_2_to(self.train_device if text_encoder_2_on_train_device else self.temp_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.transformer_to(self.train_device)

        if model.text_encoder_1:
            if config.text_encoder.train:
                model.text_encoder_1.train()
            else:
                model.text_encoder_1.eval()

        if model.text_encoder_2:
            if config.text_encoder_2.train:
                model.text_encoder_2.train()
            else:
                model.text_encoder_2.eval()

        model.vae.eval()

        if config.prior.train:
            model.transformer.train()
        else:
            model.transformer.eval()

    def after_optimizer_step(
            self,
            model: FluxModel,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        if config.preserve_embedding_norm:
            self._normalize_output_embeddings(model.all_text_encoder_2_embeddings())
            if model.embedding_wrapper_1 is not None:
                model.embedding_wrapper_1.normalize_embeddings()
            if model.embedding_wrapper_2 is not None:
                model.embedding_wrapper_2.normalize_embeddings()
        self.__setup_requires_grad(model, config)
