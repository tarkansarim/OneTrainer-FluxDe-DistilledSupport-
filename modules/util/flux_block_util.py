def get_flux_block_counts(transformer):
    """
    Detect the number of double and single blocks in a Flux transformer.
    Returns: (num_double_blocks, num_single_blocks)
    
    Flux Dev/Schnell typically have 19 double blocks and 38 single blocks.
    """
    double_blocks = 0
    single_blocks = 0
    
    # Try to get from config first
    if hasattr(transformer, 'config'):
        config = transformer.config
        if hasattr(config, 'num_layers'):
            double_blocks = config.num_layers
        if hasattr(config, 'num_single_layers'):
            single_blocks = config.num_single_layers
    
    # Fallback: count blocks by inspecting the model structure
    if double_blocks == 0 and hasattr(transformer, 'transformer_blocks'):
        double_blocks = len(transformer.transformer_blocks)
    
    if single_blocks == 0 and hasattr(transformer, 'single_transformer_blocks'):
        single_blocks = len(transformer.single_transformer_blocks)
    
    return double_blocks, single_blocks


def get_block_name_from_parameter(param_name: str) -> str | None:
    """
    Parse block name from parameter name.
    
    Examples:
        "transformer_blocks.0.attn.to_q.lora_down.weight" -> "double_block_0"
        "single_transformer_blocks.15.linear1.lora_up.weight" -> "single_block_15"
        "some_other_param" -> None
    
    Args:
        param_name: The full parameter name
        
    Returns:
        Block name string (e.g., "double_block_0") or None if not a block parameter
    """
    # Check single_transformer_blocks first since it contains "transformer_blocks"
    if "single_transformer_blocks." in param_name:
        try:
            block_idx = int(param_name.split("single_transformer_blocks.")[1].split(".")[0])
            return f"single_block_{block_idx}"
        except (IndexError, ValueError):
            return None
    elif "transformer_blocks." in param_name:
        try:
            block_idx = int(param_name.split("transformer_blocks.")[1].split(".")[0])
            return f"double_block_{block_idx}"
        except (IndexError, ValueError):
            return None
    
    return None

