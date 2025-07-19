import logging
import torch.nn as nn

def freeze_module(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = False

def apply_freeze(model: nn.Module, cfg):
    if getattr(cfg.MODEL.FREEZE, 'BACKBONE', False):
        for attr in ['global_branch', 'local_branch', 'global_extractor', 'local_extractor', 'local_patch_conv']:
            if hasattr(model, attr) and getattr(model, attr) is not None:
                freeze_module(getattr(model, attr))
    if getattr(cfg.MODEL.FREEZE, 'PROJECTION', False):
        for attr in ['global_proj', 'local_proj', 'raw_proj', 'patch_proj']:
            if hasattr(model, attr) and getattr(model, attr) is not None:
                freeze_module(getattr(model, attr))
    if getattr(cfg.MODEL.FREEZE, 'CLASSIFIER', False):
        if hasattr(model, 'classifier') and model.classifier is not None:
            freeze_module(model.classifier)

def log_freeze_status(model: nn.Module, logger: logging.Logger, name: str = ""):
    logger.info(f"🔍 [Freeze Status] {name}")
    if model is None:
        logger.info(f"❌ {name} is None (not used)")
        return
    for param_name, param in model.named_parameters():
        status = "🔒 FROZEN" if not param.requires_grad else "✅ TRAINABLE"
        logger.info(f"{status:12} | {name}.{param_name}")

def log_model_parameters(model: nn.Module, logger: logging.Logger):
    logger.info("📊 [Model Parameters Statistics]")
    logger.info("=" * 60)
    total_params = 0
    trainable_params = 0
    for name, module in model.named_modules():
        if hasattr(module, 'weight') or hasattr(module, 'bias'):
            module_params = sum(p.numel() for p in module.parameters())
            module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if module_params > 0:
                total_params += module_params
                trainable_params += module_trainable
                logger.info(f"{name:30} | Params: {module_params:>8,} | Trainable: {module_trainable:>8,}")
    logger.info("-" * 60)
    logger.info(f"{'TOTAL':30} | Params: {total_params:>8,} | Trainable: {trainable_params:>8,}")
    logger.info(f"Frozen params: {total_params - trainable_params:,}")
    logger.info("=" * 60)

def log_model_configuration(cfg, logger: logging.Logger):
    logger.info("=" * 60)
    logger.info("🏗️  [Model Configuration]")
    logger.info("=" * 60)
    raw_type = cfg.MODEL.EXTRA.RAW
    patch_type = cfg.MODEL.EXTRA.PATCH
    logger.info(f"📋 RAW (Global) Model: {raw_type}")
    logger.info(f"📋 PATCH (Local) Model: {patch_type}")
    global_active = raw_type is not None and raw_type.lower() != "none"
    local_active = patch_type is not None and patch_type.lower() != "none"
    logger.info(f"🌐 Global Branch: {'✅ ACTIVE' if global_active else '❌ INACTIVE'}")
    logger.info(f"🔍 Local Branch: {'✅ ACTIVE' if local_active else '❌ INACTIVE'}")
    if global_active and local_active:
        logger.info("🎯 Mode: Dual Branch (Global + Local)")
    elif global_active:
        logger.info("🎯 Mode: Global Only")
    elif local_active:
        logger.info("🎯 Mode: Local Only")
    else:
        logger.info("❌ ERROR: No active branches!")
    logger.info("=" * 60)

def log_branch_details(model: nn.Module, logger: logging.Logger):
    logger.info("🔧 [Branch Details]")
    logger.info("=" * 60)
    if hasattr(model, 'global_branch') and model.global_branch is not None:
        logger.info("🌐 Global Branch:")
        logger.info(f"   - Type: {'Swin Transformer' if getattr(model, 'global_is_swin', False) else 'ResNet'}")
        logger.info(f"   - Feature Dim: {getattr(model, 'global_feature_dim', 'N/A')}")
        logger.info(f"   - Projection: {getattr(model, 'global_proj', 'N/A')}")
    elif hasattr(model, 'global_extractor') and model.global_extractor is not None:
        logger.info("🌐 Global Branch:")
        logger.info(f"   - Type: FeatureExtractor")
        logger.info(f"   - Feature Dim: {getattr(model.global_extractor, 'feature_dim', 'N/A')}")
        logger.info(f"   - Projection: {getattr(model, 'raw_proj', 'N/A')}")
    else:
        logger.info("🌐 Global Branch: ❌ None")
    if hasattr(model, 'local_branch') and model.local_branch is not None:
        logger.info("🔍 Local Branch:")
        logger.info(f"   - Type: {'Swin Transformer' if getattr(model, 'local_is_swin', False) else 'ResNet'}")
        logger.info(f"   - Feature Dim: {getattr(model, 'local_feature_dim', 'N/A')}")
        logger.info(f"   - Projection: {getattr(model, 'local_proj', 'N/A')}")
    elif hasattr(model, 'local_extractor') and model.local_extractor is not None:
        logger.info("🔍 Local Branch:")
        logger.info(f"   - Type: FeatureExtractor")
        logger.info(f"   - Feature Dim: {getattr(model.local_extractor, 'feature_dim', 'N/A')}")
        logger.info(f"   - Projection: {getattr(model, 'patch_proj', 'N/A')}")
    else:
        logger.info("🔍 Local Branch: ❌ None")
    if hasattr(model, 'classifier') and model.classifier is not None:
        logger.info("🎯 Classifier:")
        logger.info(f"   - Output Dim: {getattr(model, 'output_dim', 'N/A')}")
        logger.info(f"   - Architecture: {model.classifier}")
    else:
        logger.info("🎯 Classifier: ❌ None")
    logger.info("=" * 60) 