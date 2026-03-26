# Register SO-ARM TrainConfigs into OpenPi's global registry (runtime).
# Mount repo ``training/configs`` at openpi ``src/openpi/training/configs_custom``.

from __future__ import annotations


def register() -> None:
    import openpi.training.config as cfg

    try:
        from openpi.training.configs_custom import soarm_config as sc
    except ImportError:
        try:
            # Training container copies ``soarm_config.py`` into ``openpi/training/``.
            from openpi.training import soarm_config as sc
        except ImportError as e:
            raise RuntimeError(
                "Cannot import SOARM config module. "
                "Inference: mount repo training/configs to "
                ".../openpi/training/configs_custom. "
                "Training: ensure /training/configs is mounted and copied before register."
            ) from e

    raw = getattr(sc, "SOARM_TRAIN_CONFIGS", None)
    if raw is None:
        raw = getattr(sc, "SOARM_CONFIGS", ())
    if isinstance(raw, dict):
        raw = list(raw.values())
    for train_cfg in raw:
        name = getattr(train_cfg, "name", None)
        if not name:
            continue
        if name in cfg._CONFIGS_DICT:
            continue
        cfg._CONFIGS.append(train_cfg)
        cfg._CONFIGS_DICT[name] = train_cfg


def main() -> None:
    register()


if __name__ == "__main__":
    main()
