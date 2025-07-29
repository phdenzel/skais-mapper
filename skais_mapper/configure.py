# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of skais-mapper
# SPDX-License-Identifier: GPL-3.0-or-later
"""Inspect and create configuration files."""

import logging
from pathlib import Path
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
import skais_mapper
from skais_mapper.utils import compress_encode


@hydra.main(config_path="configs", config_name="config", version_base=None)
def create(cfg: DictConfig | dict):
    """Main configuration creation routine."""
    log = logging.getLogger(__name__)
    output_dir = HydraConfig.get().runtime.output_dir
    opt = instantiate(cfg)
    if not cfg.exclude_git_state:
        log.info(f"Git state: {skais_mapper.GIT_STATE}")
    if cfg.include_git_diff:
        for d in skais_mapper.GIT_DIFF:
            log.info(f"Git diff: {compress_encode(d)}")
    log.info(f"Job id: {opt.run_id}")
    log.info(f"Output directory: {output_dir}")
    if opt.verbose:
        print("Configuration:")
        print(OmegaConf.to_yaml(cfg))

    if opt.save_configs:
        hydra_subdir = Path(HydraConfig.get().output_subdir)
        src_file = hydra_subdir / "config.yaml"
        dst_file = Path(opt.output)
        if dst_file.suffix not in [".yaml", ".yml"]:
            dst_file = Path(f"./{opt.run_id}.yaml")
        dst_file.write_bytes(src_file.read_bytes())
