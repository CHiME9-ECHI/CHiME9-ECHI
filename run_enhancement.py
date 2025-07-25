# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import hydra
from omegaconf import OmegaConf

from scripts.enhancement.enhance import enhance_all_sessions
from scripts.enhancement.resample import resample_for_enhancement


@hydra.main(version_base=None, config_path="config/enhancement", config_name="main")
def main(cfg):
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    if cfg.resample.run:
        resample_for_enhancement(cfg.resample)

    if cfg.enhance.run:
        enhance_all_sessions(cfg.enhance, cfg.enhance_args)


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
