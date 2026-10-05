# Imports
import copy
import argparse
import random
import numpy as np
import torch
import pytorch_lightning as ptl
from data import DataModule
from training import LitModule
from utils import Config

#####################################################################
#   Runners
#####################################################################

class Runner:
    """
    Handles setting up all various sub-modules for training.
    Args:
        global_cfg: A config with all sub-modules configs
            (runner, lit_module, data_module)
        _using_ray: Internal argument. Set to True when using Ray to
            prepare the PTL trainer.
        _override_exp_dir: Internal argument. Config containing save_dir,
            name, & version to override logger settings.
    """
    def __init__(self, global_cfg: Config,
                 _override_exp_dir: Config=None) -> None:
        self.global_cfg = global_cfg
        self.cfg = global_cfg.runner
        self.lm_cfg = global_cfg.lit_module
        self.dm_cfg = global_cfg.data_module
        self.cfg.file = self.global_cfg.file
        self.lm_cfg.file = self.global_cfg.file
        self.dm_cfg.file = self.global_cfg.file
        self._override_exp_dir = _override_exp_dir
        self._setup()

    def execute(self, routine: str='train') -> None:
        if routine == 'train':
            if not self.global_cfg.debug and self.lm_cfg.compile:
                self._lit_module = torch.compile(self._lit_module)
            self._ptl_trainer.fit(self._lit_module, self._data_module,
                                  ckpt_path=(self.cfg.resume_training_ckpt if
                                      self.cfg.has('resume_training_ckpt') else None))
        else:
            raise NotImplementedError(routine)

    def _setup(self) -> None:
        self._setup_data_module()
        self._setup_lit_module()
        self._setup_ptl_trainer()

    def _setup_lit_module(self, save_hparams: bool = True) -> None:
        if self.cfg.has('starting_params_ckpt'):
            self._lit_module = LitModule.load_from_checkpoint(
                self.cfg.starting_params_ckpt, cfg=self.lm_cfg, strict=False)
        elif self.cfg.has('world_model_ckpt'):
            self._lit_module = LitModule(self.lm_cfg)
            ckpt = torch.load(self.cfg.world_model_ckpt, weights_only=False)
            state_dict = ckpt['state_dict']
            wm_state_dict = {k: v for k, v in state_dict.items() if 'world_model' in k}
            self._lit_module.load_state_dict(wm_state_dict, strict=False)
        else:
            self._lit_module = LitModule(self.lm_cfg)
        cfg = copy.deepcopy(self.global_cfg).to_dict_recursive()
        if save_hparams:
            self._lit_module.save_hyperparameters(cfg)

    def _setup_data_module(self) -> None:
        self._data_module = DataModule(self.dm_cfg)

    def _setup_ptl_trainer(self) -> None:
        self._prep_loggers()
        self.cfg.ptl_trainer_args.load_all_instances()
        self._ptl_trainer = ptl.Trainer(
                num_sanity_val_steps=0,
                enable_progress_bar=False,
                **self.cfg.ptl_trainer_args.to_dict())
    
    def _prep_loggers(self) -> None:
        if self.cfg.ptl_trainer_args.has('logger'):
            if isinstance(self.cfg.ptl_trainer_args.logger, list):
                for i in range(len(self.cfg.ptl_trainer_args.logger)):
                    if self._override_exp_dir:
                        list(self.cfg.ptl_trainer_args.logger[
                            i].to_dict().values())[0].save_dir = self._override_exp_dir.save_dir
                        list(self.cfg.ptl_trainer_args.logger[
                            i].to_dict().values())[0].name = self._override_exp_dir.name
                        list(self.cfg.ptl_trainer_args.logger[
                            i].to_dict().values())[0].version = self._override_exp_dir.version
                    self.cfg.ptl_trainer_args.logger[i] = self.cfg.ptl_trainer_args.logger[i].create_instance()
                l = self.cfg.ptl_trainer_args.logger[0]
            else:
                if self._override_exp_dir:
                    list(self.cfg.ptl_trainer_args.logger.save_dir.to_dict().values())[0] = self._override_exp_dir.save_dir
                    list(self.cfg.ptl_trainer_args.logger.name)[0] = self._override_exp_dir.name
                    list(self.cfg.ptl_trainer_args.logger.version)[0] = self._override_exp_dir.version
                self.cfg.ptl_trainer_args.logger = self.cfg.ptl_trainer_args.logger.create_instance()
                l = self.cfg.ptl_trainer_args.logger
                self.cfg.ptl_trainer_args.logger = [l]
            self._log_dir = Config(save_dir=l.save_dir, name=l.name, version=(
                f'version_{l.version}' if isinstance(l.version, int) else l.version))

def main() -> None:
    # CLI
    parser = argparse.ArgumentParser(description='Run training, etc.')
    parser.add_argument('-cfg', type=str, required=True, help='Path to .yaml configuration file.')
    parser.add_argument('-routine', type=str, default='train', help='What routine to run. (Options: train)')
    parser.add_argument('-debug', action='store_true', help='Flag to run in debug mode.')
    args = parser.parse_args()

    setattr(Config, 'debug', args.debug)
    cfg = Config.from_yaml(args.cfg)
    seed = cfg.runner.rng_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    runner = Runner(cfg)
    runner.execute(args.routine)


if __name__ == '__main__':
    main()

