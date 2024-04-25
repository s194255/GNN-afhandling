import lightning as L
import lightning.pytorch
import torch
import yaml
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
import os
import wandb
import src.models as m
import src.data as d

from torch_scatter import scatter_mean, scatter_add


def TQDMProgressBar():
    return L.pytorch.callbacks.TQDMProgressBar(refresh_rate=1000)

def earlyStopping(min_delta, patience):
    return L.pytorch.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                      min_delta=min_delta, patience=patience)

def checkpoint_callback(dirpath=None):
    return L.pytorch.callbacks.ModelCheckpoint(monitor='val_loss', mode='min',
                                                              save_top_k=1, filename='best', save_last=True,
                                               dirpath=dirpath)

def learning_rate_monitor():
    return L.pytorch.callbacks.LearningRateMonitor(logging_interval='step')
def tensorBoardLogger(save_dir=None, name=None, version=None):
    if not save_dir:
        save_dir = os.getcwd()
    if not name:
        name = "lightning_logs"
    return TensorBoardLogger(save_dir=save_dir, name=name, version=version)

def wandbLogger(log_model=False, tags=None, group=None):
    return lightning.pytorch.loggers.WandbLogger(
        project='afhandling',
        log_model=log_model,
        tags=tags,
        group=group)

def get_trainer(epoker, logger=None):
    callbacks = [
        checkpoint_callback(),
        TQDMProgressBar(),
        learning_rate_monitor()
    ]
    trainer = L.Trainer(max_epochs=epoker,
                        log_every_n_steps=50,
                        callbacks=callbacks,
                        precision='16-mixed',
                        logger=logger,
                        )
    return trainer

def get_next_wandb_kørselsid():
    wandb.login()

    # Hent informasjon om alle kjøringer (runs) i et prosjekt
    # runs = wandb.Api().runs("your_username/your_project")
    runs = wandb.Api().runs("afhandling")
    kørselsider = []
    for run in runs:
        group = run.group
        if hasattr(run, "kørselsid"):
            kørselsider.append(run.kørselsid)
    return max(kørselsider, default=-1)+1

def get_selvvejledt(config, selv_ckpt_path):
    if selv_ckpt_path:
        api = wandb.Api()
        artefakt = api.artifact(selv_ckpt_path)
        artefakt_sti = os.path.join(artefakt.download(), 'model.ckpt')
        selvvejledt = m.Selvvejledt.load_from_checkpoint(artefakt_sti)
        qm9bygger = d.QM9ByggerEksp2.load_from_checkpoint(artefakt_sti, **config['datasæt'])
        run_id = artefakt.logged_by().id
    else:
        selvvejledt = m.Selvvejledt(rygrad_args=config['rygrad'],
                                    args_dict=config['selvvejledt']['model'])
        qm9bygger = d.QM9ByggerEksp2(**config['datasæt'])
        artefakt_sti = None
        run_id = None
    return selvvejledt, qm9bygger, artefakt_sti, run_id

class RiemannGaussian(L.LightningModule):

    def __init__(self):
        super().__init__()
        # TODO: gør så man kan bruge T'er
        self.T = 1

    @torch.no_grad()
    def get_s(self, pos_til, pos, batch, sigma):
        v = pos.shape[-1]
        center = scatter_mean(pos, batch, dim=-2)  # B * 3
        perturbed_center = scatter_mean(pos_til, batch, dim=-2)  # B * 3
        pos_c = pos - center[batch]
        perturbed_pos_c = pos_til - perturbed_center[batch]
        perturbed_pos_c_left = perturbed_pos_c.repeat_interleave(v, dim=-1)
        perturbed_pos_c_right = perturbed_pos_c.repeat([1, v])
        pos_c_left = pos_c.repeat_interleave(v, dim=-1)
        ptp = scatter_add(perturbed_pos_c_left * perturbed_pos_c_right, batch, dim=-2).reshape(-1, v,
                                                                                               v)  # B * 3 * 3
        otp = scatter_add(pos_c_left * perturbed_pos_c_right, batch, dim=-2).reshape(-1, v, v)  # B * 3 * 3
        ptp = ptp[batch]
        otp = otp[batch]
        # s = - 2 * (perturbed_pos_c.unsqueeze(1) @ ptp - pos_c.unsqueeze(1) @ otp).squeeze(1) / (
        #         torch.norm(ptp, dim=(1, 2)) + torch.norm(otp, dim=(1, 2))).unsqueeze(-1).repeat([1, 3])
        s = (perturbed_pos_c.unsqueeze(1) @ ptp - pos_c.unsqueeze(1) @ otp).squeeze(1)
        s = -(1/sigma**2).view(-1, 1) * s
        alpha = (torch.norm(ptp, dim=(1, 2)) + torch.norm(otp, dim=(1, 2)))/2
        return s, alpha
    @torch.no_grad()
    def forward(self,
                pos: torch.Tensor,
                batch: torch.Tensor,
                sigma: torch.Tensor,
                ):
        pos_til = pos.clone()
        for t in range(1, self.T+1):
            beta = (sigma**2)/(2**t)
            s, alpha = self.get_s(pos_til, pos, batch, sigma)
            pos_til = pos_til + (beta/alpha).view(-1, 1) * s + torch.sqrt(2*beta).view(-1, 1)*torch.randn_like(pos)
        target = (1/alpha).view(-1, 1) * s
        return pos_til, target

def _get_opgaver_in_config(config):
    return list(set(config.keys())-(set(config.keys()) - {'downstream', 'selvvejledt'}))

def load_config(path):
    with open(path, encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    opgaver_in_config = _get_opgaver_in_config(config_dict)
    for opgave_in_config in opgaver_in_config:
        hovedtype = config_dict[opgave_in_config]['model']['hovedtype']
        hoved_config_path = os.path.join("config", opgave_in_config, f"{hovedtype}.yaml")
        with open(hoved_config_path, encoding='utf-8') as f:
            hoved_config_dict = yaml.safe_load(f)
        config_dict[opgave_in_config]['model']['hoved'] = hoved_config_dict
    return config_dict

def debugify_config(config):
    config['datasæt']['debug'] = True
    config['datasæt']['batch_size'] = 1
    config['datasæt']['num_workers'] = 0
    config['datasæt']['n_trin'] = 3
    config['rygrad']['hidden_channels'] = 8
    for opgave in _get_opgaver_in_config(config):
        config[opgave]['epoker'] = 1


def get_n_epoker(artefakt_sti):
    if artefakt_sti == None:
        return 0
    else:
        state_dict = torch.load(artefakt_sti, map_location='cpu')
        return state_dict['epoch']
