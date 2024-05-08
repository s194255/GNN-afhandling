import lightning as L
import lightning.pytorch
import torch
import yaml
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
import os
import wandb
from src.models.selvvejledt import Selvvejledt, SelvvejledtQM9
from src.models.downstream import Downstream
from src.data.QM9 import QM9ByggerEksp2

# from src import data as d

MODELKLASSER = {
    'Selvvejledt': Selvvejledt,
    'SelvvejledtQM9': SelvvejledtQM9
}

MODELOPGAVER = {
    'Selvvejledt': 'selvvejledt',
    'SelvvejledtQM9': 'downstream'
}

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


def get_opgaver_in_config(config):
    return list(set(config.keys())-(set(config.keys()) - {'Downstream', 'Selvvejledt', 'SelvvejledtQM9'}))

def load_config(path):
    with open(path, encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    rygradtype = config_dict['rygradtype']
    with open(os.path.join("config", "rygrader", f"{rygradtype}.yaml"), encoding='utf-8') as f:
        rygrad = yaml.safe_load(f)
    opgaver_in_config = get_opgaver_in_config(config_dict)
    for opgave_in_config in opgaver_in_config:
        for variant in config_dict[opgave_in_config].keys():
            hovedtype = config_dict[opgave_in_config][variant]['model']['hovedtype']
            hoved_config_path = os.path.join("config", opgave_in_config, f"{hovedtype}.yaml")
            with open(hoved_config_path, encoding='utf-8') as f:
                hoved_config_dict = yaml.safe_load(f)
            config_dict[opgave_in_config][variant]['model']['hoved'] = hoved_config_dict
            config_dict[opgave_in_config][variant]['model']['rygradtype'] = rygradtype
            config_dict[opgave_in_config][variant]['model']['rygrad'] = rygrad
    return config_dict

def debugify_config(config):
    config['datasæt']['debug'] = True
    config['datasæt']['batch_size'] = 1
    config['datasæt']['num_workers'] = 0
    config['datasæt']['n_trin'] = 1
    for opgave in get_opgaver_in_config(config):
        for variant in config[opgave].keys():
            config[opgave][variant]['epoker'] = 1
            config[opgave][variant]['check_val_every_n_epoch'] = 1
            config[opgave][variant]['model']['rygrad']['hidden_channels'] = 8


def get_n_epoker(artefakt_sti):
    if artefakt_sti == None:
        return 0
    else:
        state_dict = torch.load(artefakt_sti, map_location='cpu')
        return state_dict['epoch']

def indlæs_selv_ckpt_path(selv_ckpt_path):
    api = wandb.Api()
    artefakt = api.artifact(selv_ckpt_path)
    artefakt_sti = os.path.join(artefakt.download(), 'model.ckpt')
    run_id = artefakt.logged_by().id
    return artefakt_sti, run_id

def _get_modelklasse(ckpt_path):
    state_dict = torch.load(ckpt_path, map_location='cpu')
    modelklasse_str = state_dict['hyper_parameters']['args_dict']['modelklasse']
    return MODELKLASSER[modelklasse_str]

def get_selvvejledt_fra_artefakt_sti(config, artefakt_sti):
    modelklasse = _get_modelklasse(artefakt_sti)
    selvvejledt = modelklasse.load_from_checkpoint(artefakt_sti)
    qm9bygger = QM9ByggerEksp2.load_from_checkpoint(artefakt_sti, **config['datasæt'])
    return selvvejledt, qm9bygger

def get_selvvejledt_fra_wandb(config, wandb_path):
    artefakt_sti, run_id = indlæs_selv_ckpt_path(wandb_path)
    selvvejledt, qm9bygger = get_selvvejledt_fra_artefakt_sti(config, artefakt_sti)
    return selvvejledt, qm9bygger, artefakt_sti, run_id

def build_selvvejledt(args_dict, datasæt_dict, modelklasse_str):
    qm9bygger = QM9ByggerEksp2(**datasæt_dict)
    if modelklasse_str == 'Selvvejledt':
        model = Selvvejledt(args_dict=args_dict)
    elif modelklasse_str == 'SelvvejledtQM9':
        model = SelvvejledtQM9(args_dict=args_dict, metadata=qm9bygger.get_metadata2('pretrain'))
    else:
        raise NotImplementedError
    return model, qm9bygger
