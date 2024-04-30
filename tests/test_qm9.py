import src.data as d
import src.models as m
import src.redskaber as r

def manip_config(config):
    config['datasæt']['debug'] = True
    config['datasæt']['batch_size'] = 1
    config['datasæt']['num_workers'] = 0
    config['datasæt']['n_trin'] = 1
    config['selvvejledt']['epoker'] = 1
    config['rygrad']['hidden_channels'] = 8

def test_qm9bygger2():
    config = r.load_config("config/fortræn.yaml")
    manip_config(config)
    selvvejledt, qm9bygger, _, _ = r.get_selvvejledt_fra_wandb(config, None)
    trainer = r.get_trainer(config['selvvejledt']['epoker'])
    trainer.fit(model=selvvejledt, datamodule=qm9bygger)

    ckpt_path = trainer.checkpoint_callback.best_model_path
    qm9bygger_2 = d.QM9ByggerEksp2.load_from_checkpoint(ckpt_path, **config['datasæt'])
    assert qm9bygger_2.data_splits == qm9bygger.data_splits