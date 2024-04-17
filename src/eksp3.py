# from src.eksp1 import uden_selvtræn
import argparse
import src.models as m
import lightning as L
import src.data as d
import src.redskaber as r
import src.models.downstream
from lightning.pytorch.loggers import WandbLogger

N = 130831

class QM9Bygger3(d.QM9Bygger):

    def __init__(self, *args, n_train, **kwargs):
        fordeling = self.get_fordeling(n_train)
        if 'fordeling' in kwargs.keys():
            assert fordeling == kwargs['fordeling']
        else:
            kwargs = {**kwargs, "fordeling": fordeling}
        super().__init__(*args, **kwargs)

    def get_fordeling(self, n_train):
        n = len(self.get_mother_dataset())
        assert n_train < n
        train = n_train/n
        val = 0.25*train
        test = 0.04
        pretrain = (1 - (train+ val+test))*0.8
        preval = (1 - (train + val + test)) * 0.2
        return [pretrain, preval, train, val, test]
class Eksp3:

    def __init__(self, args):
        self.args = args
        self.config = m.load_config(args.eksp3_path)

    def get_fordeling(self, n_train):

        assert n_train < N
        train = n_train / N
        val = 0.25 * train
        test = 0.04
        pretrain = 1 - (train + val + test)
        return [pretrain, 0.0, train, val, test]

    def main(self):
        if self.args.ckpt_path:
            downstream = m.Downstream.load_from_checkpoint(self.args.ckpt_path)
            qm9 = QM9Bygger3.load_from_checkpoint(self.args.ckpt_path)
        else:
            qm9 = QM9Bygger3(**self.config['datasæt'])
            downstream = m.Downstream(rygrad_args=self.config['rygrad'],
                                     hoved_args=self.config['downstream']['hoved'],
                                     args_dict=self.config['downstream']['model'])
        if self.config['frys_rygrad']:
            downstream.frys_rygrad()

        callbacks = [
            r.checkpoint_callback(),
            r.TQDMProgressBar(),
            L.pytorch.callbacks.LearningRateMonitor(logging_interval='step')
        ]
        logger = WandbLogger(project='afhandling', log_model=True)
        trainer = L.Trainer(max_epochs=self.config['downstream']['epoker'],
                            callbacks=callbacks,
                            log_every_n_steps=10,
                            logger=logger)
        trainer.fit(model=downstream, datamodule=qm9, ckpt_path=self.args.ckpt_path)
def parserargs():
    parser = argparse.ArgumentParser(description='Beskrivelse af dit script')
    parser.add_argument('--eksp3_path', type=str, default="config/eksp3.yaml", help='Sti til eksp1 YAML fil')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Sti til downstream hoved arguments YAML fil')
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    args = parserargs()
    # eksp3 = m.load_config(args.eksp3_path)
    # uden_selvtræn()
    eksp3 = Eksp3(args)
    eksp3.main()