import src.models as m
from src.eksp2 import Eksp2
import os
import yaml


class Args:
    def __init__(self):
        eksp2 = m.load_config("config/eksp2.yaml")
        eksp2['datasæt']['debug'] = True
        eksp2['datasæt']['batch_size'] = 1
        eksp2['datasæt']['num_workers'] = 0
        eksp2['datasæt']['n_trin'] = 1
        eksp2['downstream']['epoker'] = 1
        eksp2['selvvejledt']['epoker'] = 1
        eksp2['rygrad']['hidden_channels'] = 8
        self.eksp2_path = "config/eksp2_debug.yaml"
        if os.path.exists(self.eksp2_path):
            os.remove(self.eksp2_path)
        with open(self.eksp2_path, 'w', encoding='utf-8') as fil:
            yaml.dump(eksp2, fil, allow_unicode=True)
        self.selv_ckpt_path = None


def test_eksp2():
    args_1 = Args()
    eksp2_1 = Eksp2(args_1)
    eksp2_1.main()

    args_2 = Args()
    args_2.selv_ckpt_path = eksp2_1.selv_ckpt_path
    eksp2_2 = Eksp2(args_2)
    eksp2_2.main()