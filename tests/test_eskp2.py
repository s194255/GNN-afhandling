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
        eksp2['downstream']['epoker'] = 1
        eksp2['selvvejledt']['epoker'] = 1
        eksp2['trin'] = 1
        self.eksp2_path = "config/eksp2_debug.yaml"
        with open(self.eksp2_path, 'w', encoding='utf-8') as fil:
            yaml.dump(eksp2, fil, allow_unicode=True)
        self.selv_ckpt_path = None


def test_eksp2():
    args = Args()
    eksp2 = Eksp2(args)
    eksp2.main()
