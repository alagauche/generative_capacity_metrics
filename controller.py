from pathlib import Path
import json
from typing import Dict, List
from dataclasses import dataclass
from pprint import pprint

import fire

from Vis import Vis
from config import Config

class Main:
    def __init__(self, config_vis_file: Path = Path('config_vis.json')):
        print("reading config_file: ", config_vis_file)
        with config_vis_file.open("r") as fp:
            conf = json.load(fp)
            print('Config:')
            pprint(conf)
            self.config = Config(**conf)

        self.vis = Vis(config_vis_file)

    def all(self):
        self.vis.plot_hams()
        self.vis.plot_hams_loglog()
        self.vis.make_covars()
        self.vis.make_homs()
        self.vis.make_energies()

    def hams(self):
        self.vis.plot_hams()

    def hams_loglog(self):
        self.vis.plot_hams_loglog()

    def covars(self):
        self.vis.make_covars()

    def homs(self):
        self.vis.make_homs()

    def energies(self):
        self.vis.make_energies()

if __name__ == "__main__":
    fire.Fire(Main)