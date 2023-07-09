from Vis import Vis
import sys
import json
from pprint import pprint
import sys

if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("Usage: python3 controller.py [config_vis_file]")
        sys.exit(1)
    config_vis_file = sys.argv[1]
    vis = Vis(config_vis_file)

    for v in vis.which_vis:
        vis.run_vis(v)
