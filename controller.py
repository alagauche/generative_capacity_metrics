from Vis import Vis
import sys
import json
from pprint import pprint
import sys

if __name__ == '__main__':
    config_vis_file = sys.argv[1]
    vis = Vis(config_vis_file)
    '''
    vis.synth_parent_dir_name = sys.argv[1]
    vis.vis_seqs['synth-vae'] = sys.argv[2]
    vis.nat_parent_dir_name = sys.argv[3]
    vis.vis_seqs['natural-vae'] = sys.argv[4]
    vis.vis_seqs['natural'] = sys.argv[5]
    vis.which_size = sys.argv[6]
    vis.synth_nat = sys.argv[7]
    if vis.synth_nat == "synth":
        vis.name = vis.synth_parent_dir_name
    else:
        vis.name = vis.nat_parent_dir_name
        print("nat parent dir name:\t", vis.nat_parent_dir_name)
    print(vis.vis_seqs)
    '''
    for v in vis.which_vis:
        vis.run_vis(v)
