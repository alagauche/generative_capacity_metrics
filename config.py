from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path

@dataclass
class Config:
    parent_dir_name: Path
    msa_dir: Path
    output_dir: Path

    vis_seqs: Dict[str, Path]
    label_dict: Dict[str, str]
    name: str
    which_models: Dict[str, bool]
    skip: List[str]
    compute_homs_script: Path
    bvms_script: Path
    covars_script: Path
    synth_nat: str
    which_size: str
    protein: str
    L: int
    A: int
    alpha: float
    zoom: bool
    font: str
    font_scale: float
    fig_size: float
    title_size: int
    title: str
    label_size: int
    tick_size: int
    line_width: float
    line_alpha: float
    label_padding: float
    box_font_size: float
    box_padding: int
    marker_size: int
    color_set: Dict[str, str]
    z_order: Dict[str, int]
    box_style: str
    face_color: str
    box_alpha: float
    tick_rotation: int
    tick_length: int
    tick_width: float
    stats_sf: int
    dpi: int
    make_hams_dist: bool
    keep_hams: int
    pss: int
    compute_homs: bool
    parse_homs: bool
    plot_homs: bool
    r20_folder: Path
    r20_start: int
    r20_end: int
    r20_mod: bool
    r20_data_black: Path
    r20_data: Path
    compute_all_covars: bool
    which_covars: bool
    covars: Dict[str, Path]
    compute_all_bvms: bool
    which_bvms: str
    bvms: Dict[str, Path]
    keep_covars: int
    valseqs: str
    fields_couplings: Dict[str, Path]
    ALPHA: str
    energies_file: Path
