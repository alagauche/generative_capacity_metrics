import json, os, subprocess
from pprint import pprint
from copy import deepcopy
from typing import List, Dict

import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition, mark_inset

import pandas as pd
import numpy as np
import helper as h

from helper import timer
from mi3gpu.utils.seqload import loadSeqs, writeSeqs
from mi3gpu.utils.seqtools import histsim
import VisHamsHelper as VHH
import VisCovarsHelper as VCH
import VisHOMSHelper as VHOMSH

# import VisEnergiesHelper VEH
from scipy.stats import pearsonr, spearmanr


class Vis:
    which_vis: List[str]
    vis_seqs: Dict[str, str]
    label_dict: Dict[str, str]
    name: str
    parent_dir_name: str
    msa_dir: str # TODO path
    which_models: Dict[str, bool]
    output_dir: str # TODO path
    skip: List[str]
    compute_homs_script: str
    bvms_script: str
    covars_script: str
    data_home: str # TODO path
    synth_nat: str
    which_size: str
    protein: str
    L: int
    A: int
    alpha: float
    loglog: bool
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
    r20_folder: str
    r20_start: int
    r20_end: int
    r20_mod: bool
    r20_data_black: str
    r20_data: str
    compute_all_covars: bool
    which_covars: bool
    covars: Dict[str, str]
    compute_all_bvms: bool
    which_bvms: str
    bvms: Dict[str, str]
    keep_covars: int
    valseqs: str
    fields_couplings: Dict[str, str]
    ALPHA: str
    energies_file: str

    def __init__(self, config_file=None, args=None):
        if config_file:
            print("reading config_file: ", config_file)
            self.read_config(config_file)
        else:
            raise ValueError("Either config_file or args parameters must not be None.")

    # read in the JSON formatted config file passed to the VAE object as a string
    def read_config(self, config_file):
        conf = None
        with open(config_file, "r") as fp:
            conf = json.load(fp)
        for item in conf:
            print(f"{item}: {conf[item]}")

            self.which_vis = conf["which_vis"]
            self.vis_seqs = conf["vis_seqs"]
            self.label_dict = conf["label_dict"]
            self.name = conf["name"]
            self.which_models = conf["which_models"]
            self.output_dir = conf["output_dir"]
            self.parent_dir_name = conf["parent_dir_name"]
            self.msa_dir = conf["msa_dir"]
            self.skip = conf["skip"]
            self.compute_homs_script = conf["compute_homs_script"]
            self.bvms_script = conf["bvms_script"]
            self.covars_script = conf["covars_script"]
            self.synth_nat = conf["synth_nat"]
            self.which_size = conf["which_size"]
            self.protein = conf["protein"]
            self.L = conf["L"]
            self.A = conf["A"]
            self.ALPHA = conf["ALPHA"]
            self.alpha = conf["alpha"]
            self.font = conf["font"]
            self.font_scale = conf["font_scale"]
            self.fig_size = conf["fig_size"]
            self.title_size = conf["title_size"]
            self.title = conf["title"]
            self.label_size = conf["label_size"]
            self.tick_size = conf["tick_size"]
            self.line_width = conf["line_width"]
            self.line_alpha = conf["line_alpha"]
            self.label_padding = conf["label_padding"]
            self.box_font_size = conf["box_font_size"]
            self.box_padding = conf["box_padding"]
            self.marker_size = conf["marker_size"]
            self.color_set = conf["color_set"]
            self.z_order = conf["z_order"]
            self.box_style = conf["box_style"]
            self.box_font_size = conf["box_font_size"]
            self.face_color = conf["face_color"]
            self.box_alpha = conf["box_alpha"]
            self.tick_rotation = conf["tick_rotation"]
            self.tick_length = conf["tick_length"]
            self.tick_width = conf["tick_width"]
            self.stats_sf = conf["stats_sf"]
            self.dpi = conf["dpi"]
            self.loglog = conf["loglog"]
            self.zoom = conf["zoom"]
            self.make_hams_dist = conf["make_hams_dist"]
            self.keep_hams = conf["keep_hams"]
            self.pss = conf["pss"]
            self.compute_homs = conf["compute_homs"]
            self.parse_homs = conf["parse_homs"]
            self.plot_homs = conf["plot_homs"]
            self.r20_start = conf["r20_start"]
            self.r20_end = conf["r20_end"]
            self.r20_folder = conf["r20_folder"]
            self.r20_mod = conf["r20_mod"]
            self.r20_data_black = conf["r20_data_black"]
            self.r20_data = conf["r20_data"]
            self.compute_all_covars = conf["compute_all_covars"]
            self.which_covars = conf["which_covars"]
            self.covars = conf["covars"]
            self.compute_all_bvms = conf["compute_all_bvms"]
            self.which_bvms = conf["which_bvms"]
            self.bvms = conf["bvms"]
            self.keep_covars = conf["keep_covars"]
            self.valseqs = conf["valseqs"]
            self.fields_couplings = conf["fields_couplings"]
            self.energies_file = conf["energies_file"]

    def run_vis(self, vis_type):
        print("running vis")
        all_types = ["hams", "covars", "homs", "energies"]

        if vis_type == "all":
            for t in all_types:
                self.make_vis(t)()
        else:
            self.make_vis(vis_type)()

    def make_vis(self, vis_type):
        print("\tmake_vis() for: ", vis_type)
        vis_funcs = {
            "hams": self.make_hams,
            "covars": self.make_covars,
            "homs": self.make_homs,
            "energies": self.make_energies,
        }
        return vis_funcs[vis_type]

    def make_hams(self):
        print("\t\tmaking hams")
        if self.loglog:
            print("\t\t\tplotting loglog")
            self.plot_hams_loglog()
        self.plot_hams()
        print("\t\tcompleted: plotting hams")

    def norm_x(self, hams, freqs):
        y_max = max(freqs)
        index = freqs.index(y_max)
        x_max = hams[index]
        return [np.log(x) - np.log(x_max) for x in hams]

    def norm_y(self, freqs):
        y_max = max(freqs)
        return [np.log(y) - np.log(y_max) for y in freqs]

    def plot_hams_loglog(self):
        print("MAKING LOGLOG HAMS")
        # make list of labels, filenames
        fig, ax = pylab.subplots(figsize=(self.fig_size, self.fig_size))
        xlabel = r"ln($d$/$d_{Mo}$)"
        ylabel = r"ln($f$/$f_{max}$)"

        for label, seqs_file in self.vis_seqs.items():
            if label in self.skip:
                print("skipping ", label)
                continue
            if not self.which_models[
                label
            ]:  # model is 'false' in the which_models{}, then continue
                continue
            label = self.label_dict[label]
            seqs_path = self.msa_dir + "/" + seqs_file
            print("computing hams for:\t", label, "\t\t\tin:\t" + seqs_path)
            seqs = loadSeqs(self.msa_dir + "/" + seqs_file, names=self.ALPHA)[0][
                0 : self.keep_hams
            ]
            h = histsim(seqs).astype(float)[::-1][1:].tolist()
            hams = np.arange(1, len(h) + 1, 1)

            d_norm_x = self.norm_x(hams, h)
            d_norm_y = self.norm_y(h)
            x_mask = list()
            y_mask = list()
            # if label == "Target":

            line_style = "solid"
            if label == "Target":
                if "nat" in self.synth_nat:
                    target_label = "Nat-Target"
                else:
                    target_label = "Synth-Target"
                h = histsim(seqs).astype(float)
                h = h / np.sum(h)
                rev_h = h[::-1]
                line_style = "dashed"
                my_dashes = (1, 1)

                ax.plot(
                    d_norm_x,
                    d_norm_y,
                    linestyle=line_style,
                    linewidth=self.line_width,
                    dashes=my_dashes,
                    alpha=self.line_alpha,
                    color=self.color_set[label],
                    label=target_label,
                    zorder=self.z_order[label],
                )
            else:
                ax.plot(
                    d_norm_x,
                    d_norm_y,
                    linestyle=line_style,
                    linewidth=self.line_width,
                    alpha=self.line_alpha,
                    color=self.color_set[label],
                    label=label,
                    zorder=self.z_order[label],
                )

        pylab.ylim(-8, 1)
        pylab.xlim(-0.5, 0.26)
        pylab.ylabel(ylabel, fontsize=self.label_size - 1)
        pylab.xlabel(xlabel, fontsize=self.label_size - 1)
        x_tick_range = np.arange(-0.5, 0.5, 0.25)
        y_tick_range = np.arange(-8, 1, 2)
        pylab.xticks(x_tick_range, rotation=45)
        pylab.yticks(y_tick_range)
        pylab.tick_params(
            direction="in",
            axis="both",
            which="major",
            labelsize=self.tick_size,
            length=self.tick_length,
            width=self.tick_width,
        )
        file_name = (
            "/loglog_ham_"
            + self.name
            + "_"
            + self.synth_nat
            + "_"
            + self.which_size
            + ".pdf"
        )
        # pylab.title(self.which_size, fontsize=self.title_size)
        pylab.tight_layout()
        pylab.legend(fontsize=self.tick_size - 3, loc="best", frameon=False)
        save_name = self.output_dir + "/" + file_name
        pylab.savefig(save_name, dpi=self.dpi, format="pdf")
        pylab.close()

    def plot_hams(self):
        print("plotting normal hams")
        fig, ax = pylab.subplots(figsize=(self.fig_size, self.fig_size))
        box_font = self.box_font_size

        # axes labels
        xlabel = r"$d$"
        ylabel = "f"
        if self.protein == "Kinase":
            start = 120
            end = 230
            x_tick_range = np.arange(start, end, 20)
            pylab.xlim(start, end)

        all_freqs = dict()

        for label, seqs_file in self.vis_seqs.items():
            if label in self.skip:
                print("skipping ", label)
                continue
            if not self.which_models[
                label
            ]:  # model is 'false' in the which_models{}, then continue
                continue
            label = self.label_dict[label]
            print("computing hams for:\t", label)
            seqs = loadSeqs(self.msa_dir + "/" + seqs_file, names=self.ALPHA)[0][
                0 : self.keep_hams
            ]
            h = histsim(seqs).astype(float)
            h = h / np.sum(h)
            all_freqs[label] = h
            rev_h = h[::-1]
            if label == "Target":
                if "nat" in self.synth_nat:
                    target_label = "Nat-Target"
                else:
                    target_label = "Synth-Target"
                line_style = "dashed"
                my_dashes = (1, 1)
                ax.plot(
                    rev_h,
                    linestyle=line_style,
                    linewidth=self.line_width,
                    dashes=my_dashes,
                    alpha=self.line_alpha,
                    color=self.color_set[label],
                    label=target_label,
                    zorder=self.z_order[label],
                )
            else:
                line_style = "solid"
                ax.plot(
                    rev_h,
                    linestyle=line_style,
                    linewidth=self.line_width,
                    alpha=self.line_alpha,
                    color=self.color_set[label],
                    label=label,
                    zorder=self.z_order[label],
                )

        tvds = dict()
        print("all_freqs")
        print(all_freqs.keys())
        delete_key = ""
        save_value = ""
        for data_label, f in all_freqs.items():
            if "arget" in data_label:
                save_value = f
                delete_key = data_label

        del all_freqs[delete_key]
        all_freqs["Target"] = save_value

        for data_label, f in all_freqs.items():
            if data_label != "Target":
                tvds[data_label] = round(np.sum(np.abs(all_freqs["Target"] - f)) / 2, 4)

        print(tvds)
        y_tick_range = np.arange(0.0, 0.08, 0.02)
        pylab.ylabel(ylabel, fontsize=self.label_size)
        pylab.xlabel(xlabel, fontsize=self.label_size)
        pylab.xticks(x_tick_range, rotation=45)
        pylab.yticks(y_tick_range)
        pylab.tick_params(
            direction="in",
            axis="both",
            which="major",
            labelsize=self.tick_size,
            length=self.tick_length,
            width=self.tick_width,
        )
        # my_title = "Hamming Distance Distributions\n" + self.parent_dir_name
        file_name = (
            "ham_" + self.name + "_" + self.synth_nat + "_" + self.which_size + ".pdf"
        )
        # pylab.title(self.which_size, fontsize=self.title_size)
        pylab.tight_layout()
        pylab.legend(fontsize=self.tick_size - 3, loc="upper left", frameon=False)
        save_name = self.output_dir + "/" + file_name
        print(save_name)
        pylab.savefig(save_name, dpi=self.dpi, format="pdf")
        pylab.close()

    def make_covars(self):
        print("\t\tmaking covars")
        if self.compute_all_bvms:
            print("computing all bvms")
            for label in self.bvms.keys():
                if label not in self.skip:
                    msa_file = self.vis_seqs[label]
                    self.get_bvms(label, msa_file)
        if self.compute_all_covars:
            print("computing all covars")
            for label in self.covars.keys():
                if label not in self.skip:
                    bvms_file = self.bvms[label]
                    self.get_covars(label, bvms_file)

        print("\t\t\tbvms:")
        print(f"\t\t\t{self.bvms}")
        print("\t\t\tcovars:\t")
        print(f"\t\t\t{self.covars}")
        print("\n\t\tplotting covars")
        self.plot_covars()
        print("\t\tcompleted: making covars")

    def get_bvms(self, label: str, msa_file: str):
        print("bvms for: ", self.bvms[label])
        dest = self.output_dir
        source = self.msa_dir

        bvms_command = " ".join(
            "python",
            self.bvms_script,
            label,
            msa_file,
            source,
            dest,
            str(self.A),
            str(self.keep_covars),
        )

        print(f"\t\t\tgetting bvms for:\t{label}")
        print(bvms_command)
        try:
            output = subprocess.check_output(["bash", "-c", bvms_command])
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "command '{}' return with error (code {}): {}".format(
                    e.cmd, e.returncode, e.output
                )
            )

    def get_covars(self, label, bvms_file):
        print("covars for: ", self.covars[label])
        path = self.output_dir
        covars_command = (
            "python " + self.covars_script + " " + label + " " + bvms_file + " " + path
        )

        print("\t\t\tgetting covars for:\t", label)
        print(covars_command)
        try:
            # Run for it's side effects
            _output = subprocess.check_output(["bash", "-c", covars_command])
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"command '{e.cmd}' return with error (code {e.returncode}): {e.output}"
            )

        print(f"completed covars for:\t{self.covars[label]}")

    def plot_covars(self):
        print("\t\t\t\tplotting covars:\t")
        fig, ax = plt.subplots(figsize=(self.fig_size, self.fig_size))
        marker_size = self.marker_size - 4
        start = -0.10
        end = 0.15
        x_tick_range = np.arange(start, end, 0.05)
        y_tick_range = np.arange(start, end, 0.05)
        box_props = dict(boxstyle=self.box_style, facecolor=self.face_color)
        # target_covars = np.load(self.data_home + "/" + self.covars["targetSeqs"]).ravel()
        target_covars = np.load(self.output_dir + "/" + self.covars["targetSeqs"])
        target_masked = np.ma.masked_inside(target_covars, -0.01, 0.01).ravel()
        target_covars = target_covars.ravel()
        mi3_covars = np.load(self.output_dir + "/" + self.covars["mi3Seqs"]).ravel()
        indep_covars = np.load(self.output_dir + "/" + self.covars["indepSeqs"]).ravel()
        svae_covars = np.load(self.output_dir + "/" + self.covars["svaeSeqs"]).ravel()
        deepsequence_covars = np.load(
            self.output_dir + "/" + self.covars["deepSeqs"]
        ).ravel()
        data_home = ""
        other_covars = {
            "Indep": indep_covars,
            "Mi3": mi3_covars,
            "sVAE": svae_covars,
            "DeepSeq": deepsequence_covars,
        }
        for label, covars in other_covars.items():
            print("\t\t\t\t\tcovar corrs for:\ttargetSeqs", "\t\t", label)
            # pearson_r, pearson_p = pearsonr(covars_a, covars_b)
            # pearson_r = pearsonr(target_covars, covars)
            pearson_r, pearson_p = pearsonr(target_covars, covars)
            # print(pearson_r)
            c = self.color_set[label]
            print(pearson_r, pearson_p)
            label_text = label
            label_text = (
                label + ", " + r"$\rho$ = " + str(round(pearson_r, self.stats_sf))
            )  # orig with rho
            # plt.plot(target_masked, covars, 'o', markersize=marker_size, color=c, alpha=self.alpha, label=label_text)
            ax.plot(
                target_masked,
                covars,
                "o",
                markersize=marker_size,
                color=c,
                label=label_text,
                zorder=self.z_order[label],
                alpha=0.9,
            )

        ax.set_rasterization_zorder(0)
        if "nat" in self.synth_nat:
            xlabel = "Nat-Target Covariances"
        else:
            xlabel = "Synth-Target Covariances"
        pylab.xlabel(xlabel, fontsize=self.label_size)
        pylab.ylabel("GPSM Covariances", fontsize=self.label_size)
        pylab.xticks(x_tick_range, rotation=self.tick_rotation, fontsize=self.tick_size)
        pylab.yticks(y_tick_range, fontsize=self.tick_size)
        pylab.tick_params(
            direction="in",
            axis="both",
            which="major",
            labelsize=self.tick_size,
            length=self.tick_length,
            width=self.tick_width,
        )
        lim_start = -0.12
        lim_end = 0.12
        pylab.xlim((lim_start, lim_end))
        pylab.ylim((lim_start, lim_end))
        pylab.tight_layout()
        file_name = f"covars_{self.name}_{self.keep_covars}_{self.synth_nat}_{self.which_size}.pdf"
        leg_fontsize = self.tick_size - 3
        pylab.legend(
            fontsize=leg_fontsize,
            loc="upper left",
            title_fontsize=leg_fontsize,
            frameon=False,
        )
        save_name = self.output_dir + "/" + file_name
        pylab.savefig(save_name, dpi=self.dpi, format="pdf")
        pylab.close()
        print("\t\tcompleted: plotting covars")

    def make_homs(self):
        print("\t\tmaking homs")
        self.r20_folder = "r20_" + self.name
        if self.r20_mod:
            self.plot_r20_mod()
        else:
            if self.compute_homs:
                print("\t\t\tcomputing r20")
                self.compute_r20()
                print("\t\t\tcompleted: compute r20")
            if self.parse_homs:
                print("\t\t\tbeginning: parse r20")
                dir_name = self.output_dir + "/" + self.r20_folder
                print(dir_name)
                os.makedirs(dir_name, exist_ok=True)
                VHOMSH.parse_homs(
                    self.r20_folder,
                    self.output_dir,
                    int(self.r20_start),
                    int(self.r20_end),
                )
                print("\t\t\tcompleted: parse r20")
            if self.plot_homs:
                print("\t\t\tbeginning: plot r20")
                self.plot_r20_new()
                print("\t\t\tcompleted: plot r20")

        print("\n\n\t\tcompleted: making homs")

    def compute_r20(self):
        target = self.msa_dir + "/" + self.vis_seqs["targetSeqs"]
        ref = self.msa_dir + "/" + self.vis_seqs["refSeqs"]
        mi3 = self.msa_dir + "/" + self.vis_seqs["mi3Seqs"]
        vae = self.msa_dir + "/" + self.vis_seqs["svaeSeqs"]
        indep = self.msa_dir + "/" + self.vis_seqs["indepSeqs"]
        deep = self.msa_dir + "/" + self.vis_seqs["deepSeqs"]
        ref_trunc = self.msa_dir + "/" + self.vis_seqs["ref-trunc"]
        target_trunc = self.msa_dir + "/" + self.vis_seqs["target-trunc"]

        d = " "
        homs_command = " ".join([
            "python",
            self.compute_homs_script,
            str(self.pss),
            target,
            ref,
            mi3,
            vae,
            indep,
            ref_trunc,
            target_trunc,
            deep,
            self.msa_dir,
            str(self.r20_start),
            str(self.r20_end),
            self.synth_nat,
            self.output_dir,
        ])
        print(f"\t\t\tissuing homs_command: `{homs_command}`")
        try:
            # Just doing this for the side effects of the script here
            # NOTE should run the python code directly
            _output = subprocess.check_output(["bash", "-c", homs_command])
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "command '{}' return with error (code {}): {}".format(
                    e.cmd, e.returncode, e.output
                )
            )

    # synth only
    def plot_r20_mod(self):
        ms = 6
        self.line_width = self.line_width - 1
        r20_data = np.load(self.r20_data)
        r20_data = np.nanmean(r20_data, axis=1)
        fig, ax = pylab.subplots(figsize=(self.fig_size, self.fig_size))
        datasets = {"mi3": "Mi3", "vae": "vVAE", "target": "Target", "indep": "Indep"}
        n = np.arange(int(self.r20_start), int(self.r20_end))
        i = 0
        ax2 = plt.axes([0, 0, 1, 1])
        ip = InsetPosition(ax, [0.19, 0.12, 0.12, 0.35])
        ax2.set_axes_locator(ip)
        zorders = {"Target": -10, "Mi3": -20, "vVAE": -30, "Indep": -40}
        plot_limit = {"synth-vae": 2, "natural-vae": 2}
        synth_index = {"Target": 0, "Mi3": 1, "Indep": 2, "vVAE": 3}
        nat_index = {"Target": 0, "Mi3": 0, "Indep": 1, "vVAE": 2}
        if "nat" in self.synth_nat:
            index = nat_index
            black_marker = "v"
        else:
            index = synth_index
            black_marker = "o"

        for file_name, data_label in datasets.items():
            print("plotting r20 for:\t", data_label)
            if "Target" in data_label:
                if "nat" in self.synth_nat:
                    target_label = self.which_size + " 10K-Target"
                    r20_black = np.load(self.r20_data_black)
                    r20_black = np.nanmean(r20_black, axis=1)
                    target_data = r20_black[: self.r20_end - plot_limit[self.synth_nat]]
                else:
                    target_label = self.which_size + " Synth-Target"
                    target_data = r20_data[:, index[data_label]][
                        0 : self.r20_end - plot_limit[self.synth_nat]
                    ]
                ax.plot(
                    list(range(self.r20_start, self.r20_end)),
                    target_data,
                    zorder=zorders[data_label],
                    color=self.color_set[data_label],
                    linestyle="dotted",
                    linewidth=self.line_width,
                    label=target_label,
                    marker=black_marker,
                    ms=ms,
                )
                ax.errorbar(
                    list(range(self.r20_start, self.r20_end)),
                    target_data,
                    zorder=zorders[data_label],
                    color=self.color_set[data_label],
                    linestyle="dotted",
                    linewidth=self.line_width,
                    label=target_label,
                    marker=black_marker,
                    ms=ms,
                )
                ax2.plot(
                    list(range(self.r20_start, self.r20_end)),
                    target_data,
                    zorder=zorders[data_label],
                    color=self.color_set[data_label],
                    linestyle="dotted",
                    linewidth=self.line_width,
                    label=target_label,
                    marker=black_marker,
                    ms=ms - 1,
                )
            elif data_label == "Mi3":
                ax.plot(
                    list(range(self.r20_start, self.r20_end)),
                    r20_data[:, index[data_label]][
                        0 : self.r20_end - plot_limit[self.synth_nat]
                    ],
                    zorder=zorders[data_label],
                    color=self.color_set[data_label],
                    linewidth=self.line_width,
                    label=data_label,
                    marker="o",
                    ms=ms,
                )
                ax2.plot(
                    list(range(self.r20_start, self.r20_end)),
                    r20_data[:, index[data_label]][
                        0 : self.r20_end - plot_limit[self.synth_nat]
                    ],
                    zorder=zorders[data_label],
                    color=self.color_set[data_label],
                    linewidth=self.line_width,
                    label=data_label,
                    marker="o",
                    ms=ms - 1,
                )
            elif data_label == "Indep":
                ax.plot(
                    list(range(self.r20_start, self.r20_end)),
                    r20_data[:, index[data_label]][
                        0 : self.r20_end - plot_limit[self.synth_nat]
                    ],
                    zorder=zorders[data_label],
                    color=self.color_set[data_label],
                    linewidth=self.line_width,
                    label=data_label,
                    marker="o",
                    ms=ms,
                )
                ax2.plot(
                    list(range(self.r20_start, self.r20_end)),
                    r20_data[:, index[data_label]][
                        0 : self.r20_end - plot_limit[self.synth_nat]
                    ],
                    zorder=zorders[data_label],
                    color=self.color_set[data_label],
                    linewidth=self.line_width,
                    label=data_label,
                    marker="o",
                    ms=ms - 1,
                )
            else:
                ax.plot(
                    list(range(self.r20_start, self.r20_end)),
                    r20_data[:, index[data_label]][
                        0 : self.r20_end - plot_limit[self.synth_nat]
                    ],
                    zorder=zorders[data_label],
                    color=self.color_set[data_label],
                    linewidth=self.line_width,
                    label=data_label,
                    marker="o",
                    ms=ms,
                )
                ax2.plot(
                    list(range(self.r20_start, self.r20_end)),
                    r20_data[:, index[data_label]][
                        0 : self.r20_end - plot_limit[self.synth_nat]
                    ],
                    zorder=zorders[data_label],
                    color=self.color_set[data_label],
                    linewidth=self.line_width,
                    label=data_label,
                    marker="o",
                    ms=ms - 1,
                )

        ax.set_rasterization_zorder(0)
        ax2.set_rasterization_zorder(0)
        # Atitle_text = 'Average $r_{20}$ Higher Order Marginals\n' + self.name
        # pylab.title(self.which_size, fontsize=self.title_size)
        ax.set_xlabel("Higher Order Marginals", fontsize=self.label_size)
        ax.set_ylabel(r"$r_{20}$", fontsize=self.label_size)
        ax.tick_params(
            direction="in",
            axis="both",
            which="major",
            labelsize=self.tick_size,
            length=self.tick_length,
            width=self.tick_width,
        )
        ax.set_xticks(np.arange(2, 11, 1))
        ax.set_yticks(np.arange(0, 1.2, 0.2))
        ax.set_ylim(0, 1.05)
        ax2.set_xlim(1.98, 2.02)
        ax2.set_ylim(0.92, 1.005)
        ax2.tick_params(
            direction="in",
            axis="both",
            which="major",
            labelsize=self.tick_size - 3,
            length=self.tick_length - 1,
            width=self.tick_width - 0.1,
        )
        ax2.set_xticks(np.arange(2, 3, 1))
        ax2.set_yticks(np.arange(0.92, 1.02, 0.02))
        # pylab.legend(fontsize=self.label_size-2, loc=3, borderpad=self.box_padding)
        pylab.tight_layout()
        file_name = "r20_" + self.name + "_" + str(self.pss) + ".pdf"
        pylab.savefig(
            self.data_home + "/" + self.parent_dir_name + "/" + file_name,
            dpi=self.dpi,
            format="pdf",
        )

    def plot_r20_new(self):
        ms = 4
        self.line_width = self.line_width - 1
        cwd = os.getcwd()
        new_cwd = self.output_dir + "/" + self.r20_folder
        # os.chdir(new_cwd)          # automated old
        os.chdir(self.output_dir + "/r20_natcoms_alphaFix")
        fig, ax = pylab.subplots(figsize=(self.fig_size, self.fig_size))
        datasets = {"Mi3": 1, "sVAE": 2, "Target": 0, "Indep": 4, "DeepSeq": 3}
        n = np.arange(int(self.r20_start), int(self.r20_end))
        i = 0
        ax2 = plt.axes([0, 0, 1, 1])
        ip = InsetPosition(ax, [0.19, 0.12, 0.12, 0.35])
        ax2.set_axes_locator(ip)
        # mark_inset(ax, ax2, loc1=2, loc2=1, fc="none", ec='0.5')
        if "nat" in self.synth_nat:
            black_marker = "v"
        else:
            black_marker = "o"

        data = np.load("/home/tuk31788/vvae/natcom/r20_nat_avgs.npy")
        for data_label, index in datasets.items():
            print("plotting r20 for:\t", data_label)
            m = data[datasets[data_label]]

            if data_label == "Target":
                if "nat" in self.synth_nat:
                    target_label = "Nat-Target"
                else:
                    target_label = "Synth-Target"
                # ax.errorbar(n, m, zorder=self.z_order[data_label], color=self.color_set[data_label], yerr=[m-lo, hi-m], linestyle="dotted", linewidth=self.line_width, fmt='.-', label=target_label, marker=black_marker, ms=ms-5)
                ax.plot(
                    n,
                    m,
                    zorder=self.z_order[data_label],
                    color=self.color_set[data_label],
                    linestyle="dotted",
                    linewidth=self.line_width,
                    label=target_label,
                    marker=black_marker,
                    ms=ms,
                )
                ax2.plot(
                    n,
                    m,
                    zorder=self.z_order[data_label],
                    color=self.color_set[data_label],
                    linestyle="dotted",
                    linewidth=self.line_width,
                    label=target_label,
                    marker=black_marker,
                    ms=ms,
                )
            else:
                # ax.errorbar(n, m, zorder=self.z_order[data_label], color=self.color_set[data_label], yerr=[m-lo, hi-m], linewidth=self.line_width, fmt='.-', label=data_label, capthick=0.4, ms=ms)
                ax.plot(
                    n,
                    m,
                    zorder=self.z_order[data_label],
                    color=self.color_set[data_label],
                    linewidth=self.line_width,
                    marker="o",
                    ms=ms,
                    label=data_label,
                )
                ax2.plot(
                    n,
                    m,
                    zorder=self.z_order[data_label],
                    color=self.color_set[data_label],
                    linewidth=self.line_width,
                    marker="o",
                    ms=ms,
                    label=data_label,
                )
                # pylab.errorbar(n, m, color=self.color_set[data_label], yerr=[m-lo, hi-m], fmt='.-', label=data_label, capthick=0.4, zorder=z_orders[data_label])

        ax.set_rasterization_zorder(0)
        ax2.set_rasterization_zorder(0)
        # Atitle_text = 'Average $r_{20}$ Higher Order Marginals\n' + self.name
        # pylab.title(self.which_size, fontsize=self.title_size)
        ax.set_xlabel("Higher Order Marginals", fontsize=self.label_size)
        ax.set_ylabel(r"$r_{20}$", fontsize=self.label_size)
        ax.tick_params(
            direction="in",
            axis="both",
            which="major",
            labelsize=self.tick_size,
            length=self.tick_length,
            width=self.tick_width,
        )
        ax.set_xticks(np.arange(2, 11, 1))
        ax.set_yticks(np.arange(0, 1.2, 0.2))
        ax.set_ylim(0, 1.05)
        ax2.set_xlim(1.98, 2.02)
        ax2.set_ylim(0.92, 1.005)
        ax2.tick_params(
            direction="in",
            axis="both",
            which="major",
            labelsize=self.tick_size - 3,
            length=self.tick_length - 1,
            width=self.tick_width - 0.1,
        )
        ax2.set_xticks(np.arange(2, 3, 1))
        ax2.set_yticks(np.arange(0.92, 1.02, 0.02))
        pylab.legend(
            fontsize=self.tick_size - 3, loc=1, bbox_to_anchor=(6.5, 2.5), frameon=False
        )
        pylab.tight_layout()
        file_name = (
            "r20_"
            + self.name
            + "_"
            + str(self.pss)
            + "_"
            + self.synth_nat
            + "_"
            + self.which_size
            + ".pdf"
        )
        pylab.savefig(self.output_dir + "/" + file_name, dpi=self.dpi, format="pdf")
        os.chdir(cwd)

    def plot_r20(self):
        ms = 8
        self.line_width = self.line_width - 1
        cwd = os.getcwd()
        new_cwd = self.output_dir + "/" + self.r20_folder
        os.chdir(new_cwd)
        fig, ax = pylab.subplots(figsize=(self.fig_size, self.fig_size))
        datasets = {
            "mi3": "Mi3",
            "vae": "vVAE",
            "target": "Target",
            "indep": "Indep",
            "deep": "DeepSeq",
        }
        n = np.arange(int(self.r20_start), int(self.r20_end))
        i = 0
        ax2 = plt.axes([0, 0, 1, 1])
        ip = InsetPosition(ax, [0.19, 0.12, 0.12, 0.35])
        ax2.set_axes_locator(ip)
        # mark_inset(ax, ax2, loc1=2, loc2=1, fc="none", ec='0.5')
        if "nat" in self.synth_nat:
            black_marker = "v"
        else:
            black_marker = "o"

        for file_name, data_label in datasets.items():
            print("plotting r20 for:\t", data_label)
            i += 1
            d = [np.loadtxt("{}_{}".format(file_name, i)) for i in n]
            m = np.array([np.mean(di) for di in d])
            lo = np.array([np.percentile(di, 25) for di in d])
            hi = np.array([np.percentile(di, 75) for di in d])

            if data_label == "Target":
                if "nat" in self.synth_nat:
                    target_label = "Nat-Target"
                else:
                    target_label = "Synth-Target"
                # ax.errorbar(n, m, zorder=self.z_order[data_label], color=self.color_set[data_label], yerr=[m-lo, hi-m], linestyle="dotted", linewidth=self.line_width, fmt='.-', label=target_label, marker=black_marker, ms=ms-5)
                ax.errorbar(
                    n,
                    m,
                    zorder=self.z_order[data_label],
                    color=self.color_set[data_label],
                    yerr=[m - lo, hi - m],
                    linestyle="dotted",
                    linewidth=self.line_width,
                    fmt=".-",
                    label=target_label,
                    marker=black_marker,
                    ms=ms,
                    capsize=2,
                    elinewidth=0.5,
                    alpha=0.7,
                )
                ax2.errorbar(
                    n,
                    m,
                    zorder=self.z_order[data_label],
                    color=self.color_set[data_label],
                    yerr=[m - lo, hi - m],
                    linestyle="dotted",
                    linewidth=self.line_width,
                    fmt=".-",
                    label=target_label,
                    marker=black_marker,
                    ms=ms,
                )[-1][0].set_linewidth(0)
            else:
                # ax.errorbar(n, m, zorder=self.z_order[data_label], color=self.color_set[data_label], yerr=[m-lo, hi-m], linewidth=self.line_width, fmt='.-', label=data_label, capthick=0.4, ms=ms)
                ax.errorbar(
                    n,
                    m,
                    zorder=self.z_order[data_label],
                    color=self.color_set[data_label],
                    yerr=[m - lo, hi - m],
                    linewidth=self.line_width,
                    fmt=".-",
                    label=data_label,
                    capthick=0.4,
                    ms=ms,
                    capsize=2,
                    elinewidth=0.5,
                    alpha=0.8,
                )
                ax2.errorbar(
                    n,
                    m,
                    zorder=self.z_order[data_label],
                    color=self.color_set[data_label],
                    yerr=[m - lo, hi - m],
                    linewidth=self.line_width,
                    fmt=".-",
                    label=data_label,
                    capthick=0.4,
                    ms=ms,
                )[-1][0].set_linewidth(0)
                # pylab.errorbar(n, m, color=self.color_set[data_label], yerr=[m-lo, hi-m], fmt='.-', label=data_label, capthick=0.4, zorder=z_orders[data_label])

        ax.set_rasterization_zorder(0)
        ax2.set_rasterization_zorder(0)
        # Atitle_text = 'Average $r_{20}$ Higher Order Marginals\n' + self.name
        # pylab.title(self.which_size, fontsize=self.title_size)
        ax.set_xlabel("Higher Order Marginals", fontsize=self.label_size)
        ax.set_ylabel(r"$r_{20}$", fontsize=self.label_size)
        ax.tick_params(
            direction="in",
            axis="both",
            which="major",
            labelsize=self.tick_size,
            length=self.tick_length,
            width=self.tick_width,
        )
        ax.set_xticks(np.arange(2, 11, 1))
        ax.set_yticks(np.arange(0, 1.2, 0.2))
        ax.set_ylim(0, 1.05)
        ax2.set_xlim(1.98, 2.02)
        ax2.set_ylim(0.92, 1.005)
        ax2.tick_params(
            direction="in",
            axis="both",
            which="major",
            labelsize=self.tick_size - 3,
            length=self.tick_length - 1,
            width=self.tick_width - 0.1,
        )
        ax2.set_xticks(np.arange(2, 3, 1))
        ax2.set_yticks(np.arange(0.92, 1.02, 0.02))
        pylab.legend(
            fontsize=self.label_size - 5,
            loc=1,
            bbox_to_anchor=(6.5, 2.5),
            frameon=False,
        )
        pylab.tight_layout()
        file_name = (
            "r20_"
            + self.name
            + "_"
            + str(self.pss)
            + "_"
            + self.synth_nat
            + "_"
            + self.which_size
            + ".pdf"
        )
        pylab.savefig(self.output_dir + "/" + file_name, dpi=self.dpi, format="pdf")
        os.chdir(cwd)

    def make_energies(self):
        print("\t\tmaking energies")
        data = np.load(self.data_home + "/" + self.energies_file)
        if "M" not in self.which_size:
            indep_energies = data["i10K"]
            mi3_energies = data["p10K"]
            svae_energies = -data["v10K"]
        else:
            indep_energies = data["i1M"]
            mi3_energies = data["p1M"]
            svae_energies = -data["v1M"]

        self.plot_energies(data["ref"], indep_energies, "Indep")
        self.plot_energies(data["ref"], mi3_energies, "Mi3")
        self.plot_energies(data["ref"], svae_energies, "sVAE")

        print("\t\tcompleted energies")

    def plot_energies(self, target_energies, gpsm_energies, label):
        print("\t\t\tplotting " + label)
        fig, ax = pylab.subplots(figsize=(self.fig_size, self.fig_size))
        c = self.color_set[label]
        x_start = min(target_energies)
        x_end = max(target_energies)
        y_start = min(gpsm_energies)
        y_end = max(gpsm_energies)
        x_tick_range = np.arange(-725, -375, 50)

        if label == "Mi3":
            y_tick_range = np.arange(-700, -300, 100)
            title_text = "A"
        if label == "Indep":
            y_tick_range = np.arange(400, 700, 50)
            title_text = "C"
        if label == "vVAE":
            y_tick_range = np.arange(300, 650, 50)
            title_text = "B"

        pearson_r, pearson_p = pearsonr(target_energies, gpsm_energies)
        # text = "Pearson R: " + str(round(pearson_r, 3)) + " at p-value = " + str(round(pearson_p, 3))
        text = (
            self.which_size
            + " "
            + label
            + r", $\rho$ = "
            + str(round(pearson_r, self.stats_sf))
        )
        print(text)
        pylab.xlabel("Synth-Target Energy", fontsize=self.label_size)
        pylab.ylabel(
            self.which_size + " " + label + " Energy", fontsize=self.label_size
        )
        file_name = "loss-energy_target-" + label + "_" + self.name + ".pdf"

        # plot the data
        ax.plot(
            target_energies,
            gpsm_energies,
            ".",
            color=c,
            alpha=0.3,
            label=text,
            zorder=-10,
        )
        ax.plot([x_start, x_end], [y_start, y_end], "-k", linewidth=0.5, alpha=0.7)
        ax.set_rasterization_zorder(0)

        locs, labels = plt.xticks()
        pylab.xticks(x_tick_range, rotation=45, fontsize=self.tick_size)
        pylab.yticks(y_tick_range, fontsize=self.tick_size)
        pylab.tick_params(
            direction="in",
            axis="both",
            which="major",
            labelsize=self.tick_size,
            length=self.tick_length,
            width=self.tick_width,
        )
        pylab.legend(fontsize=self.label_size - 3, loc="upper left", frameon=False)
        pylab.tight_layout()
        pylab.savefig(
            self.data_home + "/" + self.parent_dir_name + "/" + file_name,
            dpi=self.dpi,
            format="pdf",
        )
        pylab.close()
        print("\t\t\tcompleted plotting " + label)