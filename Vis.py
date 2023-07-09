import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import numpy as np
# import VisEnergiesHelper VEH
from scipy.stats import pearsonr

from mi3gpu.utils.seqload import loadSeqs
from mi3gpu.utils.seqtools import histsim
import VisHOMSHelper as VHOMSH
from config import Config

def norm_x(hams, freqs):
    y_max = max(freqs)
    index = freqs.index(y_max)
    x_max = hams[index]
    return [np.log(x) - np.log(x_max) for x in hams]

def norm_y(freqs):
    y_max = max(freqs)
    return [np.log(y) - np.log(y_max) for y in freqs]

class Vis:
    config: Config

    def __init__(self, config=None):
        self.config

    def plot_hams_loglog(self):

        fig_size = self.config.fig_size
        vis_seqs = self.config.vis_seqs
        skip = self.config.skip
        which_models = self.config.which_models
        label_dict = self.config.label_dict
        msa_dir = self.config.msa_dir
        ALPHA = self.config.ALPHA
        keep_hams = self.config.keep_hams
        synth_nat = self.config.synth_nat
        line_width = self.config.line_width
        line_alpha = self.config.line_alpha
        color_set = self.config.color_set
        z_order = self.config.z_order
        label_size = self.config.label_size
        tick_size = self.config.tick_size
        tick_length = self.config.tick_length
        tick_width = self.config.tick_width
        name = self.config.name
        which_size = self.config.which_size
        output_dir = self.config.output_dir
        dpi = self.config.dpi

        print("MAKING LOGLOG HAMS")
        # make list of labels, filenames
        fig, ax = pylab.subplots(figsize=(fig_size, fig_size))
        xlabel = r"ln($d$/$d_{Mo}$)"
        ylabel = r"ln($f$/$f_{max}$)"

        for label, seqs_file in vis_seqs.items():
            if label in skip:
                print("skipping ", label)
                continue
            if not which_models[
                label
            ]:  # model is 'false' in the which_models{}, then continue
                continue
            label = label_dict[label]
            seqs_path = msa_dir / seqs_file
            print("computing hams for:\t", label, f"\t\t\tin:\t{seqs_path}")
            seqs = loadSeqs(msa_dir / seqs_file, names=ALPHA)[0][
                0 : keep_hams
            ]
            h = histsim(seqs).astype(float)[::-1][1:].tolist()
            hams = np.arange(1, len(h) + 1, 1)

            d_norm_x = norm_x(hams, h)
            d_norm_y = norm_y(h)

            line_style = "solid"
            if label == "Target":
                if "nat" in synth_nat:
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
                    linewidth=line_width,
                    dashes=my_dashes,
                    alpha=line_alpha,
                    color=color_set[label],
                    label=target_label,
                    zorder=z_order[label],
                )
            else:
                ax.plot(
                    d_norm_x,
                    d_norm_y,
                    linestyle=line_style,
                    linewidth=line_width,
                    alpha=line_alpha,
                    color=color_set[label],
                    label=label,
                    zorder=z_order[label],
                )

        pylab.ylim(-8, 1)
        pylab.xlim(-0.5, 0.26)
        pylab.ylabel(ylabel, fontsize=label_size - 1)
        pylab.xlabel(xlabel, fontsize=label_size - 1)
        x_tick_range = np.arange(-0.5, 0.5, 0.25)
        y_tick_range = np.arange(-8, 1, 2)
        pylab.xticks(x_tick_range, rotation=45)
        pylab.yticks(y_tick_range)
        pylab.tick_params(
            direction="in",
            axis="both",
            which="major",
            labelsize=tick_size,
            length=tick_length,
            width=tick_width,
        )
        file_name = f"loglog_ham_{name}_{synth_nat}_{which_size}.pdf"
        pylab.tight_layout()
        pylab.legend(fontsize=tick_size - 3, loc="best", frameon=False)
        save_name = output_dir / file_name
        pylab.savefig(save_name, dpi=dpi, format="pdf")
        pylab.close()

    def plot_hams(self):
        fig_size = self.config.fig_size
        protein = self.config.protein
        vis_seqs = self.config.vis_seqs
        skip = self.config.vis_seqs
        which_models = self.config.which_models
        label_dict = self.config.label_dict
        msa_dir = self.config.msa_dir
        ALPHA = self.config.ALPHA
        keep_hams = self.config.keep_hams
        synth_nat = self.config.synth_nat
        line_width = self.config.line_width
        line_alpha = self.config.line_alpha
        color_set = self.config.color_set
        z_order = self.config.z_order
        label_size = self.config.label_size
        tick_size = self.config.tick_size
        tick_width = self.config.tick_width
        tick_length = self.config.tick_length
        name = self.config.name
        which_size = self.config.which_size
        output_dir = self.config.output_dir
        dpi = self.config.dpi

        print("plotting normal hams")
        _fig, ax = pylab.subplots(figsize=(fig_size, fig_size))

        # axes labels
        xlabel = r"$d$"
        ylabel = "f"
        if protein == "Kinase":
            start = 120
            end = 230
            x_tick_range = np.arange(start, end, 20)
            pylab.xlim(start, end)
        else:
            # FIXME
            raise NotImplementedError("Only Kinase is supported")

        all_freqs = dict()

        for label, seqs_file in vis_seqs.items():
            if label in skip:
                print("skipping ", label)
                continue
            if not which_models[
                label
            ]:  # model is 'false' in the which_models{}, then continue
                continue
            label = label_dict[label]
            print("computing hams for:\t", label)
            seqs = loadSeqs(msa_dir / seqs_file, names=ALPHA)[0][
                0 : keep_hams
            ]
            h = histsim(seqs).astype(float)
            h = h / np.sum(h)
            all_freqs[label] = h
            rev_h = h[::-1]
            if label == "Target":
                if "nat" in synth_nat:
                    target_label = "Nat-Target"
                else:
                    target_label = "Synth-Target"
                line_style = "dashed"
                my_dashes = (1, 1)
                ax.plot(
                    rev_h,
                    linestyle=line_style,
                    linewidth=line_width,
                    dashes=my_dashes,
                    alpha=line_alpha,
                    color=color_set[label],
                    label=target_label,
                    zorder=z_order[label],
                )
            else:
                line_style = "solid"
                ax.plot(
                    rev_h,
                    linestyle=line_style,
                    linewidth=line_width,
                    alpha=line_alpha,
                    color=color_set[label],
                    label=label,
                    zorder=z_order[label],
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
        pylab.ylabel(ylabel, fontsize=label_size)
        pylab.xlabel(xlabel, fontsize=label_size)
        pylab.xticks(x_tick_range, rotation=45)
        pylab.yticks(y_tick_range)
        pylab.tick_params(
            direction="in",
            axis="both",
            which="major",
            labelsize=tick_size,
            length=tick_length,
            width=tick_width,
        )
        file_name = f"ham_{name}_{synth_nat}_{which_size}.pdf"
        pylab.tight_layout()
        pylab.legend(fontsize=tick_size - 3, loc="upper left", frameon=False)
        save_name = output_dir / file_name
        print(save_name)
        pylab.savefig(save_name, dpi=dpi, format="pdf")
        pylab.close()

    def make_covars(self):
        compute_all_bvms = self.config.compute_all_bvms
        bvms = self.config.bvms
        skip = self.config.skip
        vis_seqs = self.config.vis_seqs
        compute_all_covars = self.config.compute_all_covars
        covars = self.config.covars

        print("\t\tmaking covars")
        if compute_all_bvms:
            print("computing all bvms")
            for label in bvms.keys():
                if label not in skip:
                    msa_file = vis_seqs[label]
                    self.get_bvms(label, msa_file)
        if compute_all_covars:
            print("computing all covars")
            for label in covars.keys():
                if label not in skip:
                    bvms_file = bvms[label]
                    self.get_covars(label, bvms_file)

        print("\t\t\tbvms:")
        print(f"\t\t\t{bvms}")
        print("\t\t\tcovars:\t")
        print(f"\t\t\t{covars}")
        print("\n\t\tplotting covars")
        self.plot_covars()
        print("\t\tcompleted: making covars")

    def get_bvms(self, label: str, msa_file: Path):
        dest = self.config.output_dir
        source = self.config.msa_dir
        bvms = self.config.bvms
        bvms_script = self.config.bvms_script
        A = self.config.A
        keep_covars = self.config.keep_covars

        print("bvms for: ", bvms[label])

        bvms_command = f"python {bvms_script} {label} {msa_file} {source} {dest} {A} {keep_covars}"

        print(f"\t\t\tgetting bvms for:\t{label}")
        print(bvms_command)

        return
        try:
            output = subprocess.check_output(["bash", "-c", bvms_command])
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "command '{}' return with error (code {}): {}".format(
                    e.cmd, e.returncode, e.output
                )
            )

    def get_covars(self, label, bvms_file):
        covars = self.config.covars
        output_dir = self.config.output_dir
        covars_script = self.config.covars_script

        print("covars for: ", covars[label])
        path = output_dir
        covars_command = f"python {covars_script} {label} {bvms_file} {path}"

        print("\t\t\tgetting covars for:\t", label)
        print(covars_command)
        
        return
        try:
            _output = subprocess.check_output(["bash", "-c", covars_command])
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"command '{e.cmd}' return with error (code {e.returncode}): {e.output}"
            )

        print(f"completed covars for:\t{covars[label]}")

    def plot_covars(self):
        fig_size = self.config.fig_size
        marker_size = self.config.marker_size
        output_dir = self.config.output_dir
        covars = self.config.covars
        color_set = self.config.color_set
        stats_sf = self.config.stats_sf
        z_order = self.config.z_order
        synth_nat = self.config.synth_nat
        label_size = self.config.label_size
        tick_rotation = self.config.tick_rotation
        tick_size = self.config.tick_size
        tick_width = self.config.tick_width
        tick_length = self.config.tick_length
        name = self.config.name
        keep_covars = self.config.keep_covars
        synth_nat = self.config.synth_nat
        which_size = self.config.which_size
        dpi = self.config.dpi

        print("\t\t\t\tplotting covars:\t")
        _fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        marker_size = marker_size - 4
        start = -0.10
        end = 0.15
        x_tick_range = np.arange(start, end, 0.05)
        y_tick_range = np.arange(start, end, 0.05)
        target_covars = np.load(output_dir / covars["targetSeqs"])
        target_masked = np.ma.masked_inside(target_covars, -0.01, 0.01).ravel()
        target_covars = target_covars.ravel()
        mi3_covars = np.load(output_dir / covars["mi3Seqs"]).ravel()
        indep_covars = np.load(output_dir / covars["indepSeqs"]).ravel()
        svae_covars = np.load(output_dir / covars["svaeSeqs"]).ravel()
        deepsequence_covars = np.load(output_dir / covars["deepSeqs"]).ravel()
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
            c = color_set[label]
            print(pearson_r, pearson_p)
            label_text = label
            label_text = (
                label + ", " + r"$\rho$ = " + str(round(pearson_r, stats_sf))
            )  # orig with rho
            # plt.plot(target_masked, covars, 'o', markersize=marker_size, color=c, alpha=self.alpha, label=label_text)
            ax.plot(
                target_masked,
                covars,
                "o",
                markersize=marker_size,
                color=c,
                label=label_text,
                zorder=z_order[label],
                alpha=0.9,
            )

        ax.set_rasterization_zorder(0)
        if "nat" in synth_nat:
            xlabel = "Nat-Target Covariances"
        else:
            xlabel = "Synth-Target Covariances"
        pylab.xlabel(xlabel, fontsize=label_size)
        pylab.ylabel("GPSM Covariances", fontsize=label_size)
        pylab.xticks(x_tick_range, rotation=tick_rotation, fontsize=tick_size)
        pylab.yticks(y_tick_range, fontsize=tick_size)
        pylab.tick_params(
            direction="in",
            axis="both",
            which="major",
            labelsize=tick_size,
            length=tick_length,
            width=tick_width,
        )
        lim_start = -0.12
        lim_end = 0.12
        pylab.xlim((lim_start, lim_end))
        pylab.ylim((lim_start, lim_end))
        pylab.tight_layout()
        file_name = f"covars_{name}_{keep_covars}_{synth_nat}_{which_size}.pdf"
        leg_fontsize = tick_size - 3
        pylab.legend(
            fontsize=leg_fontsize,
            loc="upper left",
            title_fontsize=leg_fontsize,
            frameon=False,
        )
        save_name = output_dir / file_name
        pylab.savefig(save_name, dpi=dpi, format="pdf")
        pylab.close()
        print("\t\tcompleted: plotting covars")

    def make_homs(self):
        r20_folder = self.config.r20_folder
        name = self.config.name
        r20_mod = self.config.r20_mod
        compute_homs = self.config.compute_homs
        parse_homs = self.config.parse_homs
        output_dir = self.config.output_dir
        r20_start = self.config.r20_start
        r20_end = self.config.r20_end
        plot_homs = self.config.plot_homs

        print("\t\tmaking homs")
        r20_folder = "r20_" + name # XXX doesn't seem correct, was originally reassigning the config var
        if r20_mod:
            self.plot_r20_mod()
        else:
            if compute_homs:
                print("\t\t\tcomputing r20")
                self.compute_r20()
                print("\t\t\tcompleted: compute r20")
            if parse_homs:
                print("\t\t\tbeginning: parse r20")
                dir_name = output_dir / r20_folder
                print(dir_name)
                os.makedirs(dir_name, exist_ok=True)
                VHOMSH.parse_homs(
                    r20_folder,
                    output_dir,
                    int(r20_start),
                    int(r20_end),
                )
                print("\t\t\tcompleted: parse r20")
            if plot_homs:
                print("\t\t\tbeginning: plot r20")
                self.plot_r20_new()
                print("\t\t\tcompleted: plot r20")

        print("\n\n\t\tcompleted: making homs")

    def compute_r20(self):
        msa_dir = self.config.msa_dir
        vis_seqs = self.config.vis_seqs
        compute_homs_script = self.config.compute_homs_script
        pss = self.config.pss
        r20_start = self.config.r20_start
        r20_end = self.config.r20_end
        synth_nat = self.config.synth_nat
        output_dir = self.config.output_dir

        target = msa_dir / vis_seqs["targetSeqs"]
        ref = msa_dir / vis_seqs["refSeqs"]
        mi3 = msa_dir / vis_seqs["mi3Seqs"]
        vae = msa_dir / vis_seqs["svaeSeqs"]
        indep = msa_dir / vis_seqs["indepSeqs"]
        deep = msa_dir / vis_seqs["deepSeqs"]
        ref_trunc = msa_dir / vis_seqs["ref-trunc"]
        target_trunc = msa_dir / vis_seqs["target-trunc"]

        homs_command = f"python {compute_homs_script} {pss} {target} {ref} {mi3} {vae} {indep} {ref_trunc} {target_trunc} {deep} {msa_dir} {r20_start} {r20_end} {synth_nat} {output_dir}"
        print(f"\t\t\tissuing homs_command: `{homs_command}`")
        return 
        try:
            _output = subprocess.check_output(["bash", "-c", homs_command])
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "command '{}' return with error (code {}): {}".format(
                    e.cmd, e.returncode, e.output
                )
            )

    # synth only
    def plot_r20_mod(self):
        line_width = self.config.line_width
        r20_data = self.config.r20_data
        fig_size = self.config.fig_size
        r20_start = self.config.r20_start
        r20_end = self.config.r20_end
        synth_nat = self.config.synth_nat
        which_size = self.config.which_size
        r20_data_black = self.config.r20_data_black
        color_set = self.config.color_set
        line_width = self.config.line_width
        label_size = self.config.label_size
        tick_size = self.config.tick_size
        tick_width = self.config.tick_width
        tick_length = self.config.tick_length
        name = self.config.name
        pss = self.config.pss
        parent_dir_name = self.config.parent_dir_name
        dpi = self.config.dpi
        msa_dir = self.config.msa_dir

        ms = 6
        line_width = line_width - 1
        r20_data = np.load(r20_data)
        r20_data = np.nanmean(r20_data, axis=1)
        fig, ax = pylab.subplots(figsize=(fig_size, fig_size))
        datasets = {"mi3": "Mi3", "vae": "vVAE", "target": "Target", "indep": "Indep"}
        n = np.arange(int(r20_start), int(r20_end))
        i = 0
        ax2 = plt.axes((0, 0, 1, 1))
        ip = InsetPosition(ax, [0.19, 0.12, 0.12, 0.35])
        ax2.set_axes_locator(ip)
        zorders = {"Target": -10, "Mi3": -20, "vVAE": -30, "Indep": -40}
        plot_limit = {"synth-vae": 2, "natural-vae": 2}
        synth_index = {"Target": 0, "Mi3": 1, "Indep": 2, "vVAE": 3}
        nat_index = {"Target": 0, "Mi3": 0, "Indep": 1, "vVAE": 2}
        if "nat" in synth_nat:
            index = nat_index
            black_marker = "v"
        else:
            index = synth_index
            black_marker = "o"

        for file_name, data_label in datasets.items():
            print("plotting r20 for:\t", data_label)
            if "Target" in data_label:
                if "nat" in synth_nat:
                    target_label = which_size + " 10K-Target"
                    r20_black = np.load(r20_data_black)
                    r20_black = np.nanmean(r20_black, axis=1)
                    target_data = r20_black[: r20_end - plot_limit[synth_nat]]
                else:
                    target_label = which_size + " Synth-Target"
                    target_data = r20_data[:, index[data_label]][
                        0 : r20_end - plot_limit[synth_nat]
                    ]
                ax.plot(
                    list(range(r20_start, r20_end)),
                    target_data,
                    zorder=zorders[data_label],
                    color=color_set[data_label],
                    linestyle="dotted",
                    linewidth=line_width,
                    label=target_label,
                    marker=black_marker,
                    ms=ms,
                )
                ax.errorbar(
                    list(range(r20_start, r20_end)),
                    target_data,
                    zorder=zorders[data_label],
                    color=color_set[data_label],
                    linestyle="dotted",
                    linewidth=line_width,
                    label=target_label,
                    marker=black_marker,
                    ms=ms,
                )
                ax2.plot(
                    list(range(r20_start, r20_end)),
                    target_data,
                    zorder=zorders[data_label],
                    color=color_set[data_label],
                    linestyle="dotted",
                    linewidth=line_width,
                    label=target_label,
                    marker=black_marker,
                    ms=ms - 1,
                )
            elif data_label == "Mi3":
                ax.plot(
                    list(range(r20_start, r20_end)),
                    r20_data[:, index[data_label]][
                        0 : r20_end - plot_limit[synth_nat]
                    ],
                    zorder=zorders[data_label],
                    color=color_set[data_label],
                    linewidth=line_width,
                    label=data_label,
                    marker="o",
                    ms=ms,
                )
                ax2.plot(
                    list(range(r20_start, r20_end)),
                    r20_data[:, index[data_label]][
                        0 : r20_end - plot_limit[synth_nat]
                    ],
                    zorder=zorders[data_label],
                    color=color_set[data_label],
                    linewidth=line_width,
                    label=data_label,
                    marker="o",
                    ms=ms - 1,
                )
            elif data_label == "Indep":
                ax.plot(
                    list(range(r20_start, r20_end)),
                    r20_data[:, index[data_label]][
                        0 : r20_end - plot_limit[synth_nat]
                    ],
                    zorder=zorders[data_label],
                    color=color_set[data_label],
                    linewidth=line_width,
                    label=data_label,
                    marker="o",
                    ms=ms,
                )
                ax2.plot(
                    list(range(r20_start, r20_end)),
                    r20_data[:, index[data_label]][
                        0 : r20_end - plot_limit[synth_nat]
                    ],
                    zorder=zorders[data_label],
                    color=color_set[data_label],
                    linewidth=line_width,
                    label=data_label,
                    marker="o",
                    ms=ms - 1,
                )
            else:
                ax.plot(
                    list(range(r20_start, r20_end)),
                    r20_data[:, index[data_label]][
                        0 : r20_end - plot_limit[synth_nat]
                    ],
                    zorder=zorders[data_label],
                    color=color_set[data_label],
                    linewidth=line_width,
                    label=data_label,
                    marker="o",
                    ms=ms,
                )
                ax2.plot(
                    list(range(r20_start, r20_end)),
                    r20_data[:, index[data_label]][
                        0 : r20_end - plot_limit[synth_nat]
                    ],
                    zorder=zorders[data_label],
                    color=color_set[data_label],
                    linewidth=line_width,
                    label=data_label,
                    marker="o",
                    ms=ms - 1,
                )

        ax.set_rasterization_zorder(0)
        ax2.set_rasterization_zorder(0)
        # Atitle_text = 'Average $r_{20}$ Higher Order Marginals\n' + self.name
        # pylab.title(self.which_size, fontsize=self.title_size)
        ax.set_xlabel("Higher Order Marginals", fontsize=label_size)
        ax.set_ylabel(r"$r_{20}$", fontsize=label_size)
        ax.tick_params(
            direction="in",
            axis="both",
            which="major",
            labelsize=tick_size,
            length=tick_length,
            width=tick_width,
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
            labelsize=tick_size - 3,
            length=tick_length - 1,
            width=tick_width - 0.1,
        )
        ax2.set_xticks(np.arange(2, 3, 1))
        ax2.set_yticks(np.arange(0.92, 1.02, 0.02))
        # pylab.legend(fontsize=self.label_size-2, loc=3, borderpad=self.box_padding)
        pylab.tight_layout()
        file_name = "r20_" + name + "_" + str(pss) + ".pdf"
        pylab.savefig(
            msa_dir / parent_dir_name / file_name,
            dpi=dpi,
            format="pdf",
        )

    def plot_r20_new(self):
        output_dir = self.config.output_dir
        line_width = self.config.line_width
        r20_folder = self.config.r20_folder
        fig_size = self.config.fig_size
        r20_start = self.config.r20_start
        r20_end = self.config.r20_end
        synth_nat = self.config.synth_nat
        z_order = self.config.z_order
        color_set = self.config.color_set
        label_size = self.config.label_size
        tick_size = self.config.tick_size
        tick_width = self.config.tick_width
        tick_length = self.config.tick_length
        name = self.config.name
        pss = self.config.pss
        which_size = self.config.which_size
        dpi = self.config.dpi

        ms = 4
        line_width = line_width - 1
        cwd = os.getcwd()
        new_cwd = output_dir / r20_folder
        # os.chdir(new_cwd)          # automated old
        os.chdir(output_dir / "r20_natcoms_alphaFix")
        fig, ax = pylab.subplots(figsize=(fig_size, fig_size))
        datasets = {"Mi3": 1, "sVAE": 2, "Target": 0, "Indep": 4, "DeepSeq": 3}
        n = np.arange(int(r20_start), int(r20_end))
        i = 0
        ax2 = plt.axes((0, 0, 1, 1))
        ip = InsetPosition(ax, [0.19, 0.12, 0.12, 0.35])
        ax2.set_axes_locator(ip)
        # mark_inset(ax, ax2, loc1=2, loc2=1, fc="none", ec='0.5')
        if "nat" in synth_nat:
            black_marker = "v"
        else:
            black_marker = "o"

        data = np.load("/home/tuk31788/vvae/natcom/r20_nat_avgs.npy")
        for data_label, index in datasets.items():
            print("plotting r20 for:\t", data_label)
            m = data[datasets[data_label]]

            if data_label == "Target":
                if "nat" in synth_nat:
                    target_label = "Nat-Target"
                else:
                    target_label = "Synth-Target"
                # ax.errorbar(n, m, zorder=self.z_order[data_label], color=self.color_set[data_label], yerr=[m-lo, hi-m], linestyle="dotted", linewidth=self.line_width, fmt='.-', label=target_label, marker=black_marker, ms=ms-5)
                ax.plot(
                    n,
                    m,
                    zorder=z_order[data_label],
                    color=color_set[data_label],
                    linestyle="dotted",
                    linewidth=line_width,
                    label=target_label,
                    marker=black_marker,
                    ms=ms,
                )
                ax2.plot(
                    n,
                    m,
                    zorder=z_order[data_label],
                    color=color_set[data_label],
                    linestyle="dotted",
                    linewidth=line_width,
                    label=target_label,
                    marker=black_marker,
                    ms=ms,
                )
            else:
                # ax.errorbar(n, m, zorder=self.z_order[data_label], color=self.color_set[data_label], yerr=[m-lo, hi-m], linewidth=self.line_width, fmt='.-', label=data_label, capthick=0.4, ms=ms)
                ax.plot(
                    n,
                    m,
                    zorder=z_order[data_label],
                    color=color_set[data_label],
                    linewidth=line_width,
                    marker="o",
                    ms=ms,
                    label=data_label,
                )
                ax2.plot(
                    n,
                    m,
                    zorder=z_order[data_label],
                    color=color_set[data_label],
                    linewidth=line_width,
                    marker="o",
                    ms=ms,
                    label=data_label,
                )
                # pylab.errorbar(n, m, color=self.color_set[data_label], yerr=[m-lo, hi-m], fmt='.-', label=data_label, capthick=0.4, zorder=z_orders[data_label])

        ax.set_rasterization_zorder(0)
        ax2.set_rasterization_zorder(0)
        # Atitle_text = 'Average $r_{20}$ Higher Order Marginals\n' + self.name
        # pylab.title(self.which_size, fontsize=self.title_size)
        ax.set_xlabel("Higher Order Marginals", fontsize=label_size)
        ax.set_ylabel(r"$r_{20}$", fontsize=label_size)
        ax.tick_params(
            direction="in",
            axis="both",
            which="major",
            labelsize=tick_size,
            length=tick_length,
            width=tick_width,
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
            labelsize=tick_size - 3,
            length=tick_length - 1,
            width=tick_width - 0.1,
        )
        ax2.set_xticks(np.arange(2, 3, 1))
        ax2.set_yticks(np.arange(0.92, 1.02, 0.02))
        pylab.legend(
            fontsize=tick_size - 3, loc=1, bbox_to_anchor=(6.5, 2.5), frameon=False
        )
        pylab.tight_layout()
        file_name = f"r20_{name}_{pss}_{synth_nat}_{which_size}.pdf"
        pylab.savefig(output_dir / file_name, dpi=dpi, format="pdf")
        os.chdir(cwd)

    def plot_r20(self):
        line_width = self.config.line_width
        output_dir = self.config.output_dir
        r20_folder = self.config.r20_folder
        fig_size = self.config.fig_size
        r20_start = self.config.r20_start
        r20_end = self.config.r20_end
        synth_nat = self.config.synth_nat
        z_order = self.config.z_order
        color_set = self.config.color_set
        tick_size = self.config.tick_size
        tick_width = self.config.tick_width
        tick_length = self.config.tick_length
        label_size = self.config.label_size
        name = self.config.name
        pss = self.config.pss
        dpi = self.config.dpi
        which_size = self.config.which_size

        ms = 8
        line_width = line_width - 1
        cwd = os.getcwd()
        new_cwd = output_dir / r20_folder
        os.chdir(new_cwd)
        fig, ax = pylab.subplots(figsize=(fig_size, fig_size))
        datasets = {
            "mi3": "Mi3",
            "vae": "vVAE",
            "target": "Target",
            "indep": "Indep",
            "deep": "DeepSeq",
        }
        n = np.arange(int(r20_start), int(r20_end))
        i = 0
        ax2 = plt.axes((0, 0, 1, 1))
        ip = InsetPosition(ax, [0.19, 0.12, 0.12, 0.35])
        ax2.set_axes_locator(ip)
        if "nat" in synth_nat:
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
                if "nat" in synth_nat:
                    target_label = "Nat-Target"
                else:
                    target_label = "Synth-Target"
                ax.errorbar(
                    n,
                    m,
                    zorder=z_order[data_label],
                    color=color_set[data_label],
                    yerr=[m - lo, hi - m],
                    linestyle="dotted",
                    linewidth=line_width,
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
                    zorder=z_order[data_label],
                    color=color_set[data_label],
                    yerr=[m - lo, hi - m],
                    linestyle="dotted",
                    linewidth=line_width,
                    fmt=".-",
                    label=target_label,
                    marker=black_marker,
                    ms=ms,
                )[-1][0].set_linewidth(0)
            else:
                ax.errorbar(
                    n,
                    m,
                    zorder=z_order[data_label],
                    color=color_set[data_label],
                    yerr=[m - lo, hi - m],
                    linewidth=line_width,
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
                    zorder=z_order[data_label],
                    color=color_set[data_label],
                    yerr=[m - lo, hi - m],
                    linewidth=line_width,
                    fmt=".-",
                    label=data_label,
                    capthick=0.4,
                    ms=ms,
                )[-1][0].set_linewidth(0)

        ax.set_rasterization_zorder(0)
        ax2.set_rasterization_zorder(0)
        ax.set_xlabel("Higher Order Marginals", fontsize=label_size)
        ax.set_ylabel(r"$r_{20}$", fontsize=label_size)
        ax.tick_params(
            direction="in",
            axis="both",
            which="major",
            labelsize=tick_size,
            length=tick_length,
            width=tick_width,
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
            labelsize=tick_size - 3,
            length=tick_length - 1,
            width=tick_width - 0.1,
        )
        ax2.set_xticks(np.arange(2, 3, 1))
        ax2.set_yticks(np.arange(0.92, 1.02, 0.02))
        pylab.legend(
            fontsize=label_size - 5,
            loc=1,
            bbox_to_anchor=(6.5, 2.5),
            frameon=False,
        )
        pylab.tight_layout()
        file_name = f"r20_{name}_{pss}_{synth_nat}_{which_size}.pdf"
        pylab.savefig(output_dir / file_name, dpi=dpi, format="pdf")
        os.chdir(cwd)

    def make_energies(self):
        energies_file = self.config.energies_file
        which_size = self.config.which_size
        msa_dir = self.config.msa_dir

        print("\t\tmaking energies")
        # FIXME unset var data_home
        data = np.load(msa_dir / energies_file)
        if "M" not in which_size:
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

        fig_size = self.config.fig_size
        c = self.config.color_set[label]
        which_size = self.config.which_size
        stats_sf = self.config.stats_sf
        label_size = self.config.label_size
        name = self.config.name
        tick_size = self.config.tick_size
        tick_length = self.config.tick_length
        tick_width = self.config.tick_width
        parent_dir_name = self.config.parent_dir_name
        dpi = self.config.dpi
        msa_dir = self.config.msa_dir

        _fig, ax = pylab.subplots(figsize=(fig_size, fig_size))
        x_start = min(target_energies)
        x_end = max(target_energies)
        y_start = min(gpsm_energies)
        y_end = max(gpsm_energies)
        x_tick_range = np.arange(-725, -375, 50)

        y_tick_range = {
            "Mi3": np.arange(-700, -300, 100),
            "Indep": np.arange(400, 700, 50),
            "vVAE": np.arange(300, 650, 50),
        }[label]

        pearson_r, _pearson_p = pearsonr(target_energies, gpsm_energies)
        # text = "Pearson R: " + str(round(pearson_r, 3)) + " at p-value = " + str(round(pearson_p, 3))
        text = (
            which_size
            + " "
            + label
            + r", $\rho$ = "
            + str(round(pearson_r, stats_sf))
        )
        print(text)
        pylab.xlabel("Synth-Target Energy", fontsize=label_size)
        pylab.ylabel( f"{which_size} {label} Energy", fontsize=label_size)
        file_name = f"loss-energy_target-{label}_{name}.pdf"

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
        pylab.xticks(x_tick_range, rotation=45, fontsize=tick_size)
        pylab.yticks(y_tick_range, fontsize=tick_size)
        pylab.tick_params(
            direction="in",
            axis="both",
            which="major",
            labelsize=tick_size,
            length=tick_length,
            width=tick_width,
        )
        pylab.legend(fontsize=label_size - 3, loc="upper left", frameon=False)
        pylab.tight_layout()
        pylab.savefig(
            msa_dir / parent_dir_name / file_name,
            dpi=dpi,
            format="pdf",
        )
        pylab.close()
        print("\t\t\tcompleted plotting " + label)