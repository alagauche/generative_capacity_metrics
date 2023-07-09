import pandas as pd
import re


def parse_homs(r20_folder, parent_dir_name, start, end):
    prefix = "r20_"
    column_names = ["mi3", "vae", "indep", "deep", "target"]
    keepers = [0, 2, 4, 6, 8]
    for i in range(start, end):
        i = str(i)
        target_file = open(parent_dir_name + "/" + prefix + i, "r")
        lines = target_file.readlines()
        # parse_hom(lines, i, dir_name)
        nan_counts = 0
        df_lines = list()
        for line in lines:
            line = line.strip()
            line = re.sub("[(),]", "", line)
            if "nan" in line:
                nan_counts += 1
                continue
            line = line.replace("nan", "0")
            line = line.split()
            df_lines.append(line)

        print("\t\t\t\tnans in " + prefix + i + ":\t" + str(nan_counts))

        # remove_cols = [1, 3, 5, 7]
        df = pd.DataFrame(df_lines)
        df = df[keepers]
        df.columns = column_names

        for name in column_names:
            data = df[name].to_list()
            out_name = parent_dir_name + "/" + r20_folder + "/" + name + "_" + i
            with open(out_name, "w") as out_file:
                out_file.write("\n".join(data))
