import os
import re
import pandas as pd

df = pd.read_csv('~/Downloads/SJS/4ow0_sp50_xp50_ligpre_cluster_0.55.csv', header=None)
zinc_id_all = pd.unique(df[2])
print('zinc_id_all:', len(zinc_id_all))
sdf_dict = {}
with open('/mnt/home/SJS/vs_workflow/4ow0/4ow0_sp50_xp50_ligpre-out.sdf', 'r')as f:
    lines = f.read()
    # lines_ = re.split(r"(\$\$\$\$\n)", lines)
    lines_ = re.split("\$\$\$\$\n", lines)[:-1]
    lines_ = [line+"$$$$" for line in lines_]
    for line in lines_:
        sdf_dict[line.split('\n', 1)[0]]=line
no_sdf = []
for zinc_id in (zinc_id_all):
    try:
        with open('/mnt/home/SJS/vs_workflow/4ow0/4ow0_sp50_xp50_ligpre_out_split_sdf/'+zinc_id+'.sdf', 'w') as f:
            f.write(sdf_dict[zinc_id])
    except:
        no_sdf.append(zinc_id)
print('no_sdf', no_sdf)


