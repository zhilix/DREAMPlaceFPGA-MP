#!/bin/bash

dirs='Design_*'

for dir in $dirs
do
    echo $dir

    if [ -f "$dir/design.json" ]; then
        rm $dir/design.json
    fi

    cat > $dir/design.json <<EOF
{
    "aux_input" : "$(pwd)/$dir/design.aux",
    "gpu" : 0,
    "num_bins_x" : 512,
    "num_bins_y" : 512,
    "global_place_stages" : [
    {"num_bins_x" : 512, "num_bins_y" : 512, "iteration" : 2000, "learning_rate" : 0.01, "wirelength" : "weighted_average", "optimizer" : "nesterov"}
    ],
    "routability_opt_flag" : 0,
    "target_density" : 1.0,
    "density_weight" : 8e-5,
    "random_seed" : 1000,
    "scale_factor" : 1.0,
    "global_place_flag" : 1,
    "legalize_flag" : 1,
    "detailed_place_flag" : 0,
    "dtype" : "float32",
    "plot_flag" : 0,
    "num_threads" : 1,
    "deterministic_flag" : 1,
    "enable_if" : 0,
    "part_name" : "xcvu3p-ffvc1517-2-e"
}
EOF
done

