#!/bin/bash

benchmarks_dir='/home/local/eda15/zhilix/projects/MLCAD_contest/Designs'
dirs=$benchmarks_dir/Design_9*

for dir in $dirs
do
    design_name="$(basename -- $dir)"
    echo $design_name
    python dreamplacefpga/Placer.py test/mlcad_${design_name}.json>mlcad_${design_name}_place.log
    mv mlcad_${design_name}_place.log results_mlcad/log/mlcad_${design_name}_place.log
    # python dreamplacefpga/Placer.py test/mlcad_${design_name}.json
    mv results/design/design.macro.pl results_mlcad/mlcad_${design_name}.macro.pl
done