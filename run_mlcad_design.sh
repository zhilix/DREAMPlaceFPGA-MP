#!/bin/bash
DREAMPlaceFPGA_dir=$1
if [ -f ${DREAMPlaceFPGA_dir}/dreamplacefpga/Placer.py ]; then
    echo "DREAMPlaceFPGA_dir = ${DREAMPlaceFPGA_dir}"
else
    echo "DREAMPlaceFPGA_dir = ${DREAMPlaceFPGA_dir} is not a valid directory"
    exit 0
fi

gpu_flag=$2
if [[ -z $2 ]] ; then
    echo "option not given"
    echo "CPU mode: run_mlcad_design.sh ${DREAMPlaceFPGA_dir} 0"
    gpu_flag=0
else
    if [[ $2 -eq 1 ]] ; then
        echo "GPU mode: run_mlcad_design.sh ${DREAMPlaceFPGA_dir} ${gpu_flag}"
    else
        echo "CPU mode: run_mlcad_design.sh ${DREAMPlaceFPGA_dir} ${gpu_flag}"
    fi
fi

#clean bookshelf files
lib_file=$(pwd)/design.lib
if [ -f $lib_file ]; then
    sed -i 's/CELL END/END CELL/' $lib_file
fi

instance_file=$(pwd)/design.cascade_shape_instances
if [ -f $instance_file ]; then
    sed -i 's/BRAM_CASCADE /BRAM_CASCADE_2 /' $instance_file
fi

#create design.aux file
if [ -e $(pwd)/design.aux ]; then
    rm $(pwd)/design.aux
fi

touch $(pwd)/design.aux
echo -n "design : design.nodes design.nets design.pl design.scl design.lib design.regions design.cascade_shape" >> $(pwd)/design.aux

if [ -e $(pwd)/design.cascade_shape_instances ]; then
    echo -n " design.cascade_shape_instances" >> $(pwd)/design.aux
fi

if [ -e $(pwd)/design.macros ]; then
    echo -n " design.macros" >> $(pwd)/design.aux
fi

if [ -e $(pwd)/design.wts ]; then
    echo -n " design.wts" >> $(pwd)/design.aux
fi

echo "" >> $(pwd)/design.aux

#create design.json file
if [ -f "$(pwd)/design.json" ]; then
    rm $(pwd)/design.json
fi

cat > $(pwd)/design.json <<EOF
{
    "aux_input" : "$(pwd)/design.aux",
    "gpu" : ${gpu_flag},
    "num_bins_x" : 512,
    "num_bins_y" : 512,
    "global_place_stages" : [
    {"num_bins_x" : 512, "num_bins_y" : 512, "iteration" : 2000, "learning_rate" : 0.01, "wirelength" : "weighted_average", "optimizer" : "nesterov"}
    ],
    "target_density" : 1.0,
    "density_weight" : 8e-5,
    "random_seed" : 1000,
    "scale_factor" : 1.0,
    "global_place_flag" : 1,
    "dtype" : "float32",
    "num_threads" : 16,
    "deterministic_flag" : 1
}
EOF

#run the design
python ${DREAMPlaceFPGA_dir}/dreamplacefpga/Placer.py $(pwd)/design.json
