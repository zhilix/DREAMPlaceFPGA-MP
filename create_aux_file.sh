#!/bin/bash

for dir in Design_*; do
    echo $dir

    if [ -e $dir/design.aux ]; then
        rm $dir/design.aux
    fi

    touch $dir/design.aux

    echo -n "design : design.nodes design.nets design.pl design.scl design.lib design.regions design.cascade_shape" >> $dir/design.aux

    if [ -e $dir/design.cascade_shape_instances ]; then
        echo -n " design.cascade_shape_instances" >> $dir/design.aux
    fi

    if [ -e $dir/design.macros ]; then
        echo -n " design.macros" >> $dir/design.aux
    fi

    if [ -e $dir/design.wts ]; then
        echo -n " design.wts" >> $dir/design.aux
    fi

    echo "" >> $dir/design.aux
done

