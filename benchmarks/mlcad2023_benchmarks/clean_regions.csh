#!/bin/csh

foreach f (`ls */design.regions`)
    echo "editing $f"
    sed -i "s#/DSP_ALU_INST##g" $f
end