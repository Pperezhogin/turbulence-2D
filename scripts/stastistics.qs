#!/bin/bash
module load matlab/2023a

experiments=(
    "Dyn2_top_hat"
    "Dyn2_Morinishi_top_hat"
)

folder_with_runs="/scratch/pp2681/Decaying_turbulence/Experiments/Paper/Re_inf/512_filter"
folder_with_scripts="/scratch/pp2681/Decaying_turbulence/decaying-turbulence-code/DSM_Pawar/512/scripts"

for experiment in "${experiments[@]}"; do
    new_folder="$folder_with_runs/$experiment"
    cd "$new_folder"
    pwd
    n=1;
    max=50;
    while [ "$n" -le "$max" ]; do
      cd "model$n"
      "$folder_with_scripts/nse-pseq_bare" nse.dsq series.txt
      ##################
      cd ../
      n=`expr "$n" + 1`;
    done

    # Diagnostics
    cp "$folder_with_scripts"/*.m .
    matlab -batch "run('series_average'); run('spectras'); run('calculate_PDF'); run('spectras_average'); run('PDF_average')"
done