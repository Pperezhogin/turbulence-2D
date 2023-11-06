experiments=(
    "momentum-flux-no-clipping"
    "momentum-forcing-yes-clipping"
    "vorticity-forcing-no-clipping"
    "momentum-flux-yes-clipping"
    "vorticity-flux-no-clipping"
    "vorticity-forcing-yes-clipping"
    "momentum-forcing-no-clipping"
    "vorticity-flux-yes-clipping"
)

common_path="/scratch/pp2681/Decaying_turbulence/Experiments/Paper/Re_inf/128_filter/DSM_Pawar"

for experiment in "${experiments[@]}"; do
    new_folder="$common_path/$experiment"
    cd "$new_folder"
    #pwd
    #cd - # Back to orginal directory
done