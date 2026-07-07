# Chapter 6 experiment repo

This folder is a minimal runnable snapshot for the major experiments in Chapter 6 of the thesis. It includes the experiment sources, the core SPR CUDA headers/modules they use, the mesh assets needed by the shirt examples, and reference diagnostic CSVs from the thesis run.

## Requirements

- NVIDIA GPU with a working CUDA toolkit and `nvcc` on `PATH`.
- Ninja.
- PowerShell, used by the provided `build.ninja` run rules.

The reported thesis run used an NVIDIA GeForce RTX 2080 Ti. Runtimes and small floating-point differences may vary on other GPUs.

## Layout

- `code/`: experiment sources, `build.ninja`, and the minimal SPR library snapshot.
- `main_meshes/`: OBJ assets used by the body/shirt experiments.
- `working/`: output directory. Rerunning the experiments writes OBJ, PPM, and diagnostic CSV files here.
- `reference_outputs/diagnostics/`: CSV diagnostics from the thesis run.

No thesis LaTeX files, Asymptote plot sources, generated PNG/PDF figures, build products, SPR docs, or sandbox files are included.

## Running

From this folder:

```powershell
cd code
ninja
```

The default target runs all three Chapter 6 experiments:

- `run_body_shirt_clearance_experiment`
- `run_body_shirt_landmark_adsdf_experiment`
- `run_tentacle_armature_adsdf_experiment`

To run one experiment:

```powershell
ninja run_body_shirt_clearance_experiment
ninja run_body_shirt_landmark_adsdf_experiment
ninja run_tentacle_armature_adsdf_experiment
```

Each run writes preview renders as binary PPM files, optimized OBJ meshes or controls where applicable, `<name>_diagnostics.csv`, and `<name>_summary.csv` into `working/`.
