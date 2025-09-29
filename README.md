# qmc-tumor

## Code Description
The modeling codes in this repository are build on the open-source [`hIPPYlib`](https://github.com/hippylib/hippylib) library, which provides adjoint-based methods for deterministic and Bayesian inverse problems governed by PDEs, and makes use of [`FEniCS`](https://fenicsproject.org/download/archive/) for the high-level formulation, discretization, and solution of PDEs.


## Reference
If you find this library useful in your research, please consider citing the following:
```
@misc{qmconcology2025,
      title={Predictive Digital Twins with Quantified Uncertainty for Patient-Specific Decision Making in Oncology}, 
      author={Alexander D. Gilbert and Frances Y. Kuo and Dirk Nuyens and Graham Pash and Ian H. Sloan and Karen Willcox},
      year={2025},
      eprint={TODO},
      archivePrefix={arXiv},
      primaryClass={TODO},
      url={TODO}, 
}
```

## Installation (Linux and MacOS)
The easiest way to set up a suitable environment is to use `conda`. To use the prebuilt Anaconda Python packages (Linux and Mac only), first install some form of [`Anaconda`](https://www.anaconda.com/docs/getting-started/miniconda/main), then run following commands in your terminal:

```
conda create --name tumorQMC python=3.12
conda activate tumorQMC
conda install -c conda-forge fenics==2019.1.0 matplotlib scipy black
```

To install `hIPPYlib`, please follow the [installation instructions](https://github.com/hippylib/hippylib/blob/master/INSTALL.md).

> [!IMPORTANT]
> **For ARM users**: `FEniCS` is only available on x86 systems. When running on an ARM based mac with the ARM version of `conda` installed, add `CONDA_SUBDIR=osx-64` before the `conda` call,
> ```
> CONDA_SUBDIR=osx-64 conda create -n tumorQMC python=3.12 -c conda-forge fenics==2019.1.0 matplotlib scipy
> ```
> Then configure the environment to be an `osx-64` environment
> ```
> conda activate tumorQMC
> conda config --env --set subdir osx-64
> ```
> Installation can then proceed as normal.

### Alternative Solution (Docker)
Alternatively, `hIPPYlib` provides Docker containers. More information can be found [here](https://github.com/hippylib/hippylib/blob/master/INSTALL.md#run-fenics-from-docker-linux-macos-windows).

## Running the Forward Model
With the appropriate environment running, you can run the forward model with either uniform or lognormal parametric uncertainty. To do so, execute the driver script (optionally, in parallel with MPI) `run_XXX_forward.py`, where `XXX` is either `uniform` or `lognormal`. The code generates a Gaussian bump initial condition and then solves the governing equations, with the appropriate random field model. For more information and to see other options, please run, e.g.:
```
python run_uniform_forward.py --help
```

Included in the `meshes` directory are two example meshes: `lh_mesh.xdmf`, a mesh of a 2D slice of left-hemisphere pial matter and `box.xdmf`, a rectangular mesh of the bounding box of `lh_mesh.xdmf`. The `--mesh` flag is required and allows you to select which mesh should be used. The `--outdir` is used to specify where the simulation output should be saved to.

> [!TIP]
> In case you get `RuntimeError: Could not find DOLFIN pkg-config file. Please make sure appropriate paths are set.` on Mac then doing `export PKG_CONFIG_PATH=~/anaconda3/envs/tumorQMC/lib/pkgconfig:$PKG_CONFIG_PATH` could help (assuming the virtual environment resides in the given path).

### Visualizing the results
The evolution of the state is written to a XDMF file which can be read and visualized with [ParaView](https://www.paraview.org/). The parameters are also written to a XDMF file and may be similarly visualized. Alternatively, the [`movies.ipynb` notebook](./postprocessing/movies.ipynb) can be used as an example to generate visualizations using [PyVista](https://github.com/pyvista/pyvista).

### Using a Karhunen-Loève expansion (KLE) for lognormal parametric uncertainty
To build an $s$-dimensional KLE for a (log)normal random variable $\mathcal{N}(0,\mathcal{C})$, please first use the `build_kle.py` code to construct the reduced basis. An example call to do so is:
```
python -u build_kle.py --mesh meshes/box.xdmf --outdir ./kle_box/
```
This will construct and write the eigenvectors and eigenvalues to file. To sample from this basis and solve the governing equations forward, please use the `run_XXX_forward.py` script, which in turn relies on the `KLEPrior` class defined in `kle.py`. An example call is:
```
python -u run_lognormal_forward.py --mesh meshes/box.xdmf --outdir ./results/box --kledir ./kle_box/ --testkle
```
Please ensure that you are using either `qmc` or `kle` for the `--sampler`.

Run `python run_XXX_forward.py --help` for more information and to see other options.

### Convergence studies
To perform convergence studies, please use the `run_XXX_fwd_prop.py` codes, which leverage MPI to parallelize many forward solves. Example job scripts for submission on TACC machines are available in the `jobs` directory.
