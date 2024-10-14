# Single-goal Contrastive RL (SGCRL)

## Set up conda environment
**Set up conda:**
1. Load up anaconda: `module load anaconda3`
2. Clone the repository
3. Create an Anaconda environment: `conda create -n contrastive_rl python=3.9 -y`
4. Activate the environment: `conda activate contrastive_rl`

**Install package dependencies:**

1. Change library path: `export LD_LIBRARY_PATH={path to conda}/.conda/envs/contrastive_rl/lib/`
2. Install the requirements: `pip install -r requirements.txt --no-deps`
3. Download the mujoco binaries and place them in ~/.mujoco/ according to instructions in https://github.com/openai/mujoco-py. Run `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{path to mujoco}/.mujoco/mujoco210/bin`
4. Reinstall strict versions for the following packages:
```
pip install dm-acme[jax,tf] 
pip install jax==0.4.10 jaxlib==0.4.10
pip install ml_dtypes==0.2.0
pip install dm-haiku==0.0.9
pip install gymnasium-robotics 
pip uninstall scipy; pip install scipy==1.12
pip install torch==2.1.2 scikit-learn pandas
```

**Potential errors and fixes:**
Cythonizing Error:

`fatal error: GL/glew.h: No such file or directory 4 | #include <GL/glew.h>`

Fix:
```
conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c menpo glfw3
pip install patchelf
```

Cythinizing Error:

`Cannot assign type 'void (const char *) except * nogil' to 'void (*)(const char *) noexcept nogil'`

Fix: `pip install "cython<3"`

**Running with GPU:**

To enable GPU running, run these three commands in a shell with gpu access. This essentially picks out a set of gpu backend infrastructures that is simultaneously supported by jax and the repository code. Note that this step may vary depending on the specifics of the computing environment.

```
module load cudatoolkit/11.3 cudnn/cuda-11.x/8.2.0
pip install optax==0.1.7
pip install --upgrade jax==0.4.7 jaxlib==0.4.7+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/{path to cuda}/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

To run code, use
```
python lp_contrastive.py
```


## Useful flags
- `--env='{ENV_NAME}'`: Specifies the environment. Default environment is 'sawyer_bin'. Currently supported environments include 'sawyer_bin', 'sawyer_box', 'sawyer_peg', 'point_Spiral11x11'.
- `--alg='{ALG_NAME}'`: Specifies the algorithm. Default algorithm is 'contrastive_cpc'. Currently supported algorithms include 'contrastive_nce', 'contrastive_cpc', 'c_learning', 'nce+c_learning'.
- `--num_steps=12_000_000`: Specifies the maximum number of actor steps. 
- `--sample_goals`: Turning on this flag will make the agent collect data conditioned on goals uniformly sampled according to the environment.
(This behavior corresponds to that of the original Contrastive RL algorithm ([Eysenbach et. al, 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/e7663e974c4ee7a2b475a4775201ce1f-Abstract-Conference.html)). 
- `--add_uid`: Randomly generates unique uid and saves checkpoints and logs inside of a directory with that uid as the name. 
