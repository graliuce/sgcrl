# Single-goal Contrastive RL (SGCRL)
[Website](https://graliuce.github.io/sgcrl/) &nbsp;&nbsp;&nbsp;  [Paper](https://arxiv.org/pdf/2408.05804) 

[Watch the intro video!](https://github.com/graliuce/sgcrl/blob/main/intro_video.mp4
)

To set up the environment, follow [instructions linked here](https://docs.google.com/document/d/1Oj9s8-1t7tq_EK5h1iFsCiNFtck_rRVtbXvgMxIsXtQ/edit?usp=sharing).

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
