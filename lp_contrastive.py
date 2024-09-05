
r"""Example running contrastive RL in JAX.

Run using multi-threading
  python lp_contrastive.py --lp_launch_type=local_mt


"""
import functools
from typing import Any, Dict

from absl import app
from absl import flags
import contrastive
from contrastive import utils as contrastive_utils
import launchpad as lp
import numpy as np
import os

FLAGS = flags.FLAGS

flags.DEFINE_string('log_dir_path', 'logs/', 'Where to log metrics')
flags.DEFINE_integer('time_delta_minutes', 5, 'how often to save checkpoints')
flags.DEFINE_integer('seed', 42, 'Specify seed, only used if use_slurm_array is false')
flags.DEFINE_bool('add_uid', False, 'Whether to add a unique id to the log directory name')
flags.DEFINE_string('alg', 'contrastive_cpc', 'Algorithm type, e.g. default is contrastive_cpc with no entropy or KL losses')
flags.DEFINE_string('env', 'sawyer_bin', 'Environment type, e.g. default is sawyer bin')
flags.DEFINE_integer('num_steps', 8_000_000, 'Number of steps to run', lower_bound=0)
flags.DEFINE_bool('sample_goals', False, 'sample the goal position uniformly according to the environment (corresponds to the original contrastive_rl algorithm)')

# fixed goal coordinates for supported environments
fixed_goal_dict={'point_Spiral11x11': [np.array([5,5], dtype=float), np.array([10,10], dtype=float)],
                     #note: sawyer fixed goal positions vary slightly with each episode
                      'sawyer_bin': np.array([0.12, 0.7, 0.02]),
                      'sawyer_box': np.array([0.0, 0.75, 0.133]),
                      'sawyer_peg': np.array([-0.3, 0.6, 0.0])}

@functools.lru_cache
def get_env(env_name, start_index, end_index, seed, fix_goals = False, fix_goals_actor = False, use_naive_sampling=False, clock_period=None):
  if fix_goals:
    fixed_start_end = fixed_goal_dict[env_name]
  else:
    fixed_start_end = None
    
  return contrastive_utils.make_environment(env_name, start_index, end_index, seed=seed, fixed_start_end = fixed_start_end)


def get_program(params):
  """Constructs the program."""

  env_name = params['env_name']
  seed = params['seed']

  config = contrastive.ContrastiveConfig(**params)
  
  fix_goals = params['fix_goals']

  if fix_goals:
    fixed_start_end = fixed_goal_dict[env_name]
  else:
    fixed_start_end = None
    
  env_factory = lambda seed: contrastive_utils.make_environment(  # pylint: disable=g-long-lambda
      env_name, config.start_index, config.end_index, seed, fixed_start_end = fixed_start_end)

  env_factory_no_extra = lambda seed: env_factory(seed)[0]  # Remove obs_dim.
    
  environment, obs_dim = get_env(env_name, config.start_index,
                                 config.end_index, seed, fix_goals = fix_goals)

  assert (environment.action_spec().minimum == -1).all()
  assert (environment.action_spec().maximum == 1).all()
  config.obs_dim = obs_dim
  config.max_episode_steps = getattr(environment, '_step_limit') + 1
  network_factory = functools.partial(
      contrastive.make_networks, obs_dim=obs_dim, repr_dim=config.repr_dim,
      repr_norm=config.repr_norm, twin_q=config.twin_q,
      use_image_obs=config.use_image_obs,
      hidden_layer_sizes=config.hidden_layer_sizes)
    
  env_factory_fixed_goals = lambda seed: contrastive_utils.make_environment(  # pylint: disable=g-long-lambda
      env_name, config.start_index, config.end_index, seed, fixed_start_end = fixed_goal_dict[env_name])
  env_factory_no_extra_fixed_goals = lambda seed: env_factory_fixed_goals(seed)[0]  # Remove obs_dim.
    
  agent = contrastive.DistributedContrastive(
      seed=seed,
      environment_factory=env_factory_no_extra,
      environment_factory_fixed_goals=env_factory_no_extra_fixed_goals,
      network_factory=network_factory,
      config=config,
      num_actors=config.num_actors,
      log_to_bigtable=True,
      max_number_of_steps=config.max_number_of_steps)
  return agent.build()


def main(_):
  # Create experiment description.

  # 1. Select an environment.
  # Supported environments:
  #   Metaworld: sawyer_{bin,box,peg}
  #   2D nav: point_{Spiral11x11}
  env_name = FLAGS.env
  print('Using env {}...'.format(env_name))
  
  seed_idx = FLAGS.seed
  print('Using random seed {}...'.format(seed_idx))
  params = {
      'seed': seed_idx,
      'use_random_actor': True,
      # entropy_coefficient = None will use adaptive; if setting to a number, note this is log alpha
      'entropy_coefficient': 0.0,
      'env_name': env_name,
      # the number of environment steps
      'max_number_of_steps': FLAGS.num_steps,
  }
  # 2. Select an algorithm. The currently-supported algorithms are:
  # contrastive_nce, contrastive_cpc, c_learning, nce+c_learning
  # Many other algorithms can be implemented by passing other parameters
  # or adding a few lines of code.
  # By default, do contrastive CPC
  alg = FLAGS.alg
  print('Using alg {}...'.format(alg))
  params['alg_name'] = alg
  params['fix_goals'] = not FLAGS.sample_goals
  add_uid = FLAGS.add_uid
  params['add_uid'] = add_uid
  print('Adding uid: {}...'.format(params['add_uid']))
  
  params['log_dir'] = FLAGS.log_dir_path
  params['time_delta_minutes'] = FLAGS.time_delta_minutes
  
  if alg == 'contrastive_cpc':
    params['use_cpc'] = True
  elif alg == 'c_learning':
    params['use_td'] = True
    params['twin_q'] = True
  elif alg == 'nce+c_learning':
    params['use_td'] = True
    params['twin_q'] = True
    params['add_mc_to_td'] = True
  else:
    raise NotImplementedError('Unknown method: %s' % alg)


  program = get_program(params)
  # Set terminal='tmux' if you want different components in different windows.
  
  print(params)
  
  lp.launch(program, terminal='current_terminal')

if __name__ == '__main__':
  app.run(main)
