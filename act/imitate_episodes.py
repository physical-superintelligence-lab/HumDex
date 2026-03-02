import os
# os.environ['TORCH_HOME'] = '/root/autodl-fs/torch_home' # set to persistent storage
#
# Matplotlib safety: force a non-GUI backend to avoid tkinter/Tcl crashes like:
#   RuntimeError: main thread is not in main loop
#   Tcl_AsyncDelete: async handler deleted by the wrong thread
#
# We force Agg when running MuJoCo headless (MUJOCO_GL=egl/osmesa) because even if DISPLAY is set,
# Tk-based backends can still crash in long-running training + background thread cleanup.
_MUJOCO_GL = str(os.environ.get("MUJOCO_GL", "") or "").strip().lower()
_FORCE_AGG = _MUJOCO_GL in ["egl", "osmesa"]
if _FORCE_AGG:
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
    except Exception:
        # If matplotlib isn't available yet, we'll still rely on MPLBACKEND.
        pass

import gc
import torch
import numpy as np
import pickle
import argparse
import h5py
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
import time
import psutil

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data, get_num_episodes, get_episode_ids, get_norm_stats # CHANGED: Use custom HDF5 loader
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

# Optional: keep for simulation tasks
try:
    from sim_env import BOX_POSE
    from utils import sample_box_pose, sample_insertion_pose
except ImportError:
    BOX_POSE = None
    sample_box_pose = None
    sample_insertion_pose = None

import IPython
e = IPython.embed

def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_root = args['ckpt_root']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    resume = bool(args.get('resume', False))
    resume_ckpt = args.get('resume_ckpt', None)
    resume_save_every = int(args.get('resume_save_every', 100))
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']
    dataset_path = args['dataset_path']  # Can be single path or list of paths
    use_rgb = not args['no_rgb']  # ADDED: state-only training option
    hand_side = args.get('hand_side', 'left')
    state_body_dim_arg = args.get('state_body_dim', None)  # 31 or 34, or None for auto-detect
    sequential_training = args.get('sequential_training', False)
    epochs_per_dataset = args.get('epochs_per_dataset', None)
    stage_sleep_seconds = float(args.get('stage_sleep_seconds', 0.0) or 0.0)
    ckpt_prefix = str(args.get('ckpt_prefix', '') or '').strip()
    use_relative_actions = args.get('use_relative_actions', False)  # NEW: relative action space

    # Normalize dataset_path to list
    if isinstance(dataset_path, str):
        dataset_paths = [dataset_path]
    else:
        dataset_paths = list(dataset_path)
    
    # Validate sequential training arguments
    if sequential_training:
        if epochs_per_dataset is None:
            raise ValueError("--sequential_training requires --epochs_per_dataset")
        
        # Two modes:
        # 1. len(epochs_per_dataset) == len(dataset_paths): train each dataset separately (original)
        # 2. len(epochs_per_dataset) == 2: stage1 = mix all but last, stage2 = last dataset only
        if len(epochs_per_dataset) == len(dataset_paths):
            # Original behavior: one-to-one mapping
            print(f"\n=== Sequential Training Mode ===")
            print(f"Will train on {len(dataset_paths)} datasets sequentially:")
            for i, (path, epochs) in enumerate(zip(dataset_paths, epochs_per_dataset)):
                print(f"  Stage {i+1}: {os.path.basename(path)} for {epochs} epochs")
            print()
        elif len(epochs_per_dataset) == 2 and len(dataset_paths) >= 2:
            # New behavior: stage 1 = mix of first N-1 datasets, stage 2 = last dataset
            print(f"\n=== Sequential Training Mode (Two-Stage) ===")
            print(f"Stage 1: Mixed training on {len(dataset_paths)-1} dataset(s) for {epochs_per_dataset[0]} epochs")
            for path in dataset_paths[:-1]:
                print(f"  - {os.path.basename(path)}")
            print(f"Stage 2: Training on last dataset for {epochs_per_dataset[1]} epochs")
            print(f"  - {os.path.basename(dataset_paths[-1])}")
            print()
        else:
            raise ValueError(
                f"--epochs_per_dataset must have either {len(dataset_paths)} values (one per dataset) "
                f"or 2 values (stage1=mix of first {len(dataset_paths)-1}, stage2=last dataset)"
            )
    
    # Use first path for display, but pass all paths to loader
    dataset_path_display = dataset_paths[0] if len(dataset_paths) == 1 else f"{len(dataset_paths)} files"

    if not is_eval:
        if resume:
            ckpt_dir = args.get('ckpt_dir', None)
            if ckpt_dir is None:
                raise ValueError("Resume training requires --ckpt_dir. Example: --resume --ckpt_dir ckpt/<task>/<timestamp>")
        else:
            # for training, create a new ckpt dir
            ts = time.strftime("%Y%m%d_%H%M%S")
            folder = f"{ckpt_prefix}_{ts}" if ckpt_prefix else ts
            ckpt_dir = os.path.join(ckpt_root, task_name, folder)
    else:
        # for evaluation, use passed in ckpt_dir
        ckpt_dir = args['ckpt_dir']
        if ckpt_dir is None:
            raise ValueError("ckpt_dir is required for evaluation. Please pass in --ckpt_dir.")

    # Auto-detect state_body_dim from HDF5 if not specified
    if state_body_dim_arg is None:
        with h5py.File(dataset_paths[0], 'r') as f:
            # Get first episode to check dimensions
            first_ep_key = [k for k in f.keys() if k.startswith('episode_')][0]
            state_body_shape = f[first_ep_key]['state_body'].shape
            detected_state_body_dim = state_body_shape[1]
            print(f"Auto-detected state_body_dim: {detected_state_body_dim}")
            state_body_dim = detected_state_body_dim
    else:
        state_body_dim = state_body_dim_arg
        print(f"Using specified state_body_dim: {state_body_dim}")
    
    # Calculate total state_dim: state_body + hand(s)
    # Single hand (left or right): state_body_dim + 20
    # Both hands: state_body_dim + 40
    hand_dim = 20 if hand_side in ["left", "right"] else 40
    state_dim = state_body_dim + hand_dim  # 51, 54, 71, or 74
    print(f"Total state_dim: {state_dim} (state_body={state_body_dim} + hand={hand_dim})")
    print(f"Hand mode: {hand_side}")

    # CHANGED: Custom task configurations for HDF5 datasets
    # Define your custom tasks here
    CUSTOM_TASK_CONFIGS = {
        'pickball': {
            'dataset_path': dataset_paths,  # Will be set from command line
            'episode_len': 256,  # Max episode length in your data
            'camera_names': ['head'],  # CHANGED: 'rgb' -> 'head'
            'state_dim': state_dim,  # Dynamic: state_body_dim + hand_dim
            'state_body_dim': state_body_dim,  # For reference
            'hand_dim': hand_dim,  # 20 (single) or 40 (both)
        },
    }

    print("parsed args: ", args)

    # CHANGED: Get task parameters from custom config
    if task_name not in CUSTOM_TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task_name}. Available tasks: {list(CUSTOM_TASK_CONFIGS.keys())}")

    task_config = CUSTOM_TASK_CONFIGS[task_name]
    task_config['dataset_path'] = dataset_paths  # Override with command line arg

    # Auto-detect number of episodes from HDF5 file(s)
    num_episodes = get_num_episodes(dataset_paths)
    print(f"Auto-detected {num_episodes} episodes from {dataset_path_display}")

    # Extract task parameters
    state_dim = task_config['state_dim']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # fixed parameters
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'state_dim': state_dim,  # Pass dynamic state_dim
                         'action_dim': 35 + hand_dim,  # Pass dynamic action_dim (55 or 75)
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,
                         'state_dim': state_dim,  # Pass dynamic state_dim
                         'action_dim': 35 + hand_dim,  # Pass dynamic action_dim (55 or 75)
                         }
    else:
        raise NotImplementedError

    # Ensure dataset action horizon matches ACT chunk_size to avoid huge padding & RAM spikes.
    # `act/utils.py` will read ACT_CHUNK_SIZE and clamp action_data/is_pad to this length.
    if policy_class == "ACT":
        os.environ.setdefault("ACT_CHUNK_SIZE", str(int(args["chunk_size"])))

    config = {
        'ckpt_dir': ckpt_dir,
        'num_epochs': num_epochs,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'state_body_dim': state_body_dim,  # 31 or 34
        'action_body_dim': 35,  # Always 35 for action_body
        'hand_dim': hand_dim,  # 20 (single hand) or 40 (both hands)
        'total_action_dim': 35 + hand_dim,  # 55 (single) or 75 (both)
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': True,
        # wandb config
        'use_wandb': args['wandb'] and WANDB_AVAILABLE,
        'wandb_project': args['wandb_project'],
        'wandb_run_name': args['wandb_run_name'],
        'use_rgb': use_rgb,
        'hand_side': hand_side,
        'dataset_paths': dataset_paths,  # For reference
        # resume training
        'resume': resume,
        'resume_ckpt': resume_ckpt,
        'resume_save_every': resume_save_every,
        # relative action space
        'use_relative_actions': use_relative_actions,
    }

    if is_eval:
        ckpt_names = [f'policy_best.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    if sequential_training:
        all_stage_results = []
        prev_checkpoint = None
        unified_norm_stats = None
        
        # Pre-compute unified normalization stats if requested
        if args.get('sequential_unified_stats', False):
            print(f"\n{'='*80}")
            print("Computing Unified Normalization Stats on ALL Datasets")
            print(f"{'='*80}")
            all_episode_ids = get_episode_ids(dataset_paths)
            unified_norm_stats = get_norm_stats(
                dataset_paths,
                all_episode_ids,
                hand_side=hand_side,
                use_relative_actions=use_relative_actions,
                state_body_dim=state_body_dim,
            )
            unified_norm_stats['state_body_dim'] = state_body_dim
            unified_norm_stats['action_body_dim'] = 35
            unified_stats_path = os.path.join(ckpt_dir, 'dataset_stats_unified.pkl')
            with open(unified_stats_path, 'wb') as f:
                pickle.dump(unified_norm_stats, f)
            print(f"Saved unified stats to {unified_stats_path}")
            print(f"  - Computed from {len(all_episode_ids)} episodes across {len(dataset_paths)} datasets")
            print(f"  - All stages will use these unified stats\n")
        
        # Determine stage configuration
        if len(epochs_per_dataset) == 2 and len(dataset_paths) >= 2:
            # Two-stage mode: stage 1 = mix all but last, stage 2 = last only
            stage_configs = [
                (dataset_paths[:-1], epochs_per_dataset[0], 1),  # (datasets, epochs, stage_idx)
                ([dataset_paths[-1]], epochs_per_dataset[1], 2)
            ]
        else:
            # Original mode: one dataset per stage
            stage_configs = [([path], epochs, idx+1) for idx, (path, epochs) in enumerate(zip(dataset_paths, epochs_per_dataset))]
        
        for stage_datasets, stage_epochs, stage_idx in stage_configs:
            print(f"\n{'='*80}")
            if len(stage_datasets) == 1:
                print(f"STAGE {stage_idx}/{len(stage_configs)}: Training on {os.path.basename(stage_datasets[0])}")
            else:
                print(f"STAGE {stage_idx}/{len(stage_configs)}: Mixed training on {len(stage_datasets)} datasets")
                for p in stage_datasets:
                    print(f"  - {os.path.basename(p)}")
            print(f"Epochs: {stage_epochs}")
            print(f"{'='*80}\n")
            
            # Load data for this stage (single or multiple datasets)
            train_dataloader, val_dataloader, stage_stats, _ = load_data(
                stage_datasets,
                None,  # Auto-detect num_episodes
                camera_names,
                batch_size_train,
                batch_size_val,
                use_rgb=use_rgb,
                hand_side=hand_side,
                split_save_path=os.path.join(ckpt_dir, f"train_val_split_stage{stage_idx}.json"),
                val_robot_only=bool(args.get("val_robot_only", False)),
                use_relative_actions=use_relative_actions,
                state_body_dim=state_body_dim,
            )
            
            # Use unified stats if available, otherwise use stage-specific stats
            if unified_norm_stats is not None:
                stats = unified_norm_stats
                print(f"[Stage {stage_idx}] Using unified normalization stats\n")
            else:
                stats = stage_stats
                stats['state_body_dim'] = state_body_dim
                stats['action_body_dim'] = 35
                stats_path = os.path.join(ckpt_dir, f'dataset_stats_stage{stage_idx}.pkl')
                with open(stats_path, 'wb') as f:
                    pickle.dump(stats, f)
                print(f"Saved stage {stage_idx} dataset stats to {stats_path}\n")
            
            # Update config for this stage
            stage_config = deepcopy(config)
            stage_config['num_epochs'] = stage_epochs
            stage_config['stage_idx'] = stage_idx
            stage_config['total_stages'] = len(stage_configs)
            stage_config['resume_from_checkpoint'] = prev_checkpoint
            stage_config['dataset_paths'] = stage_datasets
            
            # Train on this stage
            best_ckpt_info = train_bc_sequential(
                train_dataloader, 
                val_dataloader, 
                stage_config, 
                norm_stats=stats,
                prev_checkpoint=prev_checkpoint
            )
            best_epoch, min_val_loss, best_state_dict = best_ckpt_info
            
            # Save stage checkpoint
            stage_ckpt_path = os.path.join(ckpt_dir, f'policy_stage{stage_idx}_best.ckpt')
            torch.save(best_state_dict, stage_ckpt_path)
            print(f'\nStage {stage_idx} complete: val loss {min_val_loss:.6f} @ epoch {best_epoch}')
            print(f'Saved to {stage_ckpt_path}\n')
            
            # Use this stage's best checkpoint for next stage
            prev_checkpoint = stage_ckpt_path
            all_stage_results.append({
                'stage': stage_idx,
                'datasets': [os.path.basename(p) for p in stage_datasets],
                'epochs': stage_epochs,
                'best_epoch': best_epoch,
                'min_val_loss': min_val_loss,
            })

            # Optional cooldown between stages
            if stage_sleep_seconds > 1e-6 and stage_idx < len(stage_configs):
                try:
                    gc.collect()
                except Exception:
                    pass
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                print(f"\n[sequential] stage cooldown: sleeping {stage_sleep_seconds:.1f}s before next stage...\n")
                time.sleep(stage_sleep_seconds)
        
        # Save final checkpoint
        ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
        torch.save(best_state_dict, ckpt_path)
        print(f"\n{'='*80}")
        print("Sequential Training Complete!")
        print(f"{'='*80}")
        for result in all_stage_results:
            datasets_str = ', '.join(result['datasets']) if len(result['datasets']) > 1 else result['datasets'][0]
            print(f"Stage {result['stage']}: {datasets_str}")
            print(f"  Epochs: {result['epochs']}, Best: {result['best_epoch']}, Val Loss: {result['min_val_loss']:.6f}")
        print(f"\nFinal model saved to: {ckpt_path}")
        
    else:
        # Original behavior: train on all datasets mixed together
        train_dataloader, val_dataloader, stats, _ = load_data(
            dataset_paths,  # Now supports list of paths
            num_episodes,
            camera_names,
            batch_size_train,
            batch_size_val,
            use_rgb=use_rgb,
            hand_side=hand_side,
            split_save_path=os.path.join(ckpt_dir, "train_val_split.json"),
            val_robot_only=bool(args.get("val_robot_only", False)),
            use_relative_actions=use_relative_actions,
            state_body_dim=state_body_dim,
        )

        # save dataset stats (includes metadata from get_norm_stats)
        # Add extra config info to stats for better traceability
        stats['state_body_dim'] = state_body_dim
        stats['action_body_dim'] = 35
        
        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)
        
        print(f"Saved dataset stats to {stats_path}")
        print(f"  - state_dim: {stats.get('state_dim', 'N/A')}")
        print(f"  - action_dim: {stats.get('action_dim', 'N/A')}")
        print(f"  - dataset_paths: {stats.get('dataset_paths', 'N/A')}")
        print(f"  - num_episodes: {stats.get('num_episodes', 'N/A')}")

        best_ckpt_info = train_bc(train_dataloader, val_dataloader, config, norm_stats=stats)
        best_epoch, min_val_loss, best_state_dict = best_ckpt_info

        # save best checkpoint
        ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
        torch.save(best_state_dict, ckpt_path)
        print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'

    # load policy and stats
    # ckpt dir should be like ckpt_root/task_name/time_stamp
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers # requires aloha
        from aloha_scripts.real_env import make_real_env # requires aloha
        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset

        ts = env.reset()

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            for t in range(max_timesteps):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)

                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ### step the environment
                ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

            plt.close()
        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def forward_pass(data, policy, pred_action=False):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()

    if not pred_action: # default behavior in training
        return policy(qpos_data, image_data, action_data, is_pad) # Loss dict
    else:
        return policy(qpos_data, image_data) # a_hat


def visualize_predictions(data, a_hat, norm_stats, body_viz, hand_viz, ckpt_dir, epoch, split='val', num_samples=4, action_body_dim=35, hand_side="left", hand_viz_right=None):
    """Visualize GT and predicted actions for multiple samples. Returns path to saved video.
    
    Args:
        hand_side: "left", "right", or "both" - determines how to split hand actions
        hand_viz_right: Optional right hand visualizer (required when hand_side="both")
    """
    import cv2
    from sim_viz.visualizers import save_video
    from utils import convert_actions_to_absolute
    
    image_data, qpos_data, action_data, is_pad = data
    num_samples = min(num_samples, a_hat.shape[0])
    chunk_size = a_hat.shape[1]
    
    # Check if we're using relative actions
    use_relative_actions = norm_stats.get('use_relative_actions', False)
    state_body_dim = norm_stats.get('state_body_dim', 31)
    
    # Helper to add label on every frame
    def add_label(frame, label):
        frame = frame.copy()
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        return frame
    
    # NOTE on memory:
    # Previously we built `sample_frames` for each sample and then `final_frames` for the whole grid,
    # which can easily consume multiple GB (large HxW grid * ~chunk_size frames).
    # Here we keep only the per-panel frame lists (body/hand GT & pred) and stream-write the final grid video.
    per_sample = []
    for i in range(num_samples):
        # Get valid (non-padded) length for this sample
        pad_mask = is_pad[i, :chunk_size].cpu().numpy()
        valid_len = (~pad_mask).sum()  # Number of non-padded timesteps
        valid_len = max(valid_len, 1)  # At least 1 frame
        
        # Unnormalize predicted actions (only valid part)
        pred_actions = a_hat[i, :valid_len].cpu().numpy() * norm_stats['action_std'] + norm_stats['action_mean']
        
        # Unnormalize GT actions (only valid part)
        gt_actions = action_data[i, :valid_len].cpu().numpy() * norm_stats['action_std'] + norm_stats['action_mean']
        
        # Convert to absolute if using relative action space
        if use_relative_actions:
            # Get the current state (denormalized) for this sample
            qpos = qpos_data[i].cpu().numpy() * norm_stats['qpos_std'] + norm_stats['qpos_mean']
            
            # Convert predicted relative actions to absolute
            pred_actions = convert_actions_to_absolute(pred_actions, qpos, state_body_dim=state_body_dim)
            
            # Convert GT relative actions to absolute
            gt_actions = convert_actions_to_absolute(gt_actions, qpos, state_body_dim=state_body_dim)
        
        # Split body and hand for both GT and pred (use action_body_dim instead of hardcoded 35)
        body_pred = pred_actions[:, :action_body_dim]
        body_gt = gt_actions[:, :action_body_dim]
        
        # Get RGB for this sample (convert CHW->HWC)
        rgb = (image_data[i, 0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        
        # Generate viz frames for body GT and pred
        body_gt_frames = body_viz.visualize(body_gt, verbose=False)
        body_pred_frames = body_viz.visualize(body_pred, verbose=False)
        
        # Handle hand visualization based on hand_side
        if hand_side == "both":
            # action layout: body[0:35], hand_left[35:55], hand_right[55:75]
            hand_left_pred = pred_actions[:, 35:55]
            hand_left_gt = gt_actions[:, 35:55]
            hand_right_pred = pred_actions[:, 55:75]
            hand_right_gt = gt_actions[:, 55:75]
            
            # Generate viz frames for both hands
            hand_left_gt_frames = hand_viz.visualize(hand_left_gt, verbose=False)
            hand_left_pred_frames = hand_viz.visualize(hand_left_pred, verbose=False)
            
            if hand_viz_right is not None:
                hand_right_gt_frames = hand_viz_right.visualize(hand_right_gt, verbose=False)
                hand_right_pred_frames = hand_viz_right.visualize(hand_right_pred, verbose=False)
            else:
                # Fallback: use left hand visualizer for both (will look wrong but won't crash)
                print("[WARN] Warning: hand_viz_right not provided, using left hand visualizer for right hand")
                hand_right_gt_frames = hand_viz.visualize(hand_right_gt, verbose=False)
                hand_right_pred_frames = hand_viz.visualize(hand_right_pred, verbose=False)
        else:
            # Single hand: action[35:55]
            hand_pred = pred_actions[:, action_body_dim:]
            hand_gt = gt_actions[:, action_body_dim:]
            
            hand_left_gt_frames = hand_viz.visualize(hand_gt, verbose=False)
            hand_left_pred_frames = hand_viz.visualize(hand_pred, verbose=False)
            hand_right_gt_frames = None
            hand_right_pred_frames = None
        
        # Resize RGB
        h = body_gt_frames[0].shape[0]
        rgb_resized = cv2.resize(rgb, (int(rgb.shape[1] * h / rgb.shape[0]), h))
        
        # Store per-sample assets; grid will be composed per-timestep and written to video stream.
        sample_data = {
            "rgb": rgb_resized,
            "body_gt": body_gt_frames,
            "body_pred": body_pred_frames,
            "hand_left_gt": hand_left_gt_frames,
            "hand_left_pred": hand_left_pred_frames,
        }
        
        if hand_side == "both":
            sample_data["hand_right_gt"] = hand_right_gt_frames
            sample_data["hand_right_pred"] = hand_right_pred_frames
        
        per_sample.append(sample_data)
    
    # Save video
    video_path = os.path.join(ckpt_dir, f'{split}_viz_epoch_{epoch}.mp4')
    try:
        # Compute max frames across samples; if some sample clips are shorter, repeat the last frame.
        max_frames = 0
        for s in per_sample:
            max_frames = max(max_frames, len(s["body_gt"]))
        max_frames = max(max_frames, 1)

        # Stream-write to avoid holding all grid frames in RAM.
        try:
            import imageio.v2 as imageio  # type: ignore

            # macro_block_size=1 avoids implicit resizing to multiples of 16 (can be surprising)
            writer = imageio.get_writer(video_path, fps=20, macro_block_size=1)
            try:
                expected_shape = None
                for t in range(max_frames):
                    rows = []
                    for s in per_sample:
                        bgt = add_label(s["body_gt"][min(t, len(s["body_gt"]) - 1)], "Body GT")
                        bpred = add_label(s["body_pred"][min(t, len(s["body_pred"]) - 1)], "Body Pred")
                        hlgt = add_label(s["hand_left_gt"][min(t, len(s["hand_left_gt"]) - 1)], "L-Hand GT")
                        hlpred = add_label(s["hand_left_pred"][min(t, len(s["hand_left_pred"]) - 1)], "L-Hand Pred")
                        
                        if hand_side == "both":
                            # Include right hand visualizations
                            hrgt = add_label(s["hand_right_gt"][min(t, len(s["hand_right_gt"]) - 1)], "R-Hand GT")
                            hrpred = add_label(s["hand_right_pred"][min(t, len(s["hand_right_pred"]) - 1)], "R-Hand Pred")
                            combined = np.concatenate([s["rgb"], bgt, bpred, hlgt, hlpred, hrgt, hrpred], axis=1)
                        else:
                            # Single hand mode
                            combined = np.concatenate([s["rgb"], bgt, bpred, hlgt, hlpred], axis=1)
                        
                        combined = np.asarray(combined)
                        if combined.dtype != np.uint8:
                            combined = np.clip(combined, 0, 255).astype(np.uint8)
                        combined = np.ascontiguousarray(combined)
                        rows.append(combined)
                    grid = np.concatenate(rows, axis=0)
                    grid = np.asarray(grid)
                    if grid.dtype != np.uint8:
                        grid = np.clip(grid, 0, 255).astype(np.uint8)
                    grid = np.ascontiguousarray(grid)
                    if expected_shape is None:
                        expected_shape = tuple(grid.shape)
                    elif tuple(grid.shape) != expected_shape:
                        raise ValueError(f"grid frame shape changed: {grid.shape} != {expected_shape}")
                    writer.append_data(grid)
            finally:
                try:
                    writer.close()
                except Exception:
                    pass
            return video_path
        except Exception as e:
            # Do NOT fallback to building a huge frame list (can OOM). Skip visualization instead.
            print(f"[WARN] Visualization stream writer failed ({split}, epoch={epoch}): {e}")
            return None
    except Exception as e:
        print(f"[WARN] Visualization video save failed ({split}, epoch={epoch}): {e}")
        return None


def train_bc(train_dataloader, val_dataloader, config, norm_stats=None):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    use_wandb = config.get('use_wandb', False)
    action_body_dim = config.get('action_body_dim', 35)  # Get from config
    resume = bool(config.get('resume', False))
    resume_ckpt = config.get('resume_ckpt', None)
    resume_save_every = int(config.get('resume_save_every', 20))

    set_seed(seed)

    # Initialize wandb if enabled
    if use_wandb:
        run_name = config.get('wandb_run_name') or f"{config['task_name']}_seed{seed}_{time.strftime('%m%d_%H%M')}"
        wandb.init(
            project=config.get('wandb_project', 'act-training'),
            name=run_name,
            config={
                'task_name': config['task_name'],
                'policy_class': policy_class,
                'num_epochs': num_epochs,
                'seed': seed,
                'lr': config['lr'],
                'use_rgb': config.get('use_rgb', True),
                **policy_config
            }
        )

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    # Resume support: keep a lightweight checkpoint with optimizer state, updated every epoch.
    resume_state_path = os.path.join(ckpt_dir, "resume_state.ckpt")
    start_epoch = 0
    if resume:
        ckpt_to_load = None
        if resume_ckpt:
            ckpt_to_load = resume_ckpt
            if not os.path.isabs(ckpt_to_load):
                ckpt_to_load = os.path.join(ckpt_dir, ckpt_to_load)
        elif os.path.exists(resume_state_path):
            ckpt_to_load = resume_state_path
        else:
            # Fallback: try policy_last / policy_best / latest saved epoch ckpt
            cand = [
                os.path.join(ckpt_dir, "policy_last.ckpt"),
                os.path.join(ckpt_dir, "policy_best.ckpt"),
            ]
            for p in cand:
                if os.path.exists(p):
                    ckpt_to_load = p
                    break
            if ckpt_to_load is None:
                import glob
                import re
                pat = os.path.join(ckpt_dir, f"policy_epoch_*_seed_{seed}.ckpt")
                files = glob.glob(pat)
                best_epoch = -1
                best_file = None
                for fp in files:
                    m = re.search(r"policy_epoch_(\d+)_seed_", os.path.basename(fp))
                    if m:
                        ep = int(m.group(1))
                        if ep > best_epoch:
                            best_epoch = ep
                            best_file = fp
                if best_file is not None:
                    ckpt_to_load = best_file
                    start_epoch = best_epoch + 1

        if ckpt_to_load is None or (not os.path.exists(ckpt_to_load)):
            raise FileNotFoundError(f"[resume] no checkpoint found under ckpt_dir={ckpt_dir}")

        print(f"[resume] loading from: {ckpt_to_load}")
        obj = torch.load(ckpt_to_load, map_location="cuda")
        # resume_state.ckpt is a dict; policy_*.ckpt are raw state_dict
        if isinstance(obj, dict) and ("policy" in obj or "optimizer" in obj):
            if "policy" in obj:
                policy.load_state_dict(obj["policy"])
            if "optimizer" in obj:
                try:
                    optimizer.load_state_dict(obj["optimizer"])
                except Exception as e:
                    print(f"[resume] optimizer state load failed (continue with fresh optimizer): {e}")
            if "epoch" in obj and isinstance(obj["epoch"], int):
                start_epoch = max(start_epoch, int(obj["epoch"]) + 1)
        else:
            policy.load_state_dict(obj)

        if start_epoch >= num_epochs:
            print(f"[resume] start_epoch={start_epoch} >= num_epochs={num_epochs}, nothing to do.")
            return (start_epoch - 1, np.inf, deepcopy(policy.state_dict()))

    # Initialize visualizers for validation (lazy load)
    body_viz, hand_viz, hand_viz_right = None, None, None
    hand_side = config.get('hand_side', 'left')
    try:
        from sim_viz.visualizers import HumanoidVisualizer, HandVisualizer, get_default_paths
        paths = get_default_paths()
        body_viz = HumanoidVisualizer(paths['body_xml'], paths['body_policy'])
        
        # Initialize hand visualizers based on hand_side
        if hand_side == "both":
            hand_viz = HandVisualizer(paths['left_hand_xml'], hand_side='left')
            hand_viz_right = HandVisualizer(paths['right_hand_xml'], hand_side='right')
            print('Visualizers initialized (body + both hands)')
        elif hand_side == "left":
            hand_viz = HandVisualizer(paths['left_hand_xml'], hand_side='left')
            print('Visualizers initialized (body + left hand)')
        else:  # right
            hand_viz = HandVisualizer(paths['right_hand_xml'], hand_side='right')
            print('Visualizers initialized (body + right hand)')
    except Exception as e:
        print(f'Visualizers not available: {e}')

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    save_every_k_epochs = 300  # Changed from 100 to 300
    visualize_every_k_epochs = 100

    # Fixed samples for visualization (same sample each epoch for comparison)
    val_data_viz = next(iter(val_dataloader))
    train_data_viz = next(iter(train_dataloader))
    for epoch in tqdm(range(start_epoch, num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        epoch_train_dicts = []  # FIX: Only keep current epoch's batch losses
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_train_dicts.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(epoch_train_dicts)
        train_history.append(epoch_summary)  # FIX: Only store epoch summary, not all batches
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)
        
        # Memory monitoring
        mem_gb = psutil.Process().memory_info().rss / 1024**3
        print(f'Memory: {mem_gb:.2f} GB')

        # Log to wandb
        if use_wandb:
            log_dict = {'epoch': epoch}
            for k, v in epoch_summary.items():
                log_dict[f'train/{k}'] = v.item()
            log_dict['val/loss'] = epoch_val_loss.item() if hasattr(epoch_val_loss, 'item') else epoch_val_loss
            for k, v in validation_history[-1].items():
                log_dict[f'val/{k}'] = v.item()
            wandb.log(log_dict)

        if epoch % save_every_k_epochs == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

        # Always save resume_state.ckpt (not just when --resume is set)
        # This ensures we can always resume from interruptions
        if epoch % max(1, resume_save_every) == 0 or epoch == num_epochs - 1:
            try:
                torch.save(
                    {"epoch": int(epoch), "policy": policy.state_dict(), "optimizer": optimizer.state_dict()},
                    resume_state_path,
                )
            except Exception as e:
                print(f"[resume] failed to write resume_state.ckpt: {e}")
            
        if epoch % visualize_every_k_epochs == 0:
            # Visualize train and val predictions (for overfitting comparison)
            if body_viz is not None and norm_stats is not None:
                with torch.inference_mode():
                    policy.eval()
                    # Validation sample
                    a_hat_val = forward_pass(val_data_viz, policy, pred_action=True)
                    val_video = visualize_predictions(val_data_viz, a_hat_val, norm_stats, body_viz, hand_viz, ckpt_dir, epoch, split='val', action_body_dim=action_body_dim, hand_side=config['hand_side'], hand_viz_right=hand_viz_right)
                    del a_hat_val  # FIX: Explicitly delete to help GC
                    # Training sample
                    a_hat_train = forward_pass(train_data_viz, policy, pred_action=True)
                    train_video = visualize_predictions(train_data_viz, a_hat_train, norm_stats, body_viz, hand_viz, ckpt_dir, epoch, split='train', action_body_dim=action_body_dim, hand_side=config['hand_side'], hand_viz_right=hand_viz_right)
                    del a_hat_train  # FIX: Explicitly delete to help GC
                if use_wandb:
                    wandb.log({'val/visualization': wandb.Video(val_video), 'train/visualization': wandb.Video(train_video)})
                # FIX: Force garbage collection after visualization to free memory
                gc.collect()

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    # Finish wandb run
    if use_wandb:
        wandb.log({'best_epoch': best_epoch, 'best_val_loss': min_val_loss.item() if hasattr(min_val_loss, 'item') else min_val_loss})
        wandb.finish()

    return best_ckpt_info


def train_bc_sequential(train_dataloader, val_dataloader, config, norm_stats=None, prev_checkpoint=None):
    """
    Sequential training variant that can load from a previous stage's checkpoint.
    Similar to train_bc but designed for multi-stage sequential training.
    """
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    use_wandb = config.get('use_wandb', False)
    action_body_dim = config.get('action_body_dim', 35)
    stage_idx = config.get('stage_idx', 1)
    total_stages = config.get('total_stages', 1)

    set_seed(seed)

    # Initialize wandb if enabled
    if use_wandb:
        run_name = config.get('wandb_run_name') or f"{config['task_name']}_seed{seed}_{time.strftime('%m%d_%H%M')}"
        run_name = f"{run_name}_stage{stage_idx}"  # Add stage suffix
        wandb.init(
            project=config.get('wandb_project', 'act-training'),
            name=run_name,
            config={
                'task_name': config['task_name'],
                'policy_class': policy_class,
                'num_epochs': num_epochs,
                'seed': seed,
                'lr': config['lr'],
                'use_rgb': config.get('use_rgb', True),
                'stage': stage_idx,
                'total_stages': total_stages,
                **policy_config
            }
        )

    # Create policy
    policy = make_policy(policy_class, policy_config)
    
    # Load from previous stage if available
    if prev_checkpoint is not None and os.path.exists(prev_checkpoint):
        print(f"Loading weights from previous stage: {prev_checkpoint}")
        policy.load_state_dict(torch.load(prev_checkpoint))
        print("Previous stage weights loaded successfully")
    
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    # Initialize visualizers for validation (lazy load)
    body_viz, hand_viz, hand_viz_right = None, None, None
    hand_side = config.get('hand_side', 'left')
    try:
        from sim_viz.visualizers import HumanoidVisualizer, HandVisualizer, get_default_paths
        paths = get_default_paths()
        body_viz = HumanoidVisualizer(paths['body_xml'], paths['body_policy'])
        
        # Initialize hand visualizers based on hand_side
        if hand_side == "both":
            hand_viz = HandVisualizer(paths['left_hand_xml'], hand_side='left')
            hand_viz_right = HandVisualizer(paths['right_hand_xml'], hand_side='right')
            print('Visualizers initialized (body + both hands)')
        elif hand_side == "left":
            hand_viz = HandVisualizer(paths['left_hand_xml'], hand_side='left')
            print('Visualizers initialized (body + left hand)')
        else:  # right
            hand_viz = HandVisualizer(paths['right_hand_xml'], hand_side='right')
            print('Visualizers initialized (body + right hand)')
    except Exception as e:
        print(f'Visualizers not available: {e}')

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    save_every_k_epochs = 300  # Changed from 100 to 300
    visualize_every_k_epochs = 100

    # Fixed samples for visualization
    val_data_viz = next(iter(val_dataloader))
    train_data_viz = next(iter(train_dataloader))
    
    # For sequential training, always save resume state (not just when --resume is set)
    resume_state_path = os.path.join(ckpt_dir, f"resume_state_stage{stage_idx}.ckpt")
    resume_save_every = config.get('resume_save_every', 20)
    
    for epoch in tqdm(range(num_epochs)):
        print(f'\n[Stage {stage_idx}/{total_stages}] Epoch {epoch}')
        
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        epoch_train_dicts = []
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_train_dicts.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(epoch_train_dicts)
        train_history.append(epoch_summary)
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)
        
        # Memory monitoring
        mem_gb = psutil.Process().memory_info().rss / 1024**3
        print(f'Memory: {mem_gb:.2f} GB')

        # Log to wandb
        if use_wandb:
            log_dict = {'epoch': epoch, 'stage': stage_idx}
            for k, v in epoch_summary.items():
                log_dict[f'train/{k}'] = v.item()
            log_dict['val/loss'] = epoch_val_loss.item() if hasattr(epoch_val_loss, 'item') else epoch_val_loss
            for k, v in validation_history[-1].items():
                log_dict[f'val/{k}'] = v.item()
            wandb.log(log_dict)

        if epoch % save_every_k_epochs == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_stage{stage_idx}_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, f"{seed}_stage{stage_idx}")
        
        # Always save resume_state for sequential training
        if epoch % max(1, resume_save_every) == 0 or epoch == num_epochs - 1:
            try:
                torch.save(
                    {"epoch": int(epoch), "policy": policy.state_dict(), "optimizer": optimizer.state_dict()},
                    resume_state_path,
                )
            except Exception as e:
                print(f"[resume] failed to write resume_state_stage{stage_idx}.ckpt: {e}")
            
        if epoch % visualize_every_k_epochs == 0:
            if body_viz is not None and norm_stats is not None:
                with torch.inference_mode():
                    policy.eval()
                    a_hat_val = forward_pass(val_data_viz, policy, pred_action=True)
                    val_video = visualize_predictions(val_data_viz, a_hat_val, norm_stats, body_viz, hand_viz, ckpt_dir, epoch, split=f'val_s{stage_idx}', action_body_dim=action_body_dim, hand_side=config['hand_side'], hand_viz_right=hand_viz_right)
                    del a_hat_val
                    a_hat_train = forward_pass(train_data_viz, policy, pred_action=True)
                    train_video = visualize_predictions(train_data_viz, a_hat_train, norm_stats, body_viz, hand_viz, ckpt_dir, epoch, split=f'train_s{stage_idx}', action_body_dim=action_body_dim, hand_side=config['hand_side'], hand_viz_right=hand_viz_right)
                    del a_hat_train
                if use_wandb:
                    log_payload = {}
                    if val_video is not None:
                        log_payload['val/visualization'] = wandb.Video(val_video)
                    if train_video is not None:
                        log_payload['train/visualization'] = wandb.Video(train_video)
                    if log_payload:
                        wandb.log(log_payload)
                gc.collect()

    ckpt_path = os.path.join(ckpt_dir, f'policy_stage{stage_idx}_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_stage{stage_idx}_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Stage {stage_idx} training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, f"{seed}_stage{stage_idx}")

    # Finish wandb run
    if use_wandb:
        wandb.log({'best_epoch': best_epoch, 'best_val_loss': min_val_loss.item() if hasattr(min_val_loss, 'item') else min_val_loss})
        wandb.finish()

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    # Note: train_history and validation_history now store per-epoch summaries
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        fig = plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        epochs = np.arange(len(train_history))  # FIX: x-axis is now epochs (1:1 with history)
        plt.plot(epochs, train_values, label='train')
        plt.plot(epochs, val_values, label='validation')
        plt.xlabel('Epoch')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
        plt.close(fig)  # FIX: Close figure to free memory
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_root', action='store', type=str, help='ckpt_root', required=True)
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=False, default=None)
    parser.add_argument(
        '--ckpt_prefix',
        action='store',
        type=str,
        default='',
        help='Prefix for the timestamp folder name under ckpt_root/task_name. Example: --ckpt_prefix exp1 -> exp1_YYYYmmdd_HHMMSS',
    )
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from --ckpt_dir (loads resume_state.ckpt if present).')
    parser.add_argument('--resume_ckpt', action='store', type=str, required=False, default=None,
                        help='Optional checkpoint to load for resume (absolute path or filename under ckpt_dir).')
    parser.add_argument('--resume_save_every', action='store', type=int, required=False, default=20,
                        help='When --resume is enabled, save resume_state.ckpt every N epochs (default: 20).')
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # ADDED: HDF5 dataset path(s) - supports multiple files
    parser.add_argument('--dataset_path', action='store', type=str, nargs='+',
                       help='Path(s) to HDF5 dataset file(s). Can specify multiple files to merge.',
                       required=True)

    # Select which hand state/action to read from the HDF5 dataset
    parser.add_argument('--hand_side', action='store', type=str, default='left', choices=['left', 'right', 'both'],
                        help='Which hand to read from the dataset keys: state_wuji_hand_{side} and action_wuji_qpos_target_{side}. '
                             'Use "both" to train a policy that controls both hands simultaneously.')
    
    # State body dimension configuration
    parser.add_argument('--state_body_dim', action='store', type=int, default=None, choices=[31, 34],
                        help='State body dimension: 31 (roll/pitch + joints) or 34 (ang_vel + roll/pitch + joints). '
                             'If not specified, auto-detect from dataset.')

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    
    # for state-only training (no visual observations)
    parser.add_argument('--no_rgb', action='store_true', help='Use black placeholder images instead of real images (state-only training)')

    # for wandb logging
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb_project', action='store', type=str, default='act-training', help='wandb project name')
    parser.add_argument('--wandb_run_name', action='store', type=str, default=None, help='wandb run name (default: auto-generated)')

    # for sequential training on multiple datasets
    parser.add_argument('--sequential_training', action='store_true', 
                        help='Train on datasets sequentially instead of mixing them. Requires --epochs_per_dataset.')
    parser.add_argument('--epochs_per_dataset', action='store', type=int, nargs='+', 
                        help='Number of epochs for each dataset when using --sequential_training. Must match number of datasets.')
    parser.add_argument('--sequential_unified_stats', action='store_true',
                        help='(Sequential training only) Compute normalization stats on ALL datasets before training, '
                             'then use the same stats across all stages. Default: False (each stage uses its own stats).')
    parser.add_argument(
        '--stage_sleep_seconds',
        action='store',
        type=float,
        default=0.0,
        help='(sequential only) Sleep this many seconds between stages to allow cleanup/cooldown. Default: 0 (no sleep).',
    )

    parser.add_argument(
        '--val_robot_only',
        action='store_true',
        help="Compute val_loss only on episodes from dataset files whose basename contains 'robot' (training still uses all data). "
             "Only applies when multiple HDF5 files are provided.",
    )

    # for relative action space
    parser.add_argument('--use_relative_actions', action='store_true',
                        help='Use relative action space (actions relative to current state) '
                             'instead of absolute actions. Only matched dimensions are converted: '
                             'body joints (action[3:5] and [6:35]) and hand joints. '
                             'For single hand: action[35:55] relative to state_hand. '
                             'For both hands: action[35:55] relative to state_hand_left, '
                             'action[55:75] relative to state_hand_right. '
                             'Unmatched body dimensions (action[0:3] and [5:6]) remain absolute.')

    main(vars(parser.parse_args()))
