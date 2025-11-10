#!/usr/bin/env python3
"""
Simple experiment runner for Lambda Labs GPU instance.
Run single experiment or all experiments sequentially.

Usage:
    python run_experiments.py --exp 11              # Run single experiment
    python run_experiments.py --exp 11 12 13        # Run multiple experiments
    python run_experiments.py --all                 # Run all new experiments
"""

import subprocess
import sys
from argparse import ArgumentParser

# Define new experiments based on analysis
EXPERIMENTS = {
    # NEW EXPERIMENTS: 20 epochs, cyclical schedules, higher beta_end
    '14': {
        'name': 'exp14_20ep_standard_beta3',
        'vq_type': 'ba',
        'beta_start': 0.0,
        'beta_end': 3.0,
        'num_cycles': 1,
        'ba_iters': 2,
        'entropy_weight': 0.0,
        'max_epochs': 20,
        'description': '20 epochs, standard annealing, beta_end=3.0',
    },
    '15': {
        'name': 'exp15_20ep_standard_beta5',
        'vq_type': 'ba',
        'beta_start': 0.0,
        'beta_end': 5.0,
        'num_cycles': 1,
        'ba_iters': 2,
        'entropy_weight': 0.0,
        'max_epochs': 20,
        'description': '20 epochs, standard annealing, higher beta_end=5.0',
    },
    '16': {
        'name': 'exp16_20ep_standard_beta6',
        'vq_type': 'ba',
        'beta_start': 0.0,
        'beta_end': 6.0,
        'num_cycles': 1,
        'ba_iters': 2,
        'entropy_weight': 0.0,
        'max_epochs': 20,
        'description': '20 epochs, standard annealing, higher beta_end=6.0',
    },
    '17': {
        'name': 'exp17_20ep_cyclical2_beta3',
        'vq_type': 'ba',
        'beta_start': 0.0,
        'beta_end': 3.0,
        'num_cycles': 2,
        'ba_iters': 2,
        'entropy_weight': 0.0,
        'max_epochs': 20,
        'description': '20 epochs, 2 cycles, beta_end=3.0',
    },
    '18': {
        'name': 'exp18_20ep_cyclical3_beta3',
        'vq_type': 'ba',
        'beta_start': 0.0,
        'beta_end': 3.0,
        'num_cycles': 3,
        'ba_iters': 2,
        'entropy_weight': 0.0,
        'max_epochs': 20,
        'description': '20 epochs, 3 cycles, beta_end=3.0',
    },
    '19': {
        'name': 'exp19_20ep_cyclical2_beta5',
        'vq_type': 'ba',
        'beta_start': 0.0,
        'beta_end': 5.0,
        'num_cycles': 2,
        'ba_iters': 2,
        'entropy_weight': 0.0,
        'max_epochs': 20,
        'description': '20 epochs, 2 cycles, higher beta_end=5.0',
    },
    '20': {
        'name': 'exp20_20ep_cyclical3_beta5',
        'vq_type': 'ba',
        'beta_start': 0.0,
        'beta_end': 5.0,
        'num_cycles': 3,
        'ba_iters': 2,
        'entropy_weight': 0.0,
        'max_epochs': 20,
        'description': '20 epochs, 3 cycles, higher beta_end=5.0',
    },
}

def build_command(exp_config):
    """Build vqvae.py command from experiment config."""
    cmd = ['python', 'vqvae_cifar/vqvae.py']

    # Add all parameters
    for key, value in exp_config.items():
        if key in ['name', 'description']:
            continue

        # Keep underscores to match vqvae.py argument names
        arg_name = f"--{key}"

        if isinstance(value, bool):
            if value:
                cmd.append(arg_name)
        else:
            cmd.extend([arg_name, str(value)])

    # Add wandb logging
    cmd.extend(['--use_wandb', '--wandb_name', exp_config['name']])

    # Add tags
    exp_id = exp_config['name'].split('_')[0]
    cmd.extend(['--wandb_tags', exp_id, exp_config['vq_type']])

    return cmd

def run_experiment(exp_id):
    """Run a single experiment."""
    if exp_id not in EXPERIMENTS:
        print(f"ERROR: Unknown experiment '{exp_id}'")
        print(f"Available: {', '.join(EXPERIMENTS.keys())}")
        return False

    config = EXPERIMENTS[exp_id]

    print("=" * 80)
    print(f"EXPERIMENT: {exp_id} - {config['name']}")
    if 'description' in config:
        print(f"Description: {config['description']}")
    print("=" * 80)
    print()

    # Print config
    print("Configuration:")
    for key, value in config.items():
        if key not in ['name', 'description']:
            print(f"  {key}: {value}")
    print()

    # Build and run command
    cmd = build_command(config)
    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, check=True)
        print()
        print(f"✓ Experiment {exp_id} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print()
        print(f"✗ Experiment {exp_id} failed with code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print()
        print("✗ Experiment interrupted by user")
        return False

def main():
    parser = ArgumentParser(description='Run VQ-VAE experiments on Lambda GPU')
    parser.add_argument('--exp', nargs='+', help='Experiment ID(s) to run')
    parser.add_argument('--all', action='store_true', help='Run all new experiments (14-20)')
    parser.add_argument('--list', action='store_true', help='List available experiments')

    args = parser.parse_args()

    if args.list:
        print("\nAvailable Experiments:")
        print("=" * 80)
        for exp_id, config in EXPERIMENTS.items():
            desc = config.get('description', '')
            print(f"\n{exp_id}: {config['name']}")
            if desc:
                print(f"  → {desc}")
            if config['vq_type'] == 'ba':
                print(f"  → β: {config.get('beta_start', '-')} → {config.get('beta_end', '-')}, "
                      f"cycles={config.get('num_cycles', 1)}, "
                      f"iters={config.get('ba_iters', '-')}, "
                      f"entropy={config.get('entropy_weight', '-')}")
        print()
        return

    # Determine which experiments to run
    if args.all:
        exp_ids = ['14', '15', '16', '17', '18', '19', '20']
    elif args.exp:
        exp_ids = args.exp
    else:
        parser.print_help()
        return

    # Run experiments
    print(f"\n🚀 Starting {len(exp_ids)} experiment(s)\n")

    results = {}
    for i, exp_id in enumerate(exp_ids, 1):
        print(f"\n[{i}/{len(exp_ids)}] Running experiment {exp_id}...")
        success = run_experiment(exp_id)
        results[exp_id] = success
        print()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for exp_id, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{exp_id}: {status}")
    print()

    # Exit code
    if all(results.values()):
        print("All experiments completed successfully! 🎉")
        sys.exit(0)
    else:
        print("Some experiments failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
