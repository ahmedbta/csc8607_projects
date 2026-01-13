"""
Mini grid search.
"""

import argparse
import yaml
import os
import itertools
import subprocess
import copy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the base config file.")
    parser.add_argument("--max_epochs", type=int, default=3, help="Number of epochs for each grid search run.")
    args = parser.parse_args()

    # --- 1. Load base configuration ---
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)

    # --- 2. Get hyperparameter grid ---
    hparams_grid = base_config.get('hparams', {})
    if not hparams_grid:
        print("No 'hparams' section found in the config file. Exiting.")
        return

    # Create lists of hyperparameter settings
    lr_list = hparams_grid.get('lr', [base_config['train']['optimizer']['lr']])
    wd_list = hparams_grid.get('weight_decay', [base_config['train']['optimizer']['weight_decay']])
    
    model_hparams = hparams_grid.get('model', {})
    channels_block3_list = model_hparams.get('channels_block3', [base_config['model']['channels_block3']])
    dropout_list = model_hparams.get('dropout', [base_config['model']['dropout']])

    # Generate all combinations
    combinations = list(itertools.product(lr_list, wd_list, channels_block3_list, dropout_list))
    print(f"Starting grid search with {len(combinations)} combinations.")

    # --- 3. Run training for each combination ---
    for i, (lr, wd, channels, dropout) in enumerate(combinations):
        print(f"\n--- Combination {i+1}/{len(combinations)} ---")
        print(f"LR: {lr}, WD: {wd}, Channels: {channels}, Dropout: {dropout}")

        # Create a new config for this run
        run_config = copy.deepcopy(base_config)
        run_config['train']['optimizer']['lr'] = lr
        run_config['train']['optimizer']['weight_decay'] = wd
        run_config['model']['channels_block3'] = channels
        run_config['model']['dropout'] = dropout
        
        # Create a unique run name
        run_name = f"grid_search/lr={lr}_wd={wd}_ch={channels}_do={dropout}"
        
        # Path for the temporary config file
        temp_config_dir = os.path.join(os.path.dirname(args.config), "temp")
        os.makedirs(temp_config_dir, exist_ok=True)
        temp_config_path = os.path.join(temp_config_dir, f"config_{i}.yaml")

        # Write the temporary config file
        with open(temp_config_path, 'w') as f:
            yaml.dump(run_config, f)

        # Prepare the command to run training
        command = [
            '.\\venv\\Scripts\\python.exe',
            '-m',
            'src.train',
            '--config',
            temp_config_path,
            '--max_epochs',
            str(args.max_epochs),
            '--run_name',
            run_name
        ]

        # Run the training script
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running training for combination {i+1}: {e}")
        finally:
            # Clean up the temporary config file
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)

    print("\n--- Grid Search Complete ---")


if __name__ == "__main__":
    main()
