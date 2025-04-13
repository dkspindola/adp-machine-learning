import os
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_mean_with_uncertainty(top_level_folder):
    # Initialize lists to store data for all folders
    all_means = []
    all_stds = []
    folder_labels = []

    # Get and sort timestamp-named folders numerically
    folder_names = [
        folder_name for folder_name in os.listdir(top_level_folder)
        if os.path.isdir(os.path.join(top_level_folder, folder_name))
    ]
    folder_names.sort(key=lambda x: int(x))  # Sort numerically

    # Iterate through sorted folders
    for folder_name in folder_names:
        folder_path = os.path.join(top_level_folder, folder_name)

        # Paths to averages and std JSON files
        averages_path = os.path.join(folder_path, "averages.json")
        std_path = os.path.join(folder_path, "std.json")

        # Skip if files do not exist
        if not os.path.exists(averages_path) or not os.path.exists(std_path):
            continue

        # Load data from JSON files
        with open(averages_path, 'r') as avg_file:
            averages = json.load(avg_file)
        with open(std_path, 'r') as std_file:
            std_devs = json.load(std_file)

        # Extract mean and std values for x, y, and phi
        metrics = ['Verstellweg_X', 'Verstellweg_Y', 'Verstellweg_Phi']
        means = [averages[f"{metric}_mean_absolute_error"] for metric in metrics]
        stds = [std_devs[f"{metric}_mean_absolute_error"] for metric in metrics]

        # Store data for plotting
        all_means.append(means)
        all_stds.append(stds)
        folder_labels.append(folder_name)

    # Sort folder_labels numerically for x-axis
    folder_labels = sorted(folder_labels, key=lambda x: int(x))

    # Create subplots for x, y, and phi
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    x_labels = ['X', 'Y', 'Phi']
    x_positions = np.arange(len(folder_labels))  # Adjust x_positions based on sorted labels

    for i, metric in enumerate(x_labels):
        for j, (means, stds) in enumerate(zip(all_means, all_stds)):
            axes[i].errorbar(
                [j], [means[i]], yerr=[stds[i]], fmt='o', capsize=5, label=f'Folder {folder_labels[j]}'
            )
        axes[i].set_title(f'{metric} Mean Absolute Error with Uncertainty')
        axes[i].set_ylabel('Mean Absolute Error')
        axes[i].grid(True)

    # Update x-axis tick labels to 5, 10, 20, 50
    axes[-1].set_xticks(range(len(folder_labels)))
    axes[-1].set_xticklabels([5, 10, 20, 50], rotation=45, ha='right')
    axes[-1].set_xlabel(f'')
    axes[0].legend()

    plt.tight_layout()
    plt.show()
