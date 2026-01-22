import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_mir9_data(filepath):
    """Load and clean the MIR9 CSV data."""
    df = pd.read_csv(filepath, sep=';')
    # Convert to proper float
    df['freq'] = df['freq'].str.replace(',', '.').astype(float)
    return df


def load_tsv_data(filepath):
    """Load and clean the TSV data."""
    df = pd.read_csv(filepath, sep='\t')
    df = df.replace('NA', np.nan)

    # Convert numeric columns
    numeric_cols = ['hd_frac', 'shd_frac', 'precision', 'recall', 'f1']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df



def plot_vs_nodes(df, metrics='f1', nodes_col='nodes', title_prefix='', save_path=None):
    """
    Plot F1 score as a function of number of nodes in the Bayesian network.
    Specific to TSV data.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    plot_data = df.dropna(subset=[metrics])
    grouped = plot_data.groupby(nodes_col)[metrics].agg(['mean', 'std', 'count']).reset_index()

    ax.errorbar(grouped[nodes_col], grouped['mean'], yerr=grouped['std'],
                marker='o', linewidth=2, capsize=5, capthick=2,
                label='Mean {} (n={} samples)'.format(metrics, grouped["count"].sum()))

    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('{} Score'.format(metrics))
    ax.set_title('{}{} Score vs Number of Nodes in Bayesian Network'.format(title_prefix, metrics))
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    return fig


def plot_vs_datapoints(df, mode_col='mode', datapoints_col='datapoints', metric='f1',
                       title_prefix='', save_path=None):
    """
    Plot score as a function of number of datapoints/trajectories.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get unique modes
    modes = df[mode_col].unique()

    for mode in modes:
        mode_data = df[df[mode_col] == mode].copy()

        grouped = mode_data.groupby(datapoints_col)[metric].agg(['mean', 'std']).reset_index()

        ax.plot(grouped[datapoints_col], grouped['mean'],
                marker='o', label='{}'.format(mode), linewidth=2)
        ax.fill_between(grouped[datapoints_col],
                        grouped['mean'] - grouped['std'],
                        grouped['mean'] + grouped['std'],
                        alpha=0.2)

    ax.set_xlabel('Number of Trajectories')
    ax.set_ylabel('{} Score'.format(metric))
    ax.set_title('{}{} Score vs Number of Trajectories'.format(title_prefix, metric))
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    return fig


def plot_vs_trajectory_length(df, mode_col='mode', length_col='trajectory_len', metric='f1',
                              title_prefix='', save_path=None):
    """Plot score as a function of trajectory length."""
    fig, ax = plt.subplots(figsize=(10, 6))

    modes = df[mode_col].unique()

    for mode in modes:
        mode_data = df[df[mode_col] == mode].copy()
        grouped = mode_data.groupby(length_col)[metric].agg(['mean', 'std']).reset_index()

        ax.plot(grouped[length_col], grouped['mean'],
                marker='s', label='{}'.format(mode), linewidth=2)
        ax.fill_between(grouped[length_col],
                        grouped['mean'] - grouped['std'],
                        grouped['mean'] + grouped['std'],
                        alpha=0.2)

    ax.set_xlabel('Trajectory Length')
    ax.set_ylabel('{} Score'.format(metric))
    ax.set_title('{}{} Score vs Trajectory Length'.format(title_prefix, metric))
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    return fig


def plot_metrics_heatmap(df, x_col, y_col, metric='f1', mode_filter=None,
                         title_prefix='', save_path=None):
    """
    Create a heatmap showing a metric as a function of two parameters.

    Parameters:
    -----------
    df : DataFrame
    x_col : str
        Column name for x-axis
    y_col : str
        Column name for y-axis
    metric : str
        Metric to plot (f1, precision, recall, etc.)
    mode_filter : str or None
        If provided, filter data to this mode only
    """
    plot_data = df.copy()
    if mode_filter:
        plot_data = plot_data[plot_data['mode'] == mode_filter]

    # Create pivot table
    pivot = plot_data.pivot_table(values=metric, index=y_col, columns=x_col, aggfunc='mean')

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis', ax=ax,
                cbar_kws={'label': metric.upper()})

    mode_str = ' ({})'.format(mode_filter) if mode_filter else ''
    ax.set_title('{}{} Heatmap: {} vs {}{}'.format(title_prefix, metric.upper(), y_col, x_col, mode_str))
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    return fig


def plot_vs_nodes_by_score(df, metric='f1', nodes_col='nodes', score_col='score',
                           title_prefix='', save_path=None):
    """
    Plot score vs nodes, separated by scoring function (MDL/BDE).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    plot_data = df.dropna(subset=[metric])
    scores = plot_data[score_col].unique()

    for score in scores:
        score_data = plot_data[plot_data[score_col] == score]
        grouped = score_data.groupby(nodes_col)[metric].agg(['mean', 'std']).reset_index()

        ax.plot(grouped[nodes_col], grouped['mean'],
                marker='o', label='{}'.format(score), linewidth=2)
        ax.fill_between(grouped[nodes_col],
                        grouped['mean'] - grouped['std'],
                        grouped['mean'] + grouped['std'],
                        alpha=0.2)

    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('{} Score'.format(metric))
    ax.set_title('{}{} Score vs Number of Nodes (by Scoring Function)'.format(title_prefix, metric))
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    return fig


def plot_metrics_comparison(df, group_by_col, metric_cols=None,
                            title_prefix='', save_path=None):
    """
    Create a grouped bar plot comparing multiple metrics across different groups.
    """
    if metric_cols is None:
        metric_cols = ['prec', 'recall', 'f1']
    plot_data = df.dropna(subset=metric_cols)
    grouped = plot_data.groupby(group_by_col)[metric_cols].mean().reset_index()

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(grouped))
    n_metrics = len(metric_cols)
    # narrower bars for more metrics
    width = 0.8 / n_metrics  # Total width of 0.8 per group, divided by number of metrics

    for i, metric in enumerate(metric_cols):
        # Position bars side by side within each group
        offset = (i - n_metrics / 2.0 + 0.5) * width
        ax.bar(x + offset, grouped[metric], width, label=metric.upper())

    ax.set_xlabel(group_by_col)
    ax.set_ylabel('Score')
    ax.set_title('{}Comparison of Metrics by {}'.format(title_prefix, group_by_col))
    ax.set_xticks(x)
    ax.set_xticklabels(grouped[group_by_col])
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    return fig

def plot_sync_vs_async_comparison(df, metric='f1', group_col='datapoints',
                                  mode_col='mode', title_prefix='', save_path=None):
    """
    Direct comparison of synchronous vs asynchronous performance.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    plot_data = df.dropna(subset=[metric])

    # Assume modes are something like 'Sync'/'Async' or 'synchronous'/'asynchronous'
    modes = plot_data[mode_col].unique()

    for mode in modes:
        mode_data = plot_data[plot_data[mode_col] == mode]
        grouped = mode_data.groupby(group_col)[metric].agg(['mean', 'std']).reset_index()

        ax.plot(grouped[group_col], grouped['mean'],
                marker='o', label=mode, linewidth=2, markersize=8)
        ax.fill_between(grouped[group_col],
                        grouped['mean'] - grouped['std'],
                        grouped['mean'] + grouped['std'],
                        alpha=0.2)

    ax.set_xlabel(group_col)
    ax.set_ylabel(metric.upper())
    ax.set_title('{}Synchronous vs Asynchronous: {} by {}'.format(title_prefix, metric.upper(), group_col))
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    return fig


# Example usage function
def generate_all_plots_mir9(csv_path, output_dir='plots_mir9'):
    """Generate all relevant plots for MIR9 dataset."""

    df = load_mir9_data(csv_path)

    print("Loaded {} rows from MIR9 data".format(len(df)))
    print("Modes: {}".format(df['mode'].unique()))
    print("Datapoints range: {} - {}".format(df['datapoints'].min(), df['datapoints'].max()))
    print("Trajectory length range: {} - {}".format(df['trajectory_len'].min(), df['trajectory_len'].max()))

    # F1 vs datapoints
    plot_vs_datapoints(df, save_path='{}/f1_vs_datapoints.png'.format(output_dir))

    # F1 vs trajectory length
    plot_vs_trajectory_length(df, save_path='{}/f1_vs_trajectory_length.png'.format(output_dir))

    # Heatmaps for Sync and Async
    plot_metrics_heatmap(df, 'datapoints', 'trajectory_len', metric='f1',
                         mode_filter='Sync', save_path='{}/heatmap_f1_sync.png'.format(output_dir))
    plot_metrics_heatmap(df, 'datapoints', 'trajectory_len', metric='f1',
                         mode_filter='Async', save_path='{}/heatmap_f1_async.png'.format(output_dir))

    # Metrics comparison
    plot_metrics_comparison(df,
                            'datapoints',
                            save_path='{}/metrics_by_datapoints.png'.format(output_dir),
                            metric_cols = ['prec', 'rec', 'f1', 'hd_frac', 'shd_frac'])
    plot_metrics_comparison(df,
                            'trajectory_len',
                            save_path='{}/metrics_by_length.png'.format(output_dir),
                            metric_cols = ['prec', 'rec', 'f1', 'hd_frac', 'shd_frac'])

    # Sync vs Async comparison
    plot_sync_vs_async_comparison(df, metric='f1', group_col='datapoints',
                                  save_path='{}/sync_async_datapoints.png'.format(output_dir))
    plot_sync_vs_async_comparison(df, metric='f1', group_col='trajectory_len',
                                  save_path='{}/sync_async_length.png'.format(output_dir))

    print("All plots saved to {}/".format(output_dir))


def generate_all_plots_tsv(tsv_path, output_dir='plots_tsv'):
    """Generate all relevant plots for TSV dataset."""

    df = load_tsv_data(tsv_path)

    print("Loaded {} rows from TSV data".format(len(df)))
    print("Nodes range: {} - {}".format(df['nodes'].min(), df['nodes'].max()))
    print("Scoring functions: {}".format(df['score'].unique()))
    print("Valid F1 scores: {}/{}".format(df['f1'].notna().sum(), len(df)))

    # F1 vs number of nodes
    plot_vs_nodes(df, save_path='{}/f1_vs_nodes.png'.format(output_dir))
    plot_vs_nodes_by_score(df, save_path='{}/f1_vs_nodes_by_score.png'.format(output_dir))

    # F1 vs datapoints (using ntraj column)
    plot_vs_datapoints(df,
                       datapoints_col='ntraj',
                       mode_col='mode',
                       save_path='{}/f1_vs_ntraj.png'.format(output_dir))

    # F1 vs trajectory length (using len column)
    plot_vs_trajectory_length(df,
                              length_col='len', mode_col='mode',
                              save_path='{}/f1_vs_len.png'.format(output_dir))

    # Heatmaps
    plot_metrics_heatmap(df, 'ntraj', 'len', metric='f1',
                         save_path='{}/heatmap_f1_ntraj_len.png'.format(output_dir))
    plot_metrics_heatmap(df, 'nodes', 'ntraj', metric='f1',
                         save_path='{}/heatmap_f1_nodes_ntraj.png'.format(output_dir))

    # Metrics comparison
    plot_metrics_comparison(df,
                            'nodes',
                            save_path='{}/metrics_by_nodes.png'.format(output_dir),
                            metric_cols = ['hd_frac', 'shd_frac', 'precision', 'recall', 'f1'])
    plot_metrics_comparison(df,
                            'ntraj',
                            save_path='{}/metrics_by_ntraj.png'.format(output_dir),
                            metric_cols = ['hd_frac', 'shd_frac', 'precision', 'recall', 'f1'])
    plot_metrics_comparison(df,
                            group_by_col='len',
                            metric_cols=['hd_frac', 'shd_frac', 'precision', 'recall', 'f1'],
                            save_path='{}/metrics_by_len.png'.format(output_dir)
                            )
    plot_metrics_comparison(df,
                            group_by_col='tf',
                            metric_cols=['hd_frac', 'shd_frac', 'precision', 'recall', 'f1'],
                            save_path='{}/metrics_by_tf.png'.format(output_dir)
                            )
    plot_metrics_comparison(df,
                            group_by_col='sf',
                            metric_cols=['hd_frac', 'shd_frac', 'precision', 'recall', 'f1'],
                            save_path='{}/metrics_by_sf.png'.format(output_dir)
                            )

    print("All plots saved to {}/".format(output_dir))


if __name__ == '__main__':
    pass
    # Example usage - modify paths as needed

    # For MIR9 CSV data
    #generate_all_plots_mir9('tables/MIR9_evaluation.csv')

    # For TSV data
    generate_all_plots_tsv('tables/evaluation_asynchronous.tsv', output_dir="plots_tsv_async")
    #generate_all_plots_tsv('tables/evaluation_synchronous.tsv', output_dir="plots_tsv_sync")
