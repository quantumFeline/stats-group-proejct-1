import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def load_mir9_data(filepath):
    """Load MIR9 CSV data."""
    df = pd.read_csv(filepath, sep=';')
    df['freq'] = df['freq'].str.replace(',', '.').astype(float)
    return df


def load_tsv_data(filepath):
    """Load TSV data."""
    df = pd.read_csv(filepath, sep='\t')
    df = df.replace('NA', np.nan)
    numeric_cols = ['hd_frac', 'shd_frac', 'precision', 'recall', 'f1']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def filter_data(df, **conditions):
    """
    Filter dataframe by exact conditions.

    Example:
        filter_data(df, ntraj=3, len=20, score='MDL')
    """
    result = df.copy()
    for col, val in conditions.items():
        if col in result.columns:
            result = result[result[col] == val]
    return result


def filter_data(df, **conditions):
    """
    Filter dataframe by exact conditions.

    Example:
        filter_data(df, ntraj=3, len=20, score='MDL')
    """
    result = df.copy()
    for col, val in conditions.items():
        if col in result.columns:
            result = result[result[col] == val]
    return result


def plot_metric_vs_param(df, x_col, metric='f1', split_by=None,
                         title=None, save_path=None):
    """
    Plot metric vs a single parameter, optionally split by another variable.
    Only uses data where all other parameters match - no aggregation.

    Parameters:
    -----------
    df : DataFrame
        Already filtered to comparable conditions
    x_col : str
        Parameter to plot on x-axis (e.g., 'nodes', 'ntraj', 'len')
    metric : str
        Metric to plot (e.g., 'f1', 'precision', 'shd_frac')
    split_by : str or None
        Column to create separate lines (e.g., 'score', 'mode')
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    plot_data = df.dropna(subset=[metric])

    if split_by and split_by in plot_data.columns:
        for value in sorted(plot_data[split_by].unique()):
            subset = plot_data[plot_data[split_by] == value]
            grouped = subset.groupby(x_col)[metric].agg(['mean', 'std']).reset_index()

            ax.plot(grouped[x_col], grouped['mean'],
                    marker='o', label=str(value), linewidth=2)
            ax.fill_between(grouped[x_col],
                            grouped['mean'] - grouped['std'],
                            grouped['mean'] + grouped['std'],
                            alpha=0.2)
    else:
        grouped = plot_data.groupby(x_col)[metric].agg(['mean', 'std']).reset_index()
        ax.plot(grouped[x_col], grouped['mean'], marker='o', linewidth=2)
        ax.fill_between(grouped[x_col],
                        grouped['mean'] - grouped['std'],
                        grouped['mean'] + grouped['std'],
                        alpha=0.2)

    ax.set_xlabel(x_col)
    ax.set_ylabel(metric)
    ax.set_title(title or '{} vs {}'.format(metric, x_col))
    if split_by:
        ax.legend(title=split_by)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    return fig


def plot_heatmap(df, x_col, y_col, metric='f1', title=None, save_path=None):
    """
    Heatmap of metric vs two parameters.
    df should already be filtered to comparable conditions.
    """
    plot_data = df.dropna(subset=[metric])
    pivot = plot_data.pivot_table(values=metric, index=y_col, columns=x_col, aggfunc='mean')

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis', ax=ax,
                cbar_kws={'label': metric})

    ax.set_title(title or '{}: {} vs {}'.format(metric, y_col, x_col))
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    return fig


def plot_precision_recall(df, color_by=None, title=None, save_path=None, col_names=None):
    """
    Precision-recall scatter plot.
    Shows the trade-off between precision and recall.
    """
    precision, recall = col_names
    plot_data = df.dropna(subset=[precision, recall])

    fig, ax = plt.subplots(figsize=(8, 8))

    if color_by and color_by in plot_data.columns:
        # Categorical color
        if plot_data[color_by].dtype == 'object' or len(plot_data[color_by].unique()) < 10:
            for value in plot_data[color_by].unique():
                subset = plot_data[plot_data[color_by] == value]
                ax.scatter(subset[recall], subset[precision],
                           label=str(value), alpha=0.6, s=50)
            ax.legend(title=color_by)
        else:
            # Continuous color
            scatter = ax.scatter(plot_data[recall], plot_data[precision],
                                 c=plot_data[color_by], cmap='viridis',
                                 alpha=0.6, s=50)
            plt.colorbar(scatter, label=color_by)
    else:
        ax.scatter(plot_data[recall], plot_data[precision], alpha=0.6, s=50)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title or 'Precision vs Recall')
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    return fig


def plot_multiple_metrics(df, x_col, metrics, split_by=None, title=None, save_path=None):
    """
    Plot multiple metrics on the same graph vs one parameter.
    Useful for comparing SHD, precision, recall, F1 simultaneously.
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics), sharex=True)
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        plot_data = df.dropna(subset=[metric])

        if split_by and split_by in plot_data.columns:
            for value in sorted(plot_data[split_by].unique()):
                subset = plot_data[plot_data[split_by] == value]
                grouped = subset.groupby(x_col)[metric].agg(['mean', 'std']).reset_index()
                ax.plot(grouped[x_col], grouped['mean'],
                        marker='o', label=str(value), linewidth=2)
                ax.fill_between(grouped[x_col],
                                grouped['mean'] - grouped['std'],
                                grouped['mean'] + grouped['std'],
                                alpha=0.2)
        else:
            grouped = plot_data.groupby(x_col)[metric].agg(['mean', 'std']).reset_index()
            ax.plot(grouped[x_col], grouped['mean'], marker='o', linewidth=2)
            ax.fill_between(grouped[x_col],
                            grouped['mean'] - grouped['std'],
                            grouped['mean'] + grouped['std'],
                            alpha=0.2)

        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        if split_by and ax == axes[0]:
            ax.legend(title=split_by)

    axes[-1].set_xlabel(x_col)
    axes[0].set_title(title or 'Multiple metrics vs {}'.format(x_col))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    return fig


def plot_faceted_heatmaps(df, x_col, y_col, metric, facet_by, values=None,
                          title=None, save_path=None):
    """
    Create grid of heatmaps, one for each value of facet_by parameter.

    Example:
        plot_faceted_heatmaps(df, 'ntraj', 'len', 'f1',
                             facet_by='nodes', values=[5, 8, 12, 16])
    """
    if values is None:
        values = sorted(df[facet_by].unique())

    n_facets = len(values)
    ncols = min(4, n_facets)
    nrows = (n_facets + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)

    for idx, value in enumerate(values):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        subset = df[df[facet_by] == value].dropna(subset=[metric])
        if len(subset) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title('{}={} (no data)'.format(facet_by, value))
            continue

        pivot = subset.pivot_table(values=metric, index=y_col, columns=x_col, aggfunc='mean')
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='viridis', ax=ax,
                    cbar_kws={'label': metric})
        ax.set_title('{}={}'.format(facet_by, value))

    # Hide unused subplots
    for idx in range(n_facets, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis('off')

    fig.suptitle(title or '{} heatmaps: {} vs {} (by {})'.format(
        metric, y_col, x_col, facet_by), fontsize=14, y=1.00)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    return fig


def summary_stats(df, group_by, metrics=None):
    """Print summary statistics grouped by parameter."""
    if metrics is None:
        metrics = ['f1', 'precision', 'recall', 'shd_frac']

    available_metrics = [m for m in metrics if m in df.columns]

    print("\nSummary statistics grouped by {}:".format(group_by))
    print("=" * 60)

    for metric in available_metrics:
        print("\n{}:".format(metric.upper()))
        summary = df.groupby(group_by)[metric].agg(['count', 'mean', 'std', 'min', 'max'])
        print(summary)


# Example analysis workflows

def analyze_mir9(csv_path, output_dir='plots_mir9'):
    """Example analysis for MIR9 dataset."""
    df = load_mir9_data(csv_path)

    print("MIR9 Dataset:")
    print("  Rows: {}".format(len(df)))
    print("  Modes: {}".format(df['mode'].unique()))
    print("  Datapoints: {}".format(sorted(df['datapoints'].unique())))
    print("  Lengths: {}".format(sorted(df['trajectory_len'].unique())))
    print("  Frequencies: {}".format(sorted(df['freq'].unique())))

    # Filter to freq=1 to avoid mixing frequencies
    freq1 = filter_data(df, freq=1.0)

    plot_metric_vs_param(freq1, 'datapoints', 'f1', split_by='mode',
                         title='F1 vs Trajectories (freq=1)',
                         save_path='{}/f1_vs_datapoints.png'.format(output_dir))

    plot_metric_vs_param(freq1, 'trajectory_len', 'f1', split_by='mode',
                         title='F1 vs Length (freq=1)',
                         save_path='{}/f1_vs_length.png'.format(output_dir))

    plot_precision_recall(freq1, color_by='mode',
                          title='Precision-Recall (freq=1)',
                          save_path='{}/precision_recall.png'.format(output_dir),
                          col_names = ['prec', 'rec'])

    sync_freq1 = filter_data(df, mode='Sync', freq=1.0)
    plot_heatmap(sync_freq1, 'datapoints', 'trajectory_len', 'f1',
                 title='F1 Heatmap (Sync, freq=1)',
                 save_path='{}/heatmap_sync.png'.format(output_dir))

    async_freq1 = filter_data(df, mode='Async', freq=1.0)
    plot_heatmap(async_freq1, 'datapoints', 'trajectory_len', 'f1',
                 title='F1 Heatmap (Async, freq=1)',
                 save_path='{}/heatmap_async.png'.format(output_dir))

    print("\nPlots saved to {}".format(output_dir))


def analyze_tsv(tsv_path, output_dir='plots_tsv'):
    """Example analysis for TSV dataset with proper conditioning."""
    df = load_tsv_data(tsv_path)

    print("\nTSV Dataset:")
    print("  Rows: {} (valid: {})".format(len(df), df['f1'].notna().sum()))
    print("  Nodes: {}".format(sorted(df['nodes'].unique())))
    print("  Scoring: {}".format(df['score'].unique()))

    summary_stats(df, 'nodes')

    # Heatmap: nodes vs ntraj, with MDL, sf=1, len=5, tf=0.4 fixed
    mdl_fixed = filter_data(df, score='MDL', sf=1, len=5, tf=0.4)
    plot_heatmap(mdl_fixed, 'ntraj', 'nodes', 'f1',
                 title='F1: Nodes vs Trajectories (MDL, sf=1, len=5, tf=0.4)',
                 save_path='{}/heatmap_nodes_ntraj.png'.format(output_dir))

    # Compare algorithms with sf=1, ntraj=3, len=5, tf=0.4 fixed
    for_comparison = filter_data(df, sf=1, ntraj=3, len=5, tf=0.4)
    plot_metric_vs_param(for_comparison, 'nodes', 'f1', split_by='score',
                         title='F1 vs Nodes: MDL vs BDE (sf=1, ntraj=3, len=5, tf=0.4)',
                         save_path='{}/f1_nodes_by_score.png'.format(output_dir))

    # Multiple metrics with sf=1, ntraj=3, len=5, tf=0.4 fixed
    plot_multiple_metrics(for_comparison, 'nodes',
                          ['shd_frac', 'precision', 'recall', 'f1'],
                          split_by='score',
                          title='All metrics vs Nodes (sf=1, ntraj=3, len=5, tf=0.4)',
                          save_path='{}/all_metrics_vs_nodes.png'.format(output_dir))

    # Precision-recall with MDL, sf=1, ntraj=3, len=5, tf=0.4
    mdl_specific = filter_data(df, score='MDL', sf=1, ntraj=3, len=5, tf=0.4)
    plot_precision_recall(mdl_specific, color_by='nodes',
                          title='Precision-Recall (MDL, sf=1, ntraj=3, len=5, tf=0.4)',
                          save_path='{}/precision_recall.png'.format(output_dir),
                          col_names = ['precision', 'recall'])

    print("\nPlots saved to {}".format(output_dir))


if __name__ == '__main__':
    # Example usage

    #analyze_mir9('tables/MIR9_evaluation.csv', 'plots_mir9')
    analyze_tsv('tables/evaluation_asynchronous.tsv', 'plots_tsv_async')

    pass