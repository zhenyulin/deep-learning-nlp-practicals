import seaborn as sns
import matplotlib.pyplot as plt


def plot_frequency_map(items):
    frequency_map = {}
    for item in items:
        if item in frequency_map.keys():
            frequency_map[item] += 1
        else:
            frequency_map[item] = 1
    plt.figure(figsize=(4, 4))
    sns.barplot(x=list(frequency_map.keys()), y=list(frequency_map.values()))


def plot_tf_metrics(training_history, metrics_name):
    plt.figure(figsize=(12, 8))
    sns.set(style="darkgrid", color_codes=True)
    x = range(len(training_history.history[metrics_name]))
    sns.lineplot(x=x, y=training_history.history[metrics_name], label=metrics_name)
    sns.lineplot(
        x=x,
        y=training_history.history[f"val_{metrics_name}"],
        label=f"val_{metrics_name}",
    )
