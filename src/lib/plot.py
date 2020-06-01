import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.manifold import TSNE


def plot_frequency_map(items):
    frequency_map = {}
    for item in items:
        if item in frequency_map.keys():
            frequency_map[item] += 1
        else:
            frequency_map[item] = 1
    sns.barplot(x=list(frequency_map.keys()), y=list(frequency_map.values()))


def get_top_value_key_value(input_dict, output_size=1000):
    dict_sorted_by_value = sorted(input_dict.items(), reverse=True, key=lambda x: x[1])
    top_value_key = list(map(lambda x: x[0], dict_sorted_by_value[:output_size]))
    top_value = list(map(lambda x: x[1], dict_sorted_by_value[:output_size]))
    return top_value_key, top_value


def visualise_top_frequent_word(word_requency_map, plot_size=1000, print_size=100):
    (word_frequency_top_words, word_frequency_top_frequency) = get_top_value_key_value(
        word_requency_map, output_size=plot_size
    )

    sns.distplot(word_frequency_top_frequency)
    pprint(word_frequency_top_words[:print_size])


def visualise_w2v_frequent_word_tsne(w2v_model, word_frequency_map, size=1000):
    (word_frequency_top_words, word_frequency_top_frequency) = get_top_value_key_value(
        word_frequency_map, output_size=size
    )
    top_frequent_words_w2v = w2v_model[word_frequency_top_words]

    top_frequent_words_tsne = TSNE(n_components=2, random_state=0).fit_transform(
        top_frequent_words_w2v
    )

    sns.scatterplot(x=top_frequent_words_tsne[:, 0], y=top_frequent_words_tsne[:, 1])


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
