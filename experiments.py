import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import numpy as np

from evaluate_DC import evaluate

def tsne(support_data, support_label, sampled_data, sampled_label, query_data, query_label):
    support_data = support_data.numpy() 
    support_label = support_label.numpy().astype(np.uint8)
    sampled_data = sampled_data.numpy() 
    sampled_label = sampled_label.numpy().astype(np.uint8)
    query_data = query_data
    query_label = query_label.astype(np.uint8)

    num_support = len(support_data)
    num_sampled = len(sampled_data)

    data = np.append(support_data, sampled_data, axis=0)
    data = np.append(data, query_data, axis=0)
    tsne_result = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(data)

    colors = np.array(['red', 'blue', 'orange', 'green', 'black'])
    colors_light = np.array(['pink', 'skyblue', 'yellow', 'springgreen', 'gray'])
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    ax[0].scatter(tsne_result[num_support:num_support+num_sampled, 0], tsne_result[num_support:num_support+num_sampled, 1], c=colors_light[sampled_label], s=2)
    ax[0].scatter(tsne_result[:num_support, 0], tsne_result[:num_support, 1], c=colors[support_label], marker="*", s=50)
    ax[0].set_title("Sampled Data")
    ax[0].plot()
    ax[1].scatter(tsne_result[num_support+num_sampled:, 0], tsne_result[num_support+num_sampled:, 1], c=colors_light[query_label], s=2)
    ax[1].scatter(tsne_result[:num_support, 0], tsne_result[:num_support, 1], c=colors[support_label], marker="*", s=50)
    ax[1].set_title("Query Data")
    ax[1].plot()
    plt.show()

acc_list, support_data, support_label, sampled_data, sampled_label, query_data, query_label = evaluate(dataset="miniImagenet", n_runs=3, n_shot=1, n_queries=300, alpha=0)
tsne(support_data, support_label, sampled_data, sampled_label, query_data, query_label)