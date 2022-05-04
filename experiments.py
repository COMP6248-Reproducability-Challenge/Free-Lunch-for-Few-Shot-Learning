import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import numpy as np
import os

from evaluate_DC import evaluate

def tsne(support_data, support_label, sampled_data, sampled_label, query_data, query_label):
    support_data = support_data.cpu().numpy() 
    support_label = support_label.cpu().numpy().astype(np.uint8)
    sampled_data = sampled_data.cpu().numpy() 
    sampled_label = sampled_label.cpu().numpy().astype(np.uint8)
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

###Plotting Graphs###

#Figure 1: Accuracy when increasing the power in Tukey's Transformation
def tukey_graph():
    #can change, lambda doesnt work if = 0 or negative
    lambdas = [0.25, 0.5, 0.75, 1, 2]

    acc_with_genf = []
    n_gen = 0
    for l in lambdas:
        acc = evaluate(dataset='miniImagenet', classifier='logistic', n_ways=5, n_shot=1, n_queries=15, n_runs=20, lamb=l, k=2, alpha=0.21, num_features=750)
        acc_with_genf.append(np.mean(acc[0]))
    print(acc_with_genf)

    acc_wo_genf = []
    n_gen = 0
    for l in lambdas:
        acc = evaluate(dataset='miniImagenet', classifier='logistic', n_ways=5, n_shot=1, n_queries=15, n_runs=20, lamb=l, k=2, alpha=0.21, num_features=0)
        acc_wo_genf.append(np.mean(acc[0])) 
    print(acc_wo_genf)


    plt.figure(figsize=(10, 10))
    plt.plot(lambdas, acc_with_genf, label='Training with generated features')
    plt.plot(lambdas, acc_wo_genf, label='Training without generated features')

    plt.xlabel('Values of power in Tukey Transformation', fontsize=13)
    plt.ylabel('Test accuracy (5way-1shot)', fontsize=13)
    plt.legend(prop={'size': 12})

    plt.savefig('images/tukeygraph.png')
    

#Figure 2: Accuracy when increasing the number of generated features
#with or without Tukey's transformation
#5ways, 1shot 
#The original feature can berecovered by setting λ as 1
def vary_n_generation(): 
    n_generations = [ 0, 10, 50, 100, 150, 300, 500, 650, 750]

    accs_no_tukey = []
    for n in n_generations:
        acc = evaluate(dataset='miniImagenet', classifier='logistic', n_ways=5, n_shot=1, n_queries=15, n_runs=1000, lamb=1, k=2, alpha=0.21, num_features=n)
        accs_no_tukey.append(np.mean(acc[0]))
    print(accs_no_tukey)
    
    accs_with_tukey = []
    for n1 in n_generations:
        acc = evaluate(dataset='miniImagenet', classifier='logistic', n_ways=5, n_shot=1, n_queries=15, n_runs=1000, lamb=0.5, k=2, alpha=0.21, num_features=n1)
        accs_with_tukey.append(np.mean(acc[0]))
    print(accs_with_tukey)

    plt.figure(figsize=(10, 10))
    plt.plot(n_generations, accs_no_tukey, label='training w/o Tukey transformation')
    plt.plot(n_generations, accs_with_tukey, label='training w Tukey transformation')

    plt.xlabel('Number of generated features per class', fontsize=13)
    plt.ylabel('Test accuracy (5way-1shot)', fontsize=13)
    plt.legend(prop={'size': 12})

    plt.savefig('images/ngraph.png')


#Figure 3: The effect of different values of k.
def k_graph(): 
    k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    accs_mini = []
    for kvalue in k_values:
        acc = evaluate(dataset='miniImagenet', classifier='logistic', n_ways=5, n_shot=1, n_queries=15, n_runs=1000, lamb=0.5, k=kvalue, alpha=0.21, num_features=750)
        accs_mini.append(np.mean(acc[0]))
    print(accs_mini)
    
    accs_cub = []
    for kvalue2 in k_values:
        acc = evaluate(dataset='miniImagenet', classifier='logistic', n_ways=5, n_shot=1, n_queries=15, n_runs=1000, lamb=0.5, k=kvalue2, alpha=0.21, num_features=750)
        accs_cub.append(np.mean(acc[0]))
    print(accs_cub)

    plt.figure(figsize=(10, 10))
    plt.plot(k_values, accs_mini, label='miniImageNet')
    plt.plot(k_values, accs_cub, label='CUB')

    plt.xlabel('Number of retrieved base class statistics k', fontsize=13)
    plt.ylabel('Test accuracy (5way-1shot)', fontsize=13)
    plt.legend(prop={'size': 12})

    plt.savefig('images/kgraph.png')

#Figure 4: The effect of different values of alpha.
def alpha_graph(): 
    alpha_values = [i for i in np.arange(0,0.4,0.05)]

    accs_mini = []
    for a1 in alpha_values:
        acc = evaluate(dataset='miniImagenet', classifier='logistic', n_ways=5, n_shot=1, n_queries=15, n_runs=1000, lamb=0.5, k=2, alpha=a1, num_features=750)
        accs_mini.append(np.mean(acc[0]))
    print(accs_mini)
    
    accs_CUB = []
    for a2 in alpha_values:
        acc = evaluate(dataset='miniImagenet', classifier='logistic', n_ways=5, n_shot=1, n_queries=15, n_runs=1000, lamb=0.5, k=2, alpha=a2, num_features=750)
        accs_CUB.append(np.mean(acc[0]))
    print(accs_CUB)


    plt.figure(figsize=(10, 10))
    plt.plot(alpha_values, accs_mini, label='miniImageNet')
    plt.plot(alpha_values, accs_CUB, label='CUB')

    plt.xlabel('Number of alpha added on covariance matrix', fontsize=13)
    plt.ylabel('Test accuracy (5way-1shot)', fontsize=13)
    plt.legend(prop={'size': 12})

    plt.savefig('images/aphagraph.png')


if __name__ == "__main__":
    # T-SNE (Figure 2)
    # acc_list, support_data, support_label, sampled_data, sampled_label, query_data, query_label = evaluate(dataset="miniImagenet", n_runs=3, n_shot=1, n_queries=300, alpha=0)
    # tsne(support_data, support_label, sampled_data, sampled_label, query_data, query_label)

    # Performance Table (Table 2)
    # evaluate(dataset='miniImagenet', classifier='logistic', n_ways=5, n_shot=1, n_queries=15, n_runs=10000, lamb=0.5, k=2, alpha=0.21, num_features=750)
    # evaluate(dataset='miniImagenet', classifier='logistic', n_ways=5, n_shot=5, n_queries=15, n_runs=10000, lamb=0.5, k=2, alpha=0.21, num_features=750)
    # evaluate(dataset='miniImagenet', classifier='svm', n_ways=5, n_shot=1, n_queries=15, n_runs=10000, lamb=0.5, k=2, alpha=0.21, num_features=750)
    # evaluate(dataset='miniImagenet', classifier='svm', n_ways=5, n_shot=5, n_queries=15, n_runs=10000, lamb=0.5, k=2, alpha=0.21, num_features=750)
   
    # evaluate(dataset='CUB', classifier='logistic', n_ways=5, n_shot=1, n_queries=15, n_runs=10000, lamb=0.5, k=2, alpha=0.3, num_features=750)
    # evaluate(dataset='CUB', classifier='logistic', n_ways=5, n_shot=5, n_queries=15, n_runs=10000, lamb=0.5, k=2, alpha=0.3, num_features=750)
    # evaluate(dataset='CUB', classifier='svm', n_ways=5, n_shot=1, n_queries=15, n_runs=10000, lamb=0.5, k=2, alpha=0.3, num_features=750)
    # evaluate(dataset='CUB', classifier='svm', n_ways=5, n_shot=5, n_queries=15, n_runs=10000, lamb=0.5, k=2, alpha=0.3, num_features=750)

    # # # Ablation Study Table (Table 4) (Without Tukey, Without Generated Features, or Without Both)
    # evaluate(dataset='miniImagenet', classifier='logistic', n_ways=5, n_shot=1, n_queries=15, n_runs=10, lamb=1, k=2, alpha=0.21, num_features=750)
    # evaluate(dataset='miniImagenet', classifier='logistic', n_ways=5, n_shot=1, n_queries=15, n_runs=10000, lamb=0.5, k=2, alpha=0.21, num_features=0)
    # evaluate(dataset='miniImagenet', classifier='logistic', n_ways=5, n_shot=1, n_queries=15, n_runs=10000, lamb=1, k=2, alpha=0.21, num_features=0)
    # evaluate(dataset='miniImagenet', classifier='logistic', n_ways=5, n_shot=5, n_queries=15, n_runs=10000, lamb=1, k=2, alpha=0.21, num_features=750)
    # evaluate(dataset='miniImagenet', classifier='logistic', n_ways=5, n_shot=5, n_queries=15, n_runs=10000, lamb=0.5, k=2, alpha=0.21, num_features=0)
    # evaluate(dataset='miniImagenet', classifier='logistic', n_ways=5, n_shot=5, n_queries=15, n_runs=10000, lamb=1, k=2, alpha=0.21, num_features=0)

    tukey_graph()
    # vary_n_generation()
    #k_graph()
    #alpha_graph()

    #letting my pc rest
    # os.system('shutdown /s /t 100') 