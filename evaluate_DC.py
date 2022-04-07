import time
import pickle
import numpy as np
import torch
from tqdm import tqdm
from torch.distributions import MultivariateNormal
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

use_gpu = torch.cuda.is_available()
device = 'cuda' if use_gpu else 'cpu'

def tukey_transform(x, lamb):
    if lamb == 0:
        return torch.log(x)
    return x ** lamb

def distribution_calibration(support, base_means, base_cov, k=2, alpha=0.21):
    distances = torch.sqrt(torch.sum((base_means - support) ** 2, dim=1))
    # print("Distances:",distances.shape)
    ordered_indices = torch.argsort(distances) # Ascending order
    # print("Ordered Indices:", ordered_indices)
    cali_mean = (torch.sum(base_means[ordered_indices[:k]], dim=0) + support) / (k+1)
    cali_cov = torch.sum(base_cov[ordered_indices[:k]], dim=0) / k + alpha
    # print("Calibrated Mean:", cali_mean.shape)
    # print("Calibrated Cov:", cali_cov.shape)
    return cali_mean, cali_cov

def evaluate(dataset='miniImagenet', classifier='logistic', n_ways=5, n_shot=1, n_queries=15, n_runs=10000, lamb=0.5, k=2, alpha=0.21, num_features=500):
    """
    lamb: Tukey's Transformation parameter
    k: Top-k closest classes used for distribution calibration
    alpha: Degree of dispersion of sampled features
    num_features: Total number of features to sample
    """
    # ---- data loading
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples

    import FSLTask
    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries}
    FSLTask.loadDataSet(dataset)
    FSLTask.setRandomStates(cfg)
    ndatas = FSLTask.GenerateRunSet(end=n_runs, cfg=cfg)
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, n_ways).clone().view(n_runs,
                                                                                                        n_samples)
    # print("labels:",labels)
    # print("labels:",labels.shape)

    # ---- Base class statistics
    base_means = []
    base_cov = []
    base_features_path = "./checkpoints/%s/WideResNet28_10_S2M2_R/last/base_features.plk"%dataset
    with open(base_features_path, 'rb') as f:
        data = pickle.load(f)
        for key in data.keys():
            feature = torch.tensor(np.array(data[key]))
            mean = torch.mean(feature, axis=0)
            cov = torch.cov(feature.T)
            base_means.append(mean)
            base_cov.append(cov)
    base_means = torch.stack(base_means).to(device)
    base_cov = torch.stack(base_cov).to(device)
    # print("Base Means:",base_means.shape)
    # print("Base Cov:",base_cov.shape)

    # ---- classification for each task
    acc_list = []
    print('Start classification for %d tasks...'%(n_runs))
    for i in tqdm(range(n_runs)):
        """Most time is spent on distribution calibration, so we do it on gpu, 
        then convert the tensors back to numpy for using sklearn models"""
        # st = time.time()
        support_data = ndatas[i][:n_lsamples].to(device)
        support_label = labels[i][:n_lsamples].to(device)
        query_data = ndatas[i][n_lsamples:]
        query_label = labels[i][n_lsamples:]

        # print("Support Data:",support_data.shape)
        # print("Support Label:",support_label.shape)
        # print("Query Label:",query_label)

        support_data = tukey_transform(support_data, lamb)
        query_data = tukey_transform(query_data, lamb)

        feature_per_data = num_features // n_shot
        sampled_data = torch.zeros((feature_per_data * n_lsamples, support_data.shape[1])).to(device)
        sampled_label = torch.zeros((feature_per_data * n_lsamples)).to(device)
        # print(time.time() - st)

        # st = time.time()
        for i in range(len(support_data)):
            # Calibrate distribution, then sample from the distribution
            cali_mean, cali_cov = distribution_calibration(support_data[i], base_means, base_cov, k=k, alpha=alpha)
            distribution = MultivariateNormal(cali_mean, cali_cov + torch.eye(cali_cov.shape[0]) * 0.01)
            sampled_data[i*feature_per_data:(i+1)*feature_per_data] = distribution.sample((feature_per_data,))
            sampled_label[i*feature_per_data:(i+1)*feature_per_data] = torch.full((feature_per_data,), support_label[i])
        
        # Concat
        training_data = torch.concat((support_data, sampled_data))
        training_label = torch.concat((support_label, sampled_label))
        # print("Training data:",training_data.shape)
        # print("Training Label:",training_label.shape)
        # print(time.time() - st)

        # Train
        # st = time.time()
        training_data = training_data.cpu().numpy()
        training_label = training_label.cpu().numpy()
        query_data = query_data.numpy()
        query_label = query_label.numpy()

        if classifier == "logistic":
            model = LogisticRegression(max_iter=1000)
        elif classifier == "svm":
            model = SVC(max_iter=1000)
        elif classifier == "naive_bayes":
            model = GaussianNB()
        elif classifier == "tree":
            model = DecisionTreeClassifier()
        else:
            raise ValueError(f"Unknown Classifier: {classifier}")

        model.fit(X=training_data, y=training_label)

        predicts = model.predict(query_data)
        acc = np.mean(predicts == query_label)
        acc_list.append(acc)
        # print(time.time() - st)

    print(f"{dataset} | {n_runs} runs | {n_ways} ways | {n_shot} shots | {n_queries} queries | classifier: {classifier} | lambda: {lamb} | k: {k} | alpha: {alpha} | num_features: {num_features}")
    print(f"Accuracy: {np.mean(acc_list)} | 95% Confidence Level: {1.96 * 100 * np.std(acc_list) / np.sqrt(len(acc_list))}")

    return acc_list, support_data, support_label, sampled_data, sampled_label, query_data, query_label

if __name__ == "__main__":
    evaluate()