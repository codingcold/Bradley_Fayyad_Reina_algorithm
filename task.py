import sys
import time
import random
from sklearn.cluster import KMeans
import numpy as np
import copy
import itertools


def point_dis(point, cluster):
    # Extract the data from the dictionary
    sum_1, sum_sq_1, n_1 = cluster["SUM"], cluster["SUMSQ"], len(cluster["N"])
    # Calculate the centroid and variance
    centroid = sum_1 / n_1
    variance = (sum_sq_1 / n_1) - (centroid ** 2)
    # Calculate the standardized difference between the point and the centroid
    z = (point - centroid) / np.sqrt(variance)
    # Calculate the Mahalanobis distance
    point_distance = np.sqrt(np.dot(z, z))
    return point_distance

def cluster_dis(cluster1, cluster2):
    # Extract the data from the dictionaries
    sum1, sum_1, n1 = cluster1["SUM"], cluster1["SUMSQ"], len(cluster1["N"])
    sum2, sum_2, n2 = cluster2["SUM"], cluster2["SUMSQ"], len(cluster2["N"])
    # Calculate the centroids
    centroid_1 = sum1 / n1
    centroid_2 = sum2 / n2
    # Calculate the covariance matrix
    cov_1 = sum_1 / n1 - centroid_1 ** 2
    cov_2 = sum_2 / n2 - centroid_2 ** 2
    cov = ((cov_1 * n1) + (cov_2 * n2)) / (n1 + n2)
    # Calculate the standardized difference between the centroids
    z = (centroid_1 - centroid_2) / np.sqrt(cov)
    # Calculate the Mahalanobis distance
    cluster_distance = np.sqrt(np.dot(z, z))
    return cluster_distance

def round_result(DS, CS, RS):
    DS_total = sum([len(cluster["N"]) for cluster in DS.values()])
    CS_total = sum([len(cluster["N"]) for cluster in CS.values()])
    CS_cluster = len(CS)
    RS_total = len(RS)
    return DS_total, CS_cluster, CS_total, RS_total

input_file = sys.argv[1]
num = int(sys.argv[2])
output_file = sys.argv[3]

start_time = time.time()
res = "The intermediate results:\n"
l1 = []
with open(input_file) as in_file:
    raw = in_file.readlines()
raw = list(map(lambda x: x.strip("\n").split(','), raw))
raw = [(int(dd[0]), tuple(list(map(lambda x: float(x), dd[2:])))) for dd in raw]
raw_dict = dict(raw)
raw_dict_reversed = dict(zip(list(raw_dict.values()), list(raw_dict.keys())))
raw = list(map(lambda x: np.array(x), list(raw_dict.values())))
random.shuffle(raw)
len_data = round(len(raw) / 5)

data = raw[0:len_data]
km = KMeans(n_clusters=num * 25).fit(data)

dict_clu = {label: sum(1 for x in km.labels_ if x == label) for label in set(km.labels_)}
RS_index = [i for i, x in enumerate(km.labels_) if dict_clu[x] < 20]
l1.extend([data[i] for i in RS_index])
data = [data[i] for i in range(len(data)) if i not in RS_index]

km = KMeans(n_clusters=num).fit(data)

dict_clu = {}

for label, data_point in zip(km.labels_, data):
    if label not in dict_clu:
        dict_clu[label] = {"N": [], "SUM": np.zeros_like(data_point), "SUMSQ": np.zeros_like(data_point)}
    dict_clu[label]["N"].append(raw_dict_reversed[tuple(data_point)])
    dict_clu[label]["SUM"] += data_point
    dict_clu[label]["SUMSQ"] += data_point ** 2
dc = dict_clu

if l1:
    if len(l1) > 1:
        km = KMeans(n_clusters=len(l1) - 1).fit(l1)
    else:
        km = KMeans(n_clusters=len(l1)).fit(l1)
    dict_clu = dict()
    for label in km.labels_:
        dict_clu[label] = dict_clu.get(label, 0) + 1
    RS_key = []
    for key in dict_clu:
        if dict_clu[key] == 1:
            RS_key.append(key)
    RS_index = []
    if RS_key:
        for key in RS_key:
            RS_index.append(list(km.labels_).index(key))
    cluster_pair = tuple(zip(km.labels_, l1))
    l2 = dict()
    for pair in cluster_pair:
        if pair[0] not in RS_key:
            if pair[0] not in l2:
                l2[pair[0]] = dict()
                l2[pair[0]]["N"] = [raw_dict_reversed[tuple(pair[1])]]
                l2[pair[0]]["SUM"] = pair[1]
                l2[pair[0]]["SUMSQ"] = pair[1] ** 2
            else:
                l2[pair[0]]["N"].append(raw_dict_reversed[tuple(pair[1])])
                l2[pair[0]]["SUM"] += pair[1]
                l2[pair[0]]["SUMSQ"] += pair[1] ** 2
    new_RS = []
    for index in reversed(sorted(RS_index)):
        new_RS.append(l1[index])
    l1 = copy.deepcopy(new_RS)

DS_total, l2_clu, CS_total, RS_total = round_result(dc, l2, l1)
res += "Round 1: " + str(DS_total) + "," + str(l2_clu) + "," + str(CS_total) + "," + str(RS_total) + "\n"

for _ in range(0, 4):
    if _ == 3:
        data = raw[len_data * 4:]
    else:
        data = raw[len_data * (_ + 1):len_data * (_ + 2)]

    dc_i = set()
    distance_dicts = [{cluster: point_dis(point, dc[cluster]) for cluster in dc} for point in data]
    min_distances = [min(list(distance_dict.values())) for distance_dict in distance_dicts]
    clusters = [min(distance_dict, key=distance_dict.get) for distance_dict in distance_dicts]
    dc_i.update(
        [i for i, m_distance in enumerate(min_distances) if m_distance < 2 * (len(data[i]) ** (1 / 2))])
    for i, cluster in enumerate(clusters):
        if i not in dc_i:
            continue
        dc[cluster]["N"].append(raw_dict_reversed[tuple(data[i])])
        dc[cluster]["SUM"] += data[i]
        dc[cluster]["SUMSQ"] += data[i] ** 2

    l2_i = set()
    if l2:
        for i, point in enumerate(data):
            if i not in dc_i:
                distance_dict = {cluster: point_dis(point, l2[cluster]) for cluster in l2}
                cluster = min(distance_dict, key=distance_dict.get)
                m_distance = distance_dict[cluster]
                if m_distance < 2 * (len(point) ** (1 / 2)):
                    l2[cluster]["N"].append(raw_dict_reversed[tuple(point)])
                    l2[cluster]["SUM"] += point
                    l2[cluster]["SUMSQ"] += point ** 2
                    l2_i.add(i)

    try:
        all_index = l2_i.union(dc_i)
    except NameError:
        all_index = dc_i
    for i in range(len(data)):
        if i not in all_index:
            l1.append(data[i])

    if l1:
        if len(l1) > 1:
            k_means = KMeans(n_clusters=len(l1) - 1).fit(l1)
        else:
            k_means = KMeans(n_clusters=len(l1)).fit(l1)
        CS_cluster_set = set(l2.keys())
        RS_cluster_set = set(k_means.labels_)
        intersection = CS_cluster_set.intersection(RS_cluster_set)
        union = CS_cluster_set.union(RS_cluster_set)
        change_dict = dict()
        # Generate a set of unique random integers
        random_ints = set(random.sample(range(100, len(data)), len(intersection)))
        # Map the intersection elements to the random integers
        change_dict = {ii: random_int for ii, random_int in zip(intersection, random_ints)}
        # Add the random integers to the union set
        union = union.union(random_ints)
        # Extract the labels assigned by the k-means clustering algorithm
        labels = list(k_means.labels_)
        labels = [change_dict[label] if label in change_dict else label for label in labels]

        dict_clu = dict()
        for label in labels:
            dict_clu[label] = dict_clu.get(label, 0) + 1
        RS_key = []
        for key in dict_clu:
            if dict_clu[key] == 1:
                RS_key.append(key)
        RS_index = []
        if RS_key:
            for key in RS_key:
                RS_index.append(labels.index(key))
        cluster_pair = tuple(zip(labels, l1))
        for pair in cluster_pair:
            if pair[0] not in RS_key:
                if pair[0] not in l2:
                    l2[pair[0]] = dict()
                    l2[pair[0]]["N"] = [raw_dict_reversed[tuple(pair[1])]]
                    l2[pair[0]]["SUM"] = pair[1]
                    l2[pair[0]]["SUMSQ"] = pair[1] ** 2
                else:
                    l2[pair[0]]["N"].append(raw_dict_reversed[tuple(pair[1])])
                    l2[pair[0]]["SUM"] += pair[1]
                    l2[pair[0]]["SUMSQ"] += pair[1] ** 2
        new_RS = []
        for index in reversed(sorted(RS_index)):
            new_RS.append(l1[index])
        l1 = copy.deepcopy(new_RS)

    flag = True
    while flag:
        flag = False
        for c1, c2 in itertools.combinations(list(l2.keys()), 2):
            m_distance = cluster_dis(l2[c1], l2[c2])
            if m_distance < 2 * (len(l2[c1]["SUM"]) ** (1 / 2)):
                l2[c1]["N"] = l2[c1]["N"] + l2[c2]["N"]
                l2[c1]["SUM"] += l2[c2]["SUM"]
                l2[c1]["SUMSQ"] += l2[c2]["SUMSQ"]
                l2.pop(c2)
                flag = True
                break

    l2_clu = list(l2.keys())
    if _ == 3 and l2:
        for cluster_cs in l2_clu:
            min_distance = float('inf')
            closest_cluster = None
            for cluster_ds in dc:
                distance = cluster_dis(dc[cluster_ds], l2[cluster_cs])
                if distance < min_distance:
                    min_distance = distance
                    closest_cluster = cluster_ds
            if min_distance < 2 * len(l2[cluster_cs]["SUM"]) ** (1 / 2):
                dc[closest_cluster]["N"] = dc[closest_cluster]["N"] + l2[cluster_cs]["N"]
                dc[closest_cluster]["SUM"] += l2[cluster_cs]["SUM"]
                dc[closest_cluster]["SUMSQ"] += l2[cluster_cs]["SUMSQ"]
                l2.pop(cluster_cs)

    DS_total, l2_clu, CS_total, RS_total = round_result(dc, l2, l1)
    res += "Round " + str(_ + 2) + ": " + str(DS_total) + "," + str(l2_clu) + "," + str(CS_total) + "," + str(
        RS_total) + "\n"
res += "\nThe clustering results:\n"
for cluster in dc:
    dc[cluster]["N"] = set(dc[cluster]["N"])
if l2:
    for cluster in l2:
        l2[cluster]["N"] = set(l2[cluster]["N"])

RS_set = set()
for point in l1:
    RS_set.add(raw_dict_reversed[tuple(point)])

for point in range(len(raw_dict)):
    if point in RS_set:
        res += str(point) + ",-1\n"
    else:
        for cluster in dc:
            if point in dc[cluster]["N"]:
                res += str(point) + "," + str(cluster) + "\n"
                break
        for cluster in l2:
            if point in l2[cluster]["N"]:
                res += str(point) + ",-1\n"
                break

with open(output_file, "w") as out_file:
    out_file.writelines(res)

duration = time.time() - start_time
print("Duration: " + str(duration))