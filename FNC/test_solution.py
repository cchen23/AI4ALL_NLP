import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

""" Get data"""
pd.set_option('display.max_colwidth', -1)

train_data = pd.read_csv("train_data.csv", encoding = "utf-8")
train_data.Stance.value_counts()/train_data.shape[0]
unrelated_train = train_data[train_data['Stance'] == 'unrelated']
discuss_train = train_data[train_data['Stance'] == 'discuss']
agree_train = train_data[train_data['Stance'] == 'agree']
disagree_train = train_data[train_data['Stance'] == 'disagree']

test_data = pd.read_csv("test_data.csv", encoding = "utf-8")
test_data.Stance.value_counts()/test_data.shape[0]
unrelated_test = test_data[test_data['Stance'] == 'unrelated']
discuss_test = test_data[test_data['Stance'] == 'discuss']
agree_test = test_data[test_data['Stance'] == 'agree']
disagree_test = test_data[test_data['Stance'] == 'disagree']

""" Proportion same
Choose based on closest proportion of shared words.
"""
def find_headline_in_article_proportion(example):
    headline_words = str.split(example['Headline'])
    article_words = str.split(example['articleBody'])
    counter = 0
    for word in headline_words:
        if word in article_words:
            counter += 1
    proportion = counter/len(headline_words)
    return proportion

# Let's apply the function to all examples in each of the four dataframes we created above.
# We will save the results to a list:
def compute_proportions(unrelated, discuss, agree, disagree):
    proportions_unrelated = []
    for i in range(unrelated.shape[0]):
        this_example = unrelated.iloc[i]
        proportions_unrelated.append(find_headline_in_article_proportion(this_example))
    proportions_related = []
    proportions_discuss= []
    for i in range(discuss.shape[0]):
        this_example = discuss.iloc[i]
        proportions_discuss.append(find_headline_in_article_proportion(this_example))
        proportions_related.append(find_headline_in_article_proportion(this_example))
    proportions_agree= []
    for i in range(agree.shape[0]):
        this_example = agree.iloc[i]
        proportions_agree.append(find_headline_in_article_proportion(this_example))
        proportions_related.append(find_headline_in_article_proportion(this_example))
    proportions_disagree= []
    for i in range(disagree.shape[0]):
        this_example = disagree.iloc[i]
        proportions_disagree.append(find_headline_in_article_proportion(this_example))
        proportions_related.append(find_headline_in_article_proportion(this_example))
    return {"unrelated":np.mean(proportions_unrelated), "discuss":np.mean(proportions_discuss), "agree":np.mean(proportions_agree), "disagree":np.mean(proportions_disagree), "related":np.mean(proportions_related)}

proportions = compute_proportions(unrelated_train, discuss_train, agree_train, disagree_train)

def make_predictions_proportions(unrelated, discuss, agree, disagree, proportions):
    proportion_unrelated = proportions["unrelated"]
    proportion_discuss = proportions["discuss"]
    proportion_agree = proportions["agree"]
    proportion_disagree = proportions["disagree"]
    proportion_related = proportions["related"]
    unrelated_count = unrelated.shape[0]
    unrelated_correct = 0
    discuss_count = discuss.shape[0]
    discuss_correct = 0
    agree_count = agree.shape[0]
    agree_correct = 0
    disagree_count = disagree.shape[0]
    disagree_correct = 0
    related_count = discuss_count + agree_count + disagree_count
    related_correct = 0
    for i in range(unrelated.shape[0]):
        this_example = unrelated.iloc[i]
        proportion = find_headline_in_article_proportion(this_example)
        if np.abs(proportion - proportion_unrelated) < np.abs(proportion - proportion_related):
            unrelated_correct += 1
    for i in range(discuss.shape[0]):
        this_example = discuss.iloc[i]
        proportion = find_headline_in_article_proportion(this_example)
        if np.abs(proportion - proportion_discuss) < np.min([np.abs(proportion - proportion_unrelated), np.abs(proportion - proportion_agree), np.abs(proportion - proportion_disagree)]):
            discuss_correct += 1
        if np.abs(proportion - proportion_related) < np.min(proportion - proportion_unrelated):
            related_correct += 1
    for i in range(agree.shape[0]):
        this_example = agree.iloc[i]
        proportion = find_headline_in_article_proportion(this_example)
        if np.abs(proportion - proportion_agree) < np.min([np.abs(proportion - proportion_unrelated), np.abs(proportion - proportion_discuss), np.abs(proportion - proportion_disagree)]):
            agree_correct += 1
        if np.abs(proportion - proportion_related) < np.min(proportion - proportion_unrelated):
            related_correct += 1
    for i in range(disagree.shape[0]):
        this_example = disagree.iloc[i]
        proportion = find_headline_in_article_proportion(this_example)
        if np.abs(proportion - proportion_disagree) < np.min([np.abs(proportion - proportion_unrelated), np.abs(proportion - proportion_agree), np.abs(proportion - proportion_discuss)]):
            disagree_correct += 1
        if np.abs(proportion - proportion_related) < np.min(proportion - proportion_unrelated):
            related_correct += 1
    return {"unrelated":unrelated_correct/unrelated_count, "related":related_correct/related_count, "discuss":discuss_correct/discuss_count, "agree":agree_correct/agree_count, "disagree":disagree_correct/disagree_count}

predictions_proportionoverlap = make_predictions_proportions(unrelated_test, discuss_test, agree_test, disagree_test, proportions)
#{'unrelated': 0.9008665322360891, 'related': 0.6357587768969423, 'discuss': 0.0038082437275985663, 'agree': 0.4703100367840252, 'disagree': 0.21377331420373027}

""" Jaccard similarity """
def compute_jaccard_index(example):
    headline_words = set(str.split(example['Headline']))
    article_words = set(str.split(example['articleBody']))
    return len(headline_words.intersection(article_words)) / len(headline_words.union(article_words))

def compute_average_jaccard_indices(unrelated, discuss, agree, disagree):
    proportions_unrelated = []
    for i in range(unrelated.shape[0]):
        this_example = unrelated.iloc[i]
        proportions_unrelated.append(compute_jaccard_index(this_example))
    proportions_related = []
    proportions_discuss= []
    for i in range(discuss.shape[0]):
        this_example = discuss.iloc[i]
        proportions_discuss.append(compute_jaccard_index(this_example))
        proportions_related.append(compute_jaccard_index(this_example))
    proportions_agree= []
    for i in range(agree.shape[0]):
        this_example = agree.iloc[i]
        proportions_agree.append(compute_jaccard_index(this_example))
        proportions_related.append(compute_jaccard_index(this_example))
    proportions_disagree= []
    for i in range(disagree.shape[0]):
        this_example = disagree.iloc[i]
        proportions_disagree.append(compute_jaccard_index(this_example))
        proportions_related.append(compute_jaccard_index(this_example))
    return {"unrelated":np.mean(proportions_unrelated), "discuss":np.mean(proportions_discuss), "agree":np.mean(proportions_agree), "disagree":np.mean(proportions_disagree), "related":np.mean(proportions_related)}

jaccard_indices = compute_average_jaccard_indices(unrelated_train, discuss_train, agree_train, disagree_train)

def make_predictions_jaccard_indices(unrelated, discuss, agree, disagree, jaccard_indices):
    proportion_unrelated = jaccard_indices["unrelated"]
    proportion_discuss = jaccard_indices["discuss"]
    proportion_agree = jaccard_indices["agree"]
    proportion_disagree = jaccard_indices["disagree"]
    proportion_related = jaccard_indices["related"]
    unrelated_count = unrelated.shape[0]
    unrelated_correct = 0
    discuss_count = discuss.shape[0]
    discuss_correct = 0
    agree_count = agree.shape[0]
    agree_correct = 0
    disagree_count = disagree.shape[0]
    disagree_correct = 0
    related_count = discuss_count + agree_count + disagree_count
    related_correct = 0
    for i in range(unrelated.shape[0]):
        this_example = unrelated.iloc[i]
        proportion = compute_jaccard_index(this_example)
        if np.abs(proportion - proportion_unrelated) < np.abs(proportion - proportion_related):
            unrelated_correct += 1
    for i in range(discuss.shape[0]):
        this_example = discuss.iloc[i]
        proportion = compute_jaccard_index(this_example)
        if np.abs(proportion - proportion_discuss) < np.min([np.abs(proportion - proportion_unrelated), np.abs(proportion - proportion_agree), np.abs(proportion - proportion_disagree)]):
            discuss_correct += 1
        if np.abs(proportion - proportion_related) < np.min(proportion - proportion_unrelated):
            related_correct += 1
    for i in range(agree.shape[0]):
        this_example = agree.iloc[i]
        proportion = compute_jaccard_index(this_example)
        if np.abs(proportion - proportion_agree) < np.min([np.abs(proportion - proportion_unrelated), np.abs(proportion - proportion_discuss), np.abs(proportion - proportion_disagree)]):
            agree_correct += 1
        if np.abs(proportion - proportion_related) < np.min(proportion - proportion_unrelated):
            related_correct += 1
    for i in range(disagree.shape[0]):
        this_example = disagree.iloc[i]
        proportion = compute_jaccard_index(this_example)
        if np.abs(proportion - proportion_disagree) < np.min([np.abs(proportion - proportion_unrelated), np.abs(proportion - proportion_agree), np.abs(proportion - proportion_discuss)]):
            disagree_correct += 1
        if np.abs(proportion - proportion_related) < np.min(proportion - proportion_unrelated):
            related_correct += 1
    return {"unrelated":unrelated_correct/unrelated_count, "related":related_correct/related_count, "discuss":discuss_correct/discuss_count, "agree":agree_correct/agree_count, "disagree":disagree_correct/disagree_count}

predictions_jaccardindices = make_predictions_jaccard_indices(unrelated_test, discuss_test, agree_test, disagree_test, jaccard_indices)
# {'unrelated': 0.8805929478445692, 'related': 0.5523782559456398, 'discuss': 0.06294802867383513, 'agree': 0.3536521282186022, 'disagree': 0.15208034433285508}

"""TODO"""
""" Reuse code for different methods """
""" Use word vectors """
""" Clean up text """
""" N-gram overlap """
""" Naive Bayes """
""" Decision trees """
""" Nearest neighbors """
