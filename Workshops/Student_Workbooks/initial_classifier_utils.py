import pandas as pd
import numpy as np

def merge_data(stances_filename, bodies_filename, merged_filename):
    stances = pd.read_csv(stances_filename, encoding = "utf-8")
    bodies = pd.read_csv(bodies_filename, encoding = "utf-8")
    data = pd.merge(bodies, stances, on='Body ID')
    data.to_csv(merged_filename, index=False, encoding = "utf-8")

def find_headline_in_article_proportion(example):
    headline_words = str.split(example['Headline'])
    article_words = str.split(example['articleBody'])
    counter = 0
    for word in headline_words:
        if word in article_words:
            counter += 1
    proportion = counter/len(headline_words)
    return proportion

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




