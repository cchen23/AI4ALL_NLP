import pandas as pd
import sys

# Adjust settings so that we can fully see the dataset below
pd.set_option('display.max_colwidth', -1)

def merge_data(stances_filename, bodies_filename, merged_filename):
    stances = pd.read_csv(stances_filename, encoding = "utf-8")
    bodies = pd.read_csv(bodies_filename, encoding = "utf-8")
    data = pd.merge(bodies, stances, on='Body ID')
    data.to_csv(merged_filename, index=False, encoding = "utf-8")

if __name__ == '__main__':
    stances_filename = sys.argv[1]
    bodies_filename = sys.argv[2]
    merged_filename = sys.argv[3]
    # merge_data("../Workshops/train_stances.csv", "../Workshops/train_bodies.csv", "train_data.csv")
    # merge_data("../Workshops/competition_test_stances.csv", "../Workshops/competition_test_bodies.csv", "test_data.csv")
    merge_data(stances_filename, bodies_filename, merged_filename)
