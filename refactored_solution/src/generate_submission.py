from utils import load_body, load_stance, load_title, load_dataset
from main import train_and_predict_3_steps
import csv

""" Set data paths """
TRAIN_BODY_CSV = '../train_bodies.csv'
TRAIN_STANCE_CSV = '../train_stances.csv'

TEST_BODY_CSV = '../data/test_bodies.csv'
TEST_HEADLINE_CSV = '../data/test_stances_unlabeled.csv'

""" Load data """
id2body, id2body_sentences = load_body(TRAIN_BODY_CSV)
test_id2body, test_id2body_sentences = load_body(TEST_BODY_CSV)

id2body.update(test_id2body)
id2body_sentences.update(test_id2body_sentences)

train_data = load_stance(TRAIN_STANCE_CSV)[1:]

seen_head = set()
seen_body_id = set()
for (head, body_id, stance) in train_data:
    seen_head.add(' '.join(head))
    seen_body_id.add(body_id)

test_data = load_title(TEST_HEADLINE_CSV)

print("*****DONE LOADING DATA*****")

""" Train and predict """
test_pred, test_scores = train_and_predict_3_steps(train_data, test_data, id2body, id2body_sentences)

print("*****DONE TRAINING AND PREDICTING*****")

""" Save results """
with open('results/submission_scores.csv', 'w') as out:
    for scores in test_scores:
        out.write(','.join([str(score) for score in scores]) + '\n')

test_dataset = load_dataset(TEST_HEADLINE_CSV)

with open('results/submission.csv', 'w') as csvfile:
    fieldnames = ['Headline', 'Body ID', 'Stance']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for i, t in enumerate(test_dataset):
        t['Stance'] = test_pred[i]
        writer.writerow(t)

print("*****DONE SAVING RESULTS*****")
