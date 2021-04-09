import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def questions_and_players(train_tournaments, results_data):
    questions_difficulty = []
    players = []

    for idx in train_tournaments.index:
        total_questions = train_tournaments.loc[idx].total_questions
        correct = np.zeros(total_questions, dtype=int)
        wrong = np.zeros(total_questions, dtype=int)
        canceled = np.zeros(total_questions, dtype=int)
        team_qty = len(results_data[idx])
        for team in results_data[idx]:
            if 'mask' not in team:
                continue
            mask = team['mask']
            if mask is None:
                continue
            canceled = np.array([1 if a=='X' else 0 for a in mask])
            mask = np.array([1 if a=='1' else 0 for a in mask])
            if len(mask) > total_questions:
                mask = mask[:total_questions]
            correct[:len(mask)] += mask
            wrong[:len(mask)] += (1 - mask)
            for player in team['teamMembers']:
                player_id = player['player']['id']
                for i, (pos, neg) in enumerate(zip(mask, 1 - mask)):
                    if pos == 1:
                        players.append({'player_id': player_id, 'question': f'{idx}_{i}',
                                        'answered': 1, 'not_answered': 0})
                    if neg == 1:
                        players.append({'player_id': player_id, 'question': f'{idx}_{i}',
                                        'answered': 0, 'not_answered': 1})


        for i, (pos, neg, cnl) in enumerate(zip(correct, wrong, canceled)):
            if np.all(pos == 0) and np.all(neg == 0):
                continue
            questions_difficulty.append({'id': f'{idx}_{i}',
                                         'tournament': idx, 'question': i,
                                         'canceled': cnl,
                                         'positive': pos, 'negative': neg,
                                         'difficulty': 1 - pos/(pos + neg)})
    
    return pd.DataFrame(questions_difficulty), pd.DataFrame(players)
        
    
def create_dataset(player_result, player_result_grouped):
    levels = 5
    dataset = player_result.drop(['id', 'tournament', 'question_x', 'not_answered',
                              'question_y', 'canceled', 'positive', 'negative', 'level'], axis=1)
    
    dataset.index = dataset.player_id
    dataset.drop('player_id', axis=1, inplace = True)
    dataset.dropna(inplace=True)

    dataset = dataset.merge(
        player_result_grouped.drop('total_questions', axis=1), how='left', left_index=True, right_index=True)
    
    min_qty = []
    max_qty = []
    for i in range(1, levels + 1):
        min_qty.append(dataset[('n', i)].min())
        max_qty.append(dataset[('n', i)].max())
        dataset[('n', i)] = (dataset[('n', i)] - min_qty[-1]) / (max_qty[-1] - min_qty[-1])
    
    return dataset.sample(5000000), min_qty, max_qty


def train_model_lr(dataset_sample):
    
    X = dataset_sample.drop('answered', axis=1).values
    y = dataset_sample['answered'].values
    
    model = LogisticRegression()
    model.fit(X, y)
    pred = model.predict(X)
    print(f'Accuracy: {np.sum(pred == y)/len(pred)}')
    return model