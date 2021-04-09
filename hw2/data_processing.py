import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def has_teamMembers_and_mask(tournament):
    flg = True
    for team in tournament:
        if (len(team['teamMembers']) == 0) or (team.get('mask') is None):
            flg = False
    return flg


def dict_to_df(md: dict):
    df = pd.DataFrame(md.values()).set_index('id', drop=True)
    df['dateStart'] = pd.to_datetime(df['dateStart'], utc=True)
    df['dateEnd'] = pd.to_datetime(df['dateEnd'], utc=True)

    tournament_types = {}
    for t in df['type']:
        tournament_types[t['id']] =  t['name']

    df['type'] = df['type'].apply(lambda x: x['id'])

    df.drop('orgcommittee', axis=1, inplace=True)
    df['year'] = df.dateStart.dt.year
    df['month'] = df.dateStart.dt.month
    return df


def type_tourn(results, tournaments):
    type_tourn = {}
    for id, value in results.items():
        type_tourn[tournaments[id]['type']['name']] = 0

    type_tourn_2019 = type_tourn.copy()
    type_tourn_2020 = type_tourn.copy()

    for id, value in results.items():
        y = int(tournaments[id]["dateStart"][0:4])
        if y == 2019:
            type_tourn_2019[tournaments[id]['type']['name']] += 1
        else:
            type_tourn_2020[tournaments[id]['type']['name']] += 1



    fig, (plt1, plt2) = plt.subplots(nrows=2, ncols=1, figsize=(18,10))

    x_2019 = type_tourn_2019.keys()
    y_2019 = type_tourn_2019.values()

    x_2020 = type_tourn_2020.keys()
    y_2020 = type_tourn_2020.values()

    plt1.bar(x_2019, y_2019, width = 0.8, color='green')
    plt2.bar(x_2020, y_2020, width = 0.8, color='black')
    
   
    