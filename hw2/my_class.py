import numpy as np
import pandas as pd


class Tournament():
    def __init__(self, idx, results,
                 tournaments,
                 model, p_questions, 
                 players_features):
        self.idx = idx
        self.results = results[idx]
        self.model = model
        self.players_features = players_features
        
        self.total_questions = tournaments.loc[idx, 'total_questions']
        self.teams = self._load_teams()
        self.questions = np.array(p_questions.sample(self.total_questions)).reshape(-1, 1)
        
    def _load_teams(self):
        teams = []
        for team in self.results:
            if 'position' not in team or team['mask'] is None:
                continue
            current_team = {}
            current_team['name'] = team['team']['name']
            current_team['position'] = team['position']
            current_team['members'] = self._load_team_members(team)
            teams.append(current_team)
        return teams
        
    def _load_team_members(self, team):
        players = {}
        for player in team['teamMembers']:
            player_id = player['player']['id']
            if player_id not in self.players_features.index:
                continue
            player_features = np.array(self.players_features.loc[player_id]).reshape(1, -1)
            players[player_id] = player_features
        return players
        
    def simulate(self):
        if len(self.teams) == 0:
            return None, None
        predicted_results = []
        factual_results = []
        for team in self.teams:
            prediction = self._simulate_team_result(team)
            if prediction == 0:
                continue
            factual_results.append(team['position'])
            predicted_results.append(prediction)
        return predicted_results, factual_results
    
    def _simulate_team_result(self, team):
        team_size = len(team['members'])
        if team_size == 0:
            return 0
        team_features = np.vstack([v for k, v in team['members'].items()])
        result = 0
        for q in self.questions:
            full = np.hstack([np.tile(q, (team_size, 1)), team_features])
            probs = self.model.predict_proba(full)[:,1]
            result += np.mean(probs)
        return -result