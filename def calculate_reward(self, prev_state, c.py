def calculate_reward(self, prev_state, current_state):
    """Calculate reward based on visual game state"""
    # Extract health values using computer vision
    prev_health = self.extract_health_bars(prev_state)
    current_health = self.extract_health_bars(current_state)
    
    # Adapt your existing reward logic
    health_diff = (prev_health['enemy'] - current_health['enemy']) * 2
    damage_taken = (current_health['player'] - prev_health['player'])
    
    return health_diff + damage_taken