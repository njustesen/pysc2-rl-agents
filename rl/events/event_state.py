import numpy as np
from pysc2.lib import features

class EventState:

    def __init__(self, timestep):

        self.minerals = timestep.observation["structured"]["minerals"]
        self.vespene = timestep.observation["structured"]["vespene"]
        self.food_used = timestep.observation["structured"]["food used"]
        self.food_cap = timestep.observation["structured"]["food cap"]
        self.own_unit_types = np.zeros(500)
        player_y, player_x = (timestep.observation["minimap"][features.SCREEN_FEATURES.player_relative.index] == 1).nonzero()
        unit_type = timestep.observation["screen"][features.SCREEN_FEATURES.unit_type.index]
        for i in range(player_y):
            x = player_x[i]
            y = player_y[i]
            type = unit_type[y][x]
            self.own_unit_types[type] = 0
        self.ready = True
        self.events = np.zeros(len(self.own_unit_types) + 4)

    def reset(self):
        self.ready = False

    def update(self, timestep):

        minerals = timestep.observation["structured"]["minerals"]
        vespene = timestep.observation["structured"]["vespene"]
        food_used = timestep.observation["structured"]["food used"]
        food_cap = timestep.observation["structured"]["food cap"]
        own_unit_types = np.zeros(500)
        player_y, player_x = (
        timestep.observation["minimap"][features.SCREEN_FEATURES.player_relative.index] == 1).nonzero()
        unit_type = timestep.observation["screen"][features.SCREEN_FEATURES.unit_type.index]
        for i in range(player_y):
            x = player_x[i]
            y = player_y[i]
            type = unit_type[y][x]
            if type < len(own_unit_types):
                own_unit_types[type] = 0
            else:
                print(str(type) + " type id to high")

        # Change vector and update
        change = np.zeros(504)

        for i in range(player_y):
            x = player_x[i]
            y = player_y[i]
            type = unit_type[y][x]
            if self.own_unit_types[type] == 0:
                change[type] = 1
            self.own_unit_types[type] = 1

        if minerals > self.minerals:
            change[500] = 1
        if minerals > self.vespene:
            change[501] = 1
        if minerals > self.food_used:
            change[502] = 1
        if minerals > self.food_cap:
            change[503] = 1

        self.minerals = minerals
        self.vespene = vespene
        self.food_used = food_used
        self.food_cap = food_cap

        if not self.ready:
            self.ready = True
            return np.zeros(500)

        self.events = np.sum(self.events, change)
        self.events = np.clip(self.events, 0, 1)

        return change
