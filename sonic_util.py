import numpy as np

def get_sonic_specific_actions():
    buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
    actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                ['DOWN', 'B'], ['B'], [], ['LEFT', 'B'], ['RIGHT', 'B']]
    _actions = []
    for action in actions:
        arr = np.array([False] * 12)
        for button in action:
            arr[buttons.index(button)] = True
        _actions.append(arr)
    
    return _actions