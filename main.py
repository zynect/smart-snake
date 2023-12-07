# type: ignore

# Welcome to
# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \\   __\   __\  | _/ __ \ /  ___//    \\__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\\_____>
#
# This file can be a nice home for your Battlesnake logic and helper functions.
#
# To get you started we've included code to prevent your Battlesnake from moving backwards.
# For more info see docs.battlesnake.com

import os.path
import random
import typing
import vowpalwabbit

vw = None  # The Vowpal Wabbit model
previous_action = None  # The action the model took last turn
previous_game_state = None  # The dictionary that contains the game state from last turn


# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
# TIP: If you open your Battlesnake URL in a browser you should see this data
def info() -> typing.Dict:
  print("INFO")

  return {
      "apiversion": "1",
      "author": "zynect",
      "color": "#777777",
      "head": "shades",
      "tail": "comet",
  }


# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
  global vw, previous_action, previous_game_state

  if vw == None:
    workspace = "--cb_explore_adf --epsilon 0.1 -l 0.5"

    if os.path.isfile("snake.model"):
      workspace += " -i snake.model"
      print("Using existing model")
    else:
      print("Making new model")

    vw = vowpalwabbit.Workspace(workspace, quiet=False)

  previous_action = None
  previous_game_state = None
  print("GAME START")


# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
  global vw

  learn(["up", "down", "left", "right"], 100)
  vw.save("snake.model")

  print("GAME OVER\n")


# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
def move(game_state: typing.Dict) -> typing.Dict:
  global vw, previous_action, previous_game_state

  actions = ["up", "down", "left", "right"]
  # VowpalWabbit uses postive values to indicate loss, so negative values are the reward
  learn(actions, -1)

  action = get_action(game_state, actions)
  next_move, prob = action

  previous_action = action
  previous_game_state = game_state

  print(f"MOVE {game_state['turn']}: {next_move} {prob}")
  return {"move": next_move}


def learn(actions, cost):
  global vw, previous_action, previous_game_state

  if previous_action == None or previous_game_state == None:
    print("Previous states don't exist")
    return

  print(f"+++++++++Learning with reward {-cost}++++++++++++")
  previous_move, prob = previous_action

  vw_format = to_vw_example_format(previous_game_state, actions,
                                   (previous_move, cost, prob))
  print(vw_format)
  #vw_format = vw.parse(vw_format, vowpalwabbit.LabelType.CONTEXTUAL_BANDIT)
  vw.learn(vw_format)


def to_vw_example_format(game_state, actions, cb_label=None):
  board = game_state["board"]
  example_string = ""

  #example_string = get_shared_features(game_state)

  if cb_label is not None:
    chosen_action, cost, prob = cb_label

  grid = get_grid(game_state)
  x = game_state["you"]["head"]["x"]
  y = game_state["you"]["head"]["y"]
  
  for action in actions:
    if cb_label is not None and action == chosen_action:
      example_string += f"{action}:{cost}:{prob} "
    example_string += "|Action article={} type={}\n".format(
        action, get_val_from_action(action, x, y, grid))
  # Strip the last newline
  return example_string[:-1]


def get_val_from_action(direction, x, y, grid):
  if direction == "up":
    return get_val_from_grid(x, y + 1, grid)
  if direction == "down":
    return get_val_from_grid(x, y - 1, grid)
  if direction == "right":
    return get_val_from_grid(x + 1, y, grid)
  if direction == "left":
    return get_val_from_grid(x - 1, y, grid)

  raise Exception("INVALID DIRECTION") 


# Returns a string that includes all shared features that we want to give the model
def get_shared_features(game_state):
  # Add board size features
  example_string = "shared |"
  example_string += " height={} width={} ".format(board["height"],
                                                  board["width"])
  # Add snake location features
  for snake in board["snakes"]:
    length = len(snake['body'])
    owner = "Your" if snake['id'] == game_state['you']['id'] else "Enemy"

    for i in range(length):
      x = snake['body'][i]['x']
      y = snake['body'][i]['y']
      body = "Head" if i == 0 else "Body"
      example_string += f"{owner}{body}{i}={x}x{y} "

  example_string += "\n"
  return example_string


def get_action(context, actions):
  vw_text_example = to_vw_example_format(context, actions)
  pmf = vw.predict(vw_text_example)
  chosen_action_index, prob = sample_custom_pmf(pmf)
  return actions[chosen_action_index], prob


def sample_custom_pmf(pmf):
  total = sum(pmf)
  scale = 1 / total
  pmf = [x * scale for x in pmf]
  draw = random.random()
  sum_prob = 0.0
  for index, prob in enumerate(pmf):
    sum_prob += prob
    if sum_prob > draw:
      return index, prob


def get_grid(data):
  # Generates a width x height array that holds strings indicating what is there
  width = data['board']['width']
  height = data['board']['height']
  grid = [["empty" for x in range(width)] for y in range(height)]

  for coord in data['board']['food']:
    grid[coord['x']][coord['y']] = "food"

  for snake in data['board']['snakes']:
    length = len(snake['body'])
    for i in range(length):
      grid[snake['body'][i]['x']][snake['body'][i]['y']] = "snake"

  return grid


def get_val_from_grid(x, y, grid):
  # Returns the value from the grid, otherwise if outside the grid return "edge"
  if x < len(grid) and x >= 0 and y < len(grid[0]) and y >= 0:
    return grid[x][y]

  return "edge"


# Start server when `python main.py` is run
if __name__ == "__main__":
  from server import run_server

  run_server({"info": info, "start": start, "move": move, "end": end})
