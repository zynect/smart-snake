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

# Hyperparameters
# Vowpal Wabbit has the reward as negative and loss as positive
basic_reward = 0
food_reward = -500
game_over_loss = 500
epsilon = 0.001
learning_rate = 0.1


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
def start(game_state: typing.Dict) -> None:
  global vw, previous_action, previous_game_state

  if vw == None:
    workspace = "--cb_explore_adf --quiet --epsilon {} -l {}".format(
        epsilon, learning_rate)

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
def end(game_state: typing.Dict) -> None:
  global vw, previous_game_state

  cost = get_cost(game_state, previous_game_state, True)
  print(cost)
  learn(["up", "down", "left", "right"], cost)
  vw.save("snake.model")

  print("GAME OVER\n")


# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
def move(game_state: typing.Dict) -> typing.Dict:
  global vw, previous_action, previous_game_state

  actions = ["up", "down", "left", "right"]

  # Also give the reward for the last move, since we now know the snake hasn't died
  cost = get_cost(game_state, previous_game_state)
  learn(actions, cost)

  action = get_action(game_state, actions)
  next_move, prob = action

  previous_action = action
  previous_game_state = game_state

  print(f"MOVE {game_state['turn']}: {next_move} {prob}")
  return {"move": next_move}


def get_cost(game_state: typing.Dict,
             previous_game_state: typing.Dict,
             game_over: bool = False) -> int:
  cost = basic_reward
  
  if game_over:
    # In a competitive game, the game ends if the other snakes have died
    if snake_alive(game_state):
      return basic_reward
    return game_over_loss

  # If we gained health since last turn that means we ate a food, and we should use food reward instead
  if previous_game_state is not None and game_state["you"][
      "health"] > previous_game_state["you"]["health"]:
    cost = food_reward

  return cost


def snake_alive(game_state: typing.Dict) -> bool:
  # Snake is alive if its id is in the board state's snake list
  snake_id = game_state["you"]["id"]
  for snake in game_state["board"]["snakes"]:
    if snake_id == snake["id"]:
      return True

  return False

def learn(actions: list[int], cost: int) -> None:
  global vw, previous_action, previous_game_state

  if previous_action == None or previous_game_state == None:
    print("Previous states don't exist")
    return

  #print(f"+++++++++Learning with reward {-cost}++++++++++++")
  previous_move, prob = previous_action

  vw_format = to_vw_example_format(previous_game_state, actions,
                                   (previous_move, cost, prob))
  #print(vw_format)
  vw.learn(vw_format)


def to_vw_example_format(game_state: typing.Dict,
                         actions: list[int],
                         cb_label: tuple[str, int, float] = None) -> str:
  board = game_state["board"]
  example_string = "shared |"
  example_string += " health={}\n".format(game_state["you"]["health"])

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


def get_val_from_action(direction: str, x: int, y: int,
                        grid: list[list[int]]) -> str:
  if direction == "up":
    return get_val_from_grid(x, y + 1, grid)
  if direction == "down":
    return get_val_from_grid(x, y - 1, grid)
  if direction == "right":
    return get_val_from_grid(x + 1, y, grid)
  if direction == "left":
    return get_val_from_grid(x - 1, y, grid)

  raise Exception("INVALID DIRECTION")


def get_action(context: typing.Dict, actions: list[int]) -> tuple[str, float]:
  vw_text_example = to_vw_example_format(context, actions)
  pmf = vw.predict(vw_text_example)
  chosen_action_index, prob = sample_custom_pmf(pmf)
  return actions[chosen_action_index], prob


def sample_custom_pmf(pmf: list[float]) -> tuple[int, float]:
  total = sum(pmf)
  scale = 1 / total
  pmf = [x * scale for x in pmf]
  draw = random.random()
  sum_prob = 0.0
  for index, prob in enumerate(pmf):
    sum_prob += prob
    if sum_prob > draw:
      return index, prob


def get_grid(data: typing.Dict) -> list[list[int]]:
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


def get_val_from_grid(x: int, y: int, grid: list[list[int]]) -> str:
  # Returns the value from the grid, otherwise if outside the grid return "edge"
  if x < len(grid) and x >= 0 and y < len(grid[0]) and y >= 0:
    return grid[x][y]

  return "edge"


# Start server when `python main.py` is run
if __name__ == "__main__":
  from server import run_server

  run_server({"info": info, "start": start, "move": move, "end": end})
