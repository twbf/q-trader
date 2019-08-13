import keras
from keras.models import load_model
import matplotlib.pyplot as plt
from agent.agent import Agent
from functions import *
import sys

if len(sys.argv) != 3:
	print ("Usage: python evaluate.py [stock] [model]")
	exit()

stock_name, model_name = sys.argv[1], sys.argv[2]
model = load_model("models/" + model_name)
window_size = model.layers[0].input.shape.as_list()[1]

agent = Agent(window_size, True, model_name)
data = np.ndarray.tolist(np.multiply(0.3,getStockDataVec(stock_name)))
print(data)
l = len(data) - 1
batch_size = 32

state = getState(data, 0, window_size + 1)
total_profit = 0
stocks_held_tmp = 0
total_profit_array = np.zeros((l))
stocks_held = np.zeros((l))
agent.inventory = []
for t in range(l):
	action = agent.act(state)

	# sit
	next_state = getState(data, t + 1, window_size + 1)
	reward = 0
	if action == 1: # buy
		agent.inventory.append(data[t])
		print("Buy: " + formatPrice(data[t]))
		stocks_held_tmp += 1

	elif action == 2 and len(agent.inventory) > 0: # sell
		stocks_held_tmp -= 1
		bought_price = agent.inventory.pop(0)
		reward = max(data[t] - bought_price, 0)
		total_profit += data[t] - bought_price
		print ("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

	done = True if t == l - 1 else False
	agent.memory.append((state, action, reward, next_state, done))
	state = next_state

	total_profit_array[t] = total_profit
	stocks_held[t] = stocks_held_tmp
	if done:
		print ("--------------------------------")
		print (stock_name + " Total Profit: " + formatPrice(total_profit))
		print ( "--------------------------------")

print(np.sum(stocks_held))
plt.plot(total_profit_array)
plt.plot(data)
plt.show()
plt.plot(stocks_held)
plt.show()
