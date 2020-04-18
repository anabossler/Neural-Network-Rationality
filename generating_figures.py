import os, csv, math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.ticker import FuncFormatter

os.chdir(r"C:\Users\thehu\OneDrive\Documents\2019-2020\Thesis (Economics)")

def listparse(csvfilename):
    """
    Takes a csvfile as input and parses it into a list of lists.
    """
    output = []
    with open(csvfilename, 'r', newline = '') as csvfile:
        csvreader = csv.reader(csvfile, skipinitialspace = True)
        for row in csvreader:
            output.append(row)
    return output

def list_dictparse(csvfilename):
    """
    Parses CSV file into a list of dictionaries.
    """
    output = []

    with open(csvfilename, 'r', newline = '') as csvfile:
        dictreader = csv.DictReader(csvfile, skipinitialspace = True)
        for row in dictreader:
            output.append(row)

    return output

"""
ALL OF THE FOLLOWING CODE IS THE BASIS FOR CREATING FIGURE 1
"""
ug_hyperparameter_testing = listparse('Hyperparameter Testing.csv')

os.chdir(r"C:\Users\thehu\OneDrive\Documents\2019-2020\Thesis (Economics)\Figures")

data = []
for row in ug_hyperparameter_testing:
    new_row = []
    for item in row:
        try:
            new_row.append(float(item))
        except:
            new_row.append(item)
    data.append(new_row)
x_values = data[0][1:]
data = data[1:]

trendline_colors = ["r", "b", "g"]
colnames = ['200', '250', '300']
rownames = ['0.6', '0.75', '0.9']
fig, axs = plt.subplots(3, 3, sharex = True, sharey = True, figsize = (14, 10))
padding = 5
fig.subplots_adjust(wspace = 0, hspace = 0)
for ax, col in zip(axs[0], colnames):
    ax.annotate(
        col, xy = (0.5, 1), xytext = (0, padding),
        xycoords = 'axes fraction', textcoords = 'offset points',
        size = 'large', ha = 'center', va = 'baseline'
    )
for ax, row in zip(axs[:,0], rownames):
    ax.annotate(
        row, xy = (0, 0.5), xytext = (-padding, 0),
        xycoords = ax.yaxis.label, textcoords = 'offset points',
        size = 'large', ha = 'right', va = 'center'
    )
axs[2][1].annotate(
    'Epoch Number', xy = (0.5, 0), xytext = (0, -axs[2][1].xaxis.labelpad - padding),
    xycoords = axs[2][1].xaxis.label, textcoords = 'offset points',
    size = 'x-large', ha = 'center', va = 'baseline'
)
axs[1][2].annotate(
    'Network Efficiency (% of total possible winnings captured)', xy = (1, 0.5), xytext = (25, 0),
    xycoords = 'axes fraction', textcoords = 'offset points',
    size = 'x-large', ha = 'right', va = 'center', rotation = 'vertical'
)
axs[1][0].annotate(
    'Geometric Parameter', xy = (0, 0.5), xytext = (-35, 0),
    xycoords = axs[1][0].yaxis.label, textcoords = 'offset points',
    size = 'x-large', ha = 'right', va = 'center', rotation = 'vertical'
)
axs[0][1].annotate(
    'Hidden Size', xy = (0.5, 1), xytext = (0, 25),
    xycoords = 'axes fraction', textcoords = 'offset points',
    size = 'x-large', ha = 'center', va = 'baseline'
)
fig.subplots_adjust(left = 0.2, top = 0.95, bottom = 0.05, right = 0.8)
red_patch = mpatches.Patch(color = "r", label ="0.1")
blue_patch = mpatches.Patch(color = "b", label = "0.5")
green_patch = mpatches.Patch(color = "g", label = "1.0")
plt.legend(handles = [red_patch, blue_patch, green_patch], loc = 'lower right', title = 'Initial Learning Rate:')

for index in range(0, 27, 9):
    column_data = data[index : index + 9]
    column_index = math.floor(index / 9)
    for subindex in range(0, 9, 3):
        plot_data = column_data[subindex : subindex + 3]
        row_index = math.floor(subindex / 3)
        for j in range(3):
            z = np.polyfit(x_values, plot_data[j][1:], 5)
            f = np.poly1d(z)
            axs[row_index, column_index].plot(x_values, f(x_values), color = trendline_colors[j])

plt.savefig('Hyperparameter Testing.png')


"""
THIS IS THE END OF CODE FOR FIGURE 1 (EXCEPT ACTUAL PLOTTING OF DATA)
"""

"""
Figure for showing effectiveness of neural network relative to SPE strategy
and random guessing.
"""
def percent_tick_format(x, pos):
    """
    Formats y labels to be percentages.
    """
    return '{}%'.format(int(x * 100))

y_formatter = FuncFormatter(percent_tick_format)

effectiveness = [0.0217, 0.0885, 0.3528, 0.6167]
strategies = ('SPE', 'Smallest Nonzero\nOffer', 'Random Offer', 'Trained Network')
x = np.arange(4)

fig, axs = plt.subplots(1, 1, figsize = (10, 10))
axs.yaxis.set_major_formatter(y_formatter)
axs.bar(x, effectiveness)
plt.xticks(x, strategies)
plt.ylabel('Efficiency (% of possible earnings captured)')
plt.savefig('Relative Efficiency of Different Strategies.png')

"""
End of code for comparative efficiency figure.
"""


"""
Figure for showing how "active" the neural network is as it trains.
"""
os.chdir(r"C:\Users\thehu\OneDrive\Documents\2019-2020\Thesis (Economics)")

network_activity = listparse("Overall Training Distribution of Offers.csv")

colnames1 = ['1 - 20', '21 - 40', '41 - 60', '61 - 80', '81 - 100']
colnames2 = ['101 - 120', '121 - 140', '141 - 160', '161 - 180', '181 - 200']
colnames3 = ['201 - 220', '221 - 240', '241 - 260', '261 - 280', '281 - 300']
colnames4 = ['301 - 320', '321 - 340', '341 - 360', '361 - 380', '381 - 400']
colnames = [colnames1, colnames2, colnames3, colnames4]

fig, axs = plt.subplots(4, 5, sharex = True, sharey = True, figsize = (12.5, 10))

for i in range(4):
    for ax, col in zip(axs[i], colnames[i]):
        ax.annotate(
            col, xy = (0.5, 1), xytext = (0, padding),
            xycoords = 'axes fraction', textcoords = 'offset points',
            size = 'large', ha = 'center', va = 'baseline'
        )
for i in range(4):
    for j in range(5):
        axs[i, j].hist(network_activity[(5 * i) + j][1:])

axs[3][2].annotate(
    'Offer amount (% of initial stake)', xy = (0.5, -20), xytext = (0, -axs[3][2].xaxis.labelpad - padding - 10),
    xycoords = axs[3][2].xaxis.label, textcoords = 'offset points',
    size = 'x-large', ha = 'center', va = 'baseline'
)
axs[2][0].annotate(
    'Frequency', xy = (0, 0), xytext = (-20, 30),
    xycoords = axs[2][0].yaxis.label, textcoords = 'offset points',
    size = 'x-large', ha = 'center', va = 'bottom', rotation = 'vertical'
)

plt.setp(axs, xticks = [1, 5, 9], xticklabels = [0.1, 0.5, 0.9])
plt.savefig('Network Activity.png')

"""
End of code for figure showing how active the network is.
"""


"""
Set up figure to compare human play to neural network play.
"""
os.chdir(r"C:\Users\thehu\OneDrive\Documents\2019-2020\Thesis (Economics)")

human_data = list_dictparse('Cooper and Dutcher.csv')

human_proposals_by_round = {}
for entry in human_data:
    round_number = int(entry['round'])
    proposal = float(entry['amount_offered (%)']) / 100
    try:
        human_proposals_by_round[round_number].append(proposal)
    except:
        human_proposals_by_round[round_number] = [proposal]

BINS = list(range(21))
BINS = [round(x * 0.05, 2) for x in BINS]

fig, axs = plt.subplots(2, 5, sharex = True, sharey = True, figsize = (30, 12))
colnames = ['1', '2', '3', '4', '5']
padding = 5
for ax, col in zip(axs[0], colnames):
    ax.annotate(
        col, xy = (0.5, 1), xytext = (0, padding),
        xycoords = 'axes fraction', textcoords = 'offset points',
        size = 'x-large', ha = 'center', va = 'baseline'
    )
axs[0][2].annotate(
    'Round Number', xy = (0.5, 1), xytext = (0, 25),
    xycoords = 'axes fraction', textcoords = 'offset points',
    size = 'xx-large', ha = 'center', va = 'baseline'
)
axs[1][2].annotate(
    'Amount Offered (% of total)', xy = (0.5, 0), xytext = (0, -axs[1][2].xaxis.labelpad - (3 * padding)),
    xycoords = axs[1][2].xaxis.label, textcoords = 'offset points',
    size = 'xx-large', ha = 'center', va = 'baseline'
)
rownames = ['Human Proposers', 'Neural Netowrk']
for ax, row in zip(axs[:,0], rownames):
    ax.annotate(
        row, xy = (0, 0.5), xytext = (-padding, 0),
        xycoords = ax.yaxis.label, textcoords = 'offset points',
        size = 'x-large', ha = 'right', va = 'center', rotation = 'vertical'
    )

neural_net_proposals = listparse('neural_net_proposals.csv')
reformatted_neural_net_proposals = []
for i in range(len(neural_net_proposals)):
    reformatted_round = [float(entry) for entry in neural_net_proposals[i]]
    reformatted_neural_net_proposals.append(reformatted_round)

for round_number in range(5):
    axs[1, round_number].hist(reformatted_neural_net_proposals[round_number], density = True, ec = "#ff9900", fc = "#ffebcc", bins = BINS)
for round_number in human_proposals_by_round:
    if round_number > 5:
        break
    axs[0, round_number - 1].hist(human_proposals_by_round[round_number], density = True, ec = "b", fc = "#ccddff", bins = BINS)

os.chdir(r"C:\Users\thehu\OneDrive\Documents\2019-2020\Thesis (Economics)\Figures")
plt.savefig('Net vs. Humans (rounds 1-5) - 4-2-20.png')

# Now for rounds 6-10:
os.chdir(r"C:\Users\thehu\OneDrive\Documents\2019-2020\Thesis (Economics)")

fig, axs = plt.subplots(2, 5, sharex = True, sharey = True, figsize = (30, 12))
colnames = ['6', '7', '8', '9', '10']
padding = 5
for ax, col in zip(axs[0], colnames):
    ax.annotate(
        col, xy = (0.5, 1), xytext = (0, padding),
        xycoords = 'axes fraction', textcoords = 'offset points',
        size = 'x-large', ha = 'center', va = 'baseline'
    )
axs[0][2].annotate(
    'Round Number', xy = (0.5, 1), xytext = (0, 25),
    xycoords = 'axes fraction', textcoords = 'offset points',
    size = 'xx-large', ha = 'center', va = 'baseline'
)
axs[1][2].annotate(
    'Amount Offered (% of total)', xy = (0.5, 0), xytext = (0, -axs[1][2].xaxis.labelpad - (3 * padding)),
    xycoords = axs[1][2].xaxis.label, textcoords = 'offset points',
    size = 'xx-large', ha = 'center', va = 'baseline'
)
rownames = ['Human Proposers', 'Neural Netowrk']
for ax, row in zip(axs[:,0], rownames):
    ax.annotate(
        row, xy = (0, 0.5), xytext = (-padding, 0),
        xycoords = ax.yaxis.label, textcoords = 'offset points',
        size = 'x-large', ha = 'right', va = 'center', rotation = 'vertical'
    )

neural_net_proposals = listparse('neural_net_proposals.csv')
reformatted_neural_net_proposals = []
for i in range(len(neural_net_proposals)):
    reformatted_round = [float(entry) for entry in neural_net_proposals[i]]
    reformatted_neural_net_proposals.append(reformatted_round)

for round_number in range(5, 10):
    axs[1, round_number - 5].hist(reformatted_neural_net_proposals[round_number], density = True, ec = "#ff9900", fc = "#ffebcc", bins = BINS)
    axs[0, round_number - 5].hist(human_proposals_by_round[round_number + 1], density = True, ec = "b", fc = "#ccddff", bins = BINS)

os.chdir(r"C:\Users\thehu\OneDrive\Documents\2019-2020\Thesis (Economics)\Figures")
plt.savefig('Net vs. Humans (rounds 6-10) - 4-2-20.png')
