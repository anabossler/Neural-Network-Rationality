from scipy.stats import gaussian_kde
import numpy as np
from scipy.stats import rv_continuous, kstest, gaussian_kde, norm
import os, csv, math, time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import shuffle
import matplotlib.patches as mpatches

os.chdir(r"C:\Users\thehu\OneDrive\Documents\2019-2020\Thesis (Economics)\Beauty Contest Game data")

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

"""
===========================================================
===================== DATA FORMATTING =====================
===========================================================
"""
# Some data formatting has been done in Stata prior to that which is done here

# Parse data into list of lists
updated_bcg_data = listparse('New Updated BCG Data.csv')
newspaper_data = listparse('Newspaper Data.csv')

# Format list of predicted coefficients from regression run in Stata
predicted_coefficients = []
for player_data in updated_bcg_data[1:]:
    predicted_c = player_data[-2]
    predicted_se = player_data[-1]
    try:
        predicted_c = float(predicted_c)
        predicted_se = float(predicted_se)
    except:
        predicted_c = 0
    if predicted_c and not (predicted_c, predicted_se) in predicted_coefficients:
        predicted_coefficients.append((predicted_c, predicted_se))

# Make lists of first round responses for each p-value
first_round_responses_0_7 = []
first_round_responses_0_9 = []
print(updated_bcg_data[:3])
for i in range(1, 1961, 10):
    p_value = updated_bcg_data[i][0]
    first_round_response = float(updated_bcg_data[i][-4])
    if float(p_value) == 0.7:
        first_round_responses_0_7.append(first_round_response)
    elif float(p_value) == 0.9:
        first_round_responses_0_9.append(first_round_response)

first_round_responses_2_3 = []
for i in range(1, len(newspaper_data)):
    first_round_response = float(newspaper_data[i][-1])
    first_round_responses_2_3.append(first_round_response)

# Create kernel density estimations for distribution of first round responses
class first_round_response_pdf(rv_continuous):
    """
    Class for creating a pdf, round-by-round, from which samples may be drawn.
    """
    def __init__(self, data):
        super(first_round_response_pdf, self).__init__(a = 0, b = 100)
        self.kde = gaussian_kde(data, bw_method = 0.18)

    def _pdf(self, x):
        return self.kde.evaluate(x)

kde_0_7 = first_round_response_pdf(first_round_responses_0_7)
kde_0_9 = first_round_response_pdf(first_round_responses_0_9)
kde_2_3 = first_round_response_pdf(first_round_responses_2_3)

# Store kdes in dictionary with corresponding p_value
first_round_pdfs = {
    0.7: kde_0_7,
    0.9: kde_0_9,
    2/3: kde_2_3,
}

# Assemble groups of actual human data to train network
bcg_data = listparse('Beauty Contest Game Data.csv')

class real_human_player():
    """
    Class for modeling actual human data in bcg.
    """
    def __init__(self, data_entry):

        self.responses = []
        for i in range(10):
            self.responses.append(float(data_entry[i + 1]))

        self.p_value = float(data_entry[0])
        self.group = int(data_entry[-1])

# Create real human players and store them in list of lists where each sublist
# represents a different group of real human players
real_human_players = []
for entry in bcg_data[1:]:
    group_number = int(entry[-1])
    try:
        real_human_players[group_number].append(real_human_player(entry))
    except:
        real_human_players.append([real_human_player(entry)])

# Now create all possible subgroups of 6 real human players from each of the
# groups of 7 and create a dataset of these groups and the original groups of 7
real_human_groups_7 = []
real_human_groups_9 = []
for group in real_human_players:
    p_value = group[0].p_value
    for i in range(7):
        new_group = group[:i] + group[i + 1:]
        if p_value == 0.7:
            real_human_groups_7.append(new_group)
        else:
            real_human_groups_9.append(new_group)
    if p_value == 0.7:
        real_human_groups_7.append(group)
    else:
        real_human_groups_9.append(group)

"""
==========================================================
================== DATA GENERATION =======================
==========================================================
"""

# Create class for BCG fictitious player
class fictitious_player_bcg():
    """
    Class for creating fictitious players in the beauty pageant game.
    """
    def __init__(self, p_value):

        self.first_round_response = first_round_pdfs[p_value].rvs()
        self.previous_guess = None

        # Below, the coefficients in the estimated regression are set for each individual
        # fictitious player, sampled from a normal distribution with mean and standard
        # error taken directly from actual regression
        self.alpha = np.random.normal(-1.07, 0.032)
        self.beta = np.random.normal(0.913, 0.048)
        self.gamma = np.random.normal(-0.516, 0.177)

        gamma_idx = np.random.randint(0, 194)
        self.delta = np.random.normal(predicted_coefficients[gamma_idx][0], predicted_coefficients[gamma_idx][1])

        self.epsilon = np.random.normal(22.907, 8.571)


    def respond(self, previous_winning_number, round_number):
        if round_number == 0:
            self.previous_guess = self.first_round_response
            return self.first_round_response
        guess = self.previous_guess + ((self.alpha * self.previous_guess) + (self.beta * previous_winning_number) + (self.gamma * self.first_round_response) + self.delta + self.epsilon)
        if guess > 100:
            guess = 100
        if guess < 0:
            guess = 0
        self.previous_guess = guess
        return guess

# Create one million fictitious players in the beauty pageant game and store them to a container list
bcg_players_0_7 = []
number_of_players = 1000
start = time.time()
for i in range(number_of_players):
    bcg_player = fictitious_player_bcg(0.7)
    bcg_players_0_7.append(bcg_player)
    if (i + 1) % 50 == 0:
        elapsed = time.time() - start
        estimated_remaining = round((((elapsed / (i + 1)) * (number_of_players) - elapsed) / 60), 2)
        print('='*len('{} Players Created'.format(i + 1)))
        print('{} Players Created'.format(i + 1))
        print('Time Elapsed: {} minutes'.format(round(elapsed / 60, 2)))
        print('Estimated Time Remaining: {} minutes'.format(estimated_remaining))
        print('='*len('{} PLayers Created'.format(i + 1)))



"""
========================================================
============== SETTING UP NEURAL NETWORK ===============
========================================================
"""

# Define a generic class for an lstm net with parameters to make it customizable
class generic_net(nn.Module):
    """
    A neural network that will be trained to play as a proposer in the Ultimatum Game.
    """
    def __init__(self, input_size, hidden_size, output_size, bias = True, bidirectional = True, num_layers = 2):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, bias = bias, bidirectional = bidirectional, num_layers = num_layers)

        if bidirectional:
            linear_input = 2 * hidden_size
        else:
            linear_input = hidden_size

        self.linear1 = nn.Linear(linear_input, hidden_size, bias = bias)
        self.linear2 = nn.Linear(hidden_size, output_size, bias = bias)

    def forward(self, input_sequence, h_init, c_init):

        output_sequence, (h_last, c_last) = self.lstm(input_sequence, (h_init, c_init))

        h_direc_1 = h_last[2,:,:]
        h_direc_2 = h_last[3,:,:]
        h_direc_12 = torch.cat([h_direc_1, h_direc_2], dim = 1)

        x = self.linear1(h_direc_12)
        x = F.relu(x)
        scores = self.linear2(x)

        return scores, (h_last, c_last)


"""
============================================================
========== DEFINING FUNCTIONS NEEDED FOR TRAINING ==========
============================================================
"""

# Define a function for normalizing the gradient of the network during training
def normalize_gradient(net):
    """
    Function to normalize gradient to prevent exploding gradients.
    """
    grad_norm_sq = 0

    for p in net.parameters():
        grad_norm_sq += p.grad.data.norm() ** 2

    grad_norm = math.sqrt(grad_norm_sq)

    if grad_norm < .0001:
        net.zero_grad()
        print('Grad norm close to zero.')
        print('')

    else:
        for p in net.parameters():
             p.grad.data.div_(grad_norm)

# Define function for tracking error of network
def get_error(scores, labels):
    """
    Will take the output of the network (scores; a tensor) and what
    the network should have output (labels; also a tensor) and compute
    the error for a specific batch.
    """
    bs = scores.size(0) # 'bs' stands for 'batch size'
    predicted_labels = scores.argmax(dim = 1) # Tensor with 'bs' entries
    indicator = (predicted_labels == labels) # Tensor containing 'True' for each success
    num_matches = indicator.sum().item()
    return 1 - (num_matches / bs)

def int_to_normal_tensor(index, output_length = 101, sd = 5):
    """
    This function will take an integer (called index) as input and return a
    tensor of length "output_length". This output tensor will contain the evaluation
    of a normal pdf with mean of "index" and standard deviation of "sd" over the 5
    values on either side of "index". The other values in the output tensor will
    all be zeros.
    """
    # First, check to make sure index is an integer
    if not isinstance(index, int):
        raise TypeError("Input index must be an integer, not a {}.".format(type(index)))

    # Now check to make sure the output length is an integer
    if not isinstance(output_length, int):
        raise TypeError("Output length must be an integer, not a {}.".format(type(output_length)))

    # Validate that the output length is at least 1
    if not output_length >= 1:
        raise ValueError("The output length must be at least 1.")

    # Check to make sure index is between 0 and (output_length - 1), inclusive
    if (index < 0) or (index >= output_length):
        raise ValueError("Index must be between 0 and one less than the output length, inclusive.")

    # Check to make sure standard deviation is a float or integer
    if not (isinstance(sd, float) or isinstance(sd, int)):
        raise TypeError("Standard deviation parameter must be a float or integer, not a {}.".format(type(sd)))
    sd = float(sd)

    # Begin setting up output tensor
    normal_dist = norm(index, sd)
    output = torch.arange(101)
    output = torch.Tensor(normal_dist.pdf(output))
    zero_indices = output < 0.05
    output[zero_indices] = 0
    #
    # if (index - 5 > 0) and (index + 5 < 100):
    #     leading_zeros = torch.zeros(index - 5)
    #     trailing_zeros = torch.zeros(95 - index)
    #     normal_output = []
    #     for i in range(index - 5, index + 6):
    #         normal_output.append(normal_dist.pdf(i))
    #     normal_output = torch.Tensor(normal_output)
    #     output = torch.cat([leading_zeros, normal_output, trailing_zeros], dim = 0)
    # elif index - 5 <= 0:
    #     trailing_zeros = torch.zeros(95 - index)
    #     normal_output = []
    #     for i in range(0, index + 6):
    #         normal_output.append(normal_dist.pdf(i))
    #     normal_output = torch.Tensor(normal_output)
    #     output = torch.cat([normal_output, trailing_zeros], dim = 0)
    # elif index + 5 >= 100:
    #     leading_zeros = torch.zeros(index - 5)
    #     normal_output = []
    #     for i in range(index - 5, 101):
    #         normal_output.append(normal_dist.pdf(i))
    #     normal_output = torch.Tensor(normal_output)
    #     output = torch.cat([leading_zeros, normal_output], dim = 0)

    # Normalize output so that all valued add to 1
    output = output / torch.sum(output).item()
    return output

def tensor_to_normal_tensor(labels, output_length = 101, sd = 5):
    """
    Takes as input a 1d tensor called "labels". Passes each label through the
    function int_to_normal_tensor and returns the resulting tensor of tensors.
    """
    # Validate that input is a tensor
    if not isinstance(labels, torch.Tensor):
        raise TypeError("Input should be a tensor, not a {}.".format(type(labels)))

    for entry in labels:
        try:
            output = torch.cat([output, int_to_normal_tensor(entry.item(), output_length = output_length, sd = sd).unsqueeze_(0)], dim = 0)
        except:
            output = int_to_normal_tensor(entry.item(), output_length = output_length, sd = sd).unsqueeze_(0)

    return output

def custom_loss_bcg(output, labels, output_length = 101, sd = 5):
    """
    This defines a custom loss function for the beauty contest game.

    It takes the following inputs:
        -"output" is a 2-D tensor. It is the output of an LSTM neural network of size
            (bs, output_size). Each of the "bs" rows of output is the neural network's
            output to a single input sequence.
        -"labels" is a 1-D tensor. It contains simply the index for the winning number
            for each of the "bs" rounds (hence it is a vector of length "bs").

    This function operates in two main parts.

        1) First, the Tensor "labels" must be formatted properly to take a difference.
                Specifically, it must be converted from a 1-D tensor to a 2-D tensor.
                The way this will be done is that every entry in the initial tensor
                "labels" will be converted to a "output_length"-entry vector with a
                normal distribution imposed around the winning number.

        2) Next, the difference between this updated 2-D label tensor and the initial
                output tensor will be taken and this difference will be returned.

    The function will return the difference between the updated 2-D label tensor
    and the initial output tensor. The return type will always be float.
    """
    # Validate that "output" is a tensor
    if not isinstance(output, torch.Tensor):
        raise TypeError("Expected inputs to be tensors. Instead, received input of unexpected type: {}.".format(type(output)))

    # Reformat labels as 2-D tensor
    target = tensor_to_normal_tensor(labels, output_length = output_length, sd = sd)

    # Take MSE and return this value
    loss = torch.mean((output - target) ** 2)
    return loss

print(int_to_normal_tensor(20, sd = 1))

"""
==================================================
========== BEAUTY CONTEST GAME TRAINING ==========
==================================================
"""

bs = 10
gs = 6
hidden_size = 200
num_layers = 15
bcg_net = generic_net(13, hidden_size, 101, num_layers = num_layers, bias = False)
criterion = nn.CrossEntropyLoss()
lr = 1
sd = 1
network_efficiency = []
epochs = []
total_wins = 0
total_rounds_played = 0

start = time.time()
for epoch in range(1, 101):
    # Learning schedule
    if epoch % 25 == 0:
        lr = lr / 5

    # Setup optimizer
    optimizer = optim.SGD(bcg_net.parameters(), lr = lr)

    # Initialize descriptive stats to zeros
    running_loss = 0
    running_error = 0
    num_batches = 0
    epoch_wins = 0
    within_5 = 0
    net_guess_count = 0

    # Shuffle fictitious players to randomize training
    # shuffle(bcg_players_0_7)

    # Create 'bs' number of groups (bs stands for "batch size")
    # groups = []
    # for i in range(0, bs * gs, gs):
    #     # Initialize blank list for this particular random grouping of responders
    #     group = []
    #     for j in range(gs):
    #         group.append(bcg_players_0_7[i + j])
    #     groups.append(group)
    if epoch % 2 == 0:
        shuffle(real_human_groups_7)
        groups = real_human_groups_7[:bs]
        p_value = 0.7
    else:
        shuffle(real_human_groups_9)
        groups = real_human_groups_9[:bs]
        p_value = 0.9

    for round_number in range(10):
        # If it is the first round, set up first round histories
        if round_number == 0:
            temp = torch.ones((3,))
            p_values = temp.new_full((1, bs, 1), p_value)
            previous_round_win = torch.zeros(1, bs, 1)
            previous_round_guess = torch.zeros(1, bs, 1)
            round_one_binary = torch.ones(1, bs, 1)
            remaining_rounds = torch.zeros(1, bs, 9)
            input_seq = torch.cat([p_values, previous_round_win, previous_round_guess, round_one_binary, remaining_rounds], dim = 2)
            previous_winning_numbers = np.zeros(bs)

        # Initialize network memory to zeros
        h = torch.zeros(2 * num_layers, bs, hidden_size)
        c = torch.zeros(2 * num_layers, bs, hidden_size)

        # Detach prior gradient
        h = h.detach()
        c = c.detach()

        # Begin tracking changes
        h = h.requires_grad_()
        c = c.requires_grad_()
        input_seq = input_seq.requires_grad_()

        # Clear prior gradient from network
        optimizer.zero_grad()

        # Send data through network
        scores, (h, c) = bcg_net(input_seq, h, c)
        net_guesses = scores.argmax(dim = 1)

        # Setup minibatch labels (this part is a bit complicated in BCG)
        minibatch_label = []
        for group_idx in range(len(groups)):
            group = groups[group_idx]
            net_index = len(group)
            group_guesses = [player.responses[round_number] for player in group]
            # group_guesses = [player.respond(previous_winning_numbers[group_idx], round_number) for player in group]
            total_guesses = np.array(group_guesses + [net_guesses[group_idx].item()])
            winning_number = np.mean(total_guesses) * p_value
            if abs(net_guesses[group_idx] - winning_number) < 5:
                within_5 += 1
            net_guess_count += 1
            total_rounds_played += 1
            winning_number_idx = round(winning_number)
            minibatch_label.append(winning_number_idx)
            previous_winning_numbers[group_idx] = winning_number
            if np.argmin(abs(total_guesses - winning_number)) == net_index:
                total_wins += 1
                epoch_wins += 1
                net_won = True
            else:
                net_won = False
        minibatch_label = torch.tensor(minibatch_label, dtype = torch.long)

        # Compute loss
        loss = custom_loss_bcg(scores, minibatch_label, sd = sd)

        # Backward pass
        loss.backward()

        # Do one step of stochastic gradient descent
        normalize_gradient(bcg_net)
        optimizer.step()

        # Update batch statistics
        with torch.no_grad():
            running_loss += loss.item()
            error = get_error(scores, minibatch_label)
            running_error += error
            num_batches += 1

        # Update input sequence to get it ready for next round
        with torch.no_grad():
            if round_number < 9:
                temp = torch.ones((3,))
                p_values = temp.new_full((1, bs, 1), p_value)
                previous_round_win = torch.zeros((1, bs, 1))
                previous_round_guess = torch.zeros((1, bs, 1))
                rounds_passed = torch.zeros(1, bs, round_number + 1)
                round_binary = torch.ones(1, bs, 1)
                if round_number < 8:
                    rounds_remaining = torch.zeros(1, bs, 8 - round_number)
                    input_seq = torch.cat([p_values, previous_round_win, previous_round_guess, rounds_passed, round_binary, rounds_remaining], dim = 2)
                else:
                    input_seq = torch.cat([p_values, previous_round_win, previous_round_guess, rounds_passed, round_binary], dim = 2)

    # At the end of each epoch, print summary statistics
    elapsed = time.time() - start
    avg_loss = running_loss / num_batches
    avg_error = running_error / num_batches
    print('')
    print('| EPOCH {} |'.format(epoch))
    print('='*len('| EPOCH {} |'.format(epoch)))
    print('Error: ', '{}%'.format(avg_error * 100), '\t Loss: ', avg_loss, '\t Time: ', '{} minutes'.format(elapsed / 60))
    print('Total wins this epoch: {} wins out of {} rounds played.'.format(epoch_wins, bs * 10))
    print('Close guesses: ', (100 * (within_5 / net_guess_count)), 'percent')
    print('')
    network_efficiency.append(within_5 / net_guess_count)
    epochs.append(epoch)
z = np.polyfit(epochs, network_efficiency, 5)
f = np.poly1d(z)
plt.plot(epochs, f(epochs), color = "r")
plt.scatter(epochs, network_efficiency, color = "b")
plt.show()

with open('Attempt to overfit real data.csv', 'w+', newline = '') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(epochs)
    csvwriter.writerow(network_efficiency)
    
