"""
===================================================================================
========== THIS SCRIPT IS FOR TRAINING THE NETWORK IN THE ULTIMATUM GAME ==========
===================================================================================
"""


import numpy as np
from scipy.stats import rv_continuous, kstest, gaussian_kde, ttest_1samp
import os, csv, math, time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib.patches as mpatches

"""
First, set some important, document-wide parameters.

Namely, set the random seed for both Python and Numpy and set the
directory where data is located.
"""
random.seed(12345)
np.random.seed(12345)
os.chdir(r"C:\Users\thehu\OneDrive\Documents\2019-2020\Thesis (Economics)")

"""
===========================================================
===================== DATA FORMATTING =====================
===========================================================
"""

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

CSVFILENAME = 'Cooper and Dutcher.csv'

full_data = list_dictparse(CSVFILENAME)
full_data_sorted = sorted(full_data, key = lambda x: int(x['round']))

for entry in full_data_sorted:
    num_round = int(entry['round'])
    round_index = num_round - 1
    try:
        round_data[round_index].append(entry)
    except:
        round_data.append([entry])

counter = 1
round_dict = {}
for item in round_data:
    # Initialize "blank" lists of proposal and acceptance count for each bin
    proposal_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    acceptance_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for entry in item:
        # Identify bin of proposal
        proposal = float(entry['amount_offered (%)'])
        proposal_index = min(math.floor(proposal / 10), 9)
        # Increment proposal count for that bin by 1
        proposal_count[proposal_index] += 1
        # If offer was accepted, increment acceptance count for that bin by 1
        if int(entry['binary_for_acceptance']) == 1:
            acceptance_count[proposal_index] += 1
    # Store full counts in dictionary
    round_dict[counter] = (proposal_count, acceptance_count)
    counter += 1

round_empirical_distributions = {}
# Loop through each round and assign observed empirical probability of acceptance
# for each bin
for item in round_dict:
    d = round_dict[item]
    empirical_dist = []
    for i in range(10):
        if d[0][i] != 0:
            empirical_prob = d[1][i] / d[0][i]
        else:
            empirical_prob = None
        empirical_dist.append(empirical_prob)
    round_empirical_distributions[item] = empirical_dist

for item in round_empirical_distributions:
    d = round_empirical_distributions[item]
    for i in range(len(d)):
        if i != 0:
            try:
                d[i] = max(float(d[i]), float(d[i - 1]))
            except:
                pass
            if d[i] is None:
                d[i] = d[i - 1]
        else:
            if d[i] is None:
                d[i] = 0

# Truncate dataset to only look at 10 rounds of play
final_round_data = {}
for i in range(1, 11):
    final_round_data[i] = round_empirical_distributions[i]


empirical_dist = final_round_data[1]

# Turn cmf into pmf
round_pmf = []
running_sum = 0
for j in range(len(empirical_dist)):
    observed_maos = max(0, empirical_dist[j] - running_sum)
    round_pmf.append(observed_maos)
    running_sum += observed_maos

round_one_dummy_data = []
for i in range(len(round_pmf)):
    number_obs = math.floor(round_pmf[i] * 5000)
    for j in range(number_obs):
        mao = (np.random.uniform() + i) / 10
        round_one_dummy_data.append(mao)

NUM_ROUNDS = 10


"""
==========================================================
================== DATA GENERATION =======================
==========================================================
"""

# Setup class for generating sample-able continuous pdfs for moa distribution
# in each round
class mao_pdf(rv_continuous):
    """
    Class for creating a pdf, round-by-round, from which samples may be drawn.
    """
    def __init__(self, data):
        super(mao_pdf, self).__init__(a = 0, b = 1)
        self.kde = gaussian_kde(data, bw_method = 0.18)

    def _pdf(self, x):
        return self.kde.evaluate(x)

# Establish container list for a pdf of moa distribution for each round
round_pdfs = []
for i in range(1, NUM_ROUNDS + 1):
    empirical_dist = final_round_data[i]

    # Turn cmf into pmf
    round_pmf = []
    running_sum = 0
    for j in range(len(empirical_dist)):
        observed_maos = max(0, empirical_dist[j] - running_sum)
        round_pmf.append(observed_maos)
        running_sum += observed_maos

    # Generate pseudo-random MAOs in appropriate subintervals
    maos = []

    for j in range(len(round_pmf)):
        number_obs = round(round_pmf[j] * 100)

        for k in range(number_obs):
            mao = np.random.uniform(j, j + 1) / 10
            maos.append(mao)

    # Generate continuous pdf from distribution of moas
    round_pdf = mao_pdf(maos)

    # Add round pdf to container list
    round_pdfs.append(round_pdf)

# Define a Python object to represent a responder in the ultimatum game
class fictitious_player_ug():
    """
    A class for objects that will play the responder role of the ultimatum game
    in ways similar to how humans play.
    """
    def __init__(self):
        # Setup container list for distinct MOA for each round
        self.round_maos = []

        # For each round, randomly choose MOA and add it to container list
        for round_number in range(len(round_pdfs)):
            round_pdf = round_pdfs[round_number]
            round_mao = round_pdf.rvs()
            rounded_round_mao = round(round_mao / 0.05)
            self.round_maos.append(rounded_round_mao * 0.05)

    # Define a method for deciding whether to accept or reject a proposal
    def accept(self, proposal, round_number):
        # Pull MOA for this particular round from container
        relevant_mao = self.round_maos[round_number]
        if proposal >= relevant_mao:
            return True
        return False

    # Define a method for spitting out the index corresponding to responder's
    # MAO in a given round
    def mao_to_tensor_index(self, round_number):
        mao = self.round_maos[round_number]
        return int(mao / 0.05)

# Use for loop to generate many fictitious players and store them in a container list
ug_responders = []
start = time.time()
for i in range(1000):
    responder = fictitious_player_ug()
    ug_responders.append(responder)
    if ( (i + 1) % 50 ) == 0:
        elapsed = time.time() - start
        print('=====')
        print('{} Players Created'.format(i + 1))
        print('{} Minutes Elapsed'.format(round((elapsed / 60), 2)))
        print('Estimated {} Hours Remaining'.format(round((((100000 - i) * (elapsed / i)) / 3600), 2)))
        print('=====')
        print('')


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

# Define a function for taking an integer and outputting a tensor with properly
# formatted geometric distribution for ultimatum game training
def int_to_geometric_tensor(index, output_length = 21, geometric_parameter = 0.75, mao_as_percent = False):
    """
    This function converts a responder's MAO index to a tensor with geometric
    distribution.

    The input is "index", an integer corresponding to the index of the MAO of
    the responder. There is an option to change the length of the output tensor
    if desired. The default value is 21 (to work with the ultimatum game as it
    is played in this analysis). There is another option to set the parameter
    (probability of success) for the geometric distribution, with the default
    being 0.75. There is also an option to enter the MAO as a percent instead
    of as an index. By default this option is false.

    The output of this function is a 1-D tensor with 21 entries. It has a 0 in
    the first (index - 1) entries. The remaining (22 - index) entries are filled
    with a shifted geometric distribution that has been softmaxed so that all
    values sum to 1.
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

    # Check to make sure geometric_parameter is a float
    if not isinstance(geometric_parameter, float):
        raise TypeError("Geometric parameter must be a float, not a {}.".format(type(geometric_parameter)))

    # Check to make sure geometric_parameter is between 0 and 1
    if (geometric_parameter < 0) or (geometric_parameter > 1):
        raise ValueError("Geometric parameter must be between 0 and 1, inclusive.")

    # Check to make sure mao_as_percent is a boolean
    if not isinstance(mao_as_percent, bool):
        raise TypeError("The parameter 'mao_as_percent' must be a boolean, not a {}.".format(type(mao_as_percent)))

    # Begin setting up output tensor
    output_zeros = torch.zeros(index)
    counter = 1
    for x in range(index, output_length):
        f_x = ( (1 - geometric_parameter) ** (counter - 1) ) * geometric_parameter
        try:
            output_nonzeros = torch.cat([output_nonzeros, torch.Tensor([f_x])], dim = 0)
        except:
            output_nonzeros = torch.Tensor([f_x])
        counter += 1

    # Softmax nonzero output values before concatenating
    output_nonzeros = output_nonzeros / torch.sum(output_nonzeros).item()

    # Concatenate zero and nonzero parts of output tensor and return it
    output = torch.cat([output_zeros, output_nonzeros], dim = 0)

    return output

# Define a function for converting a 1-D tensor to 2-D geometric tensor
def tensor_to_geometric_tensor(labels, output_length = 21, geometric_parameter = 0.75, mao_as_percent = False):
    """
    This function will take a 1-D tensor as input and convert it into a 2-D tensor
    in proper geometric format.

    The input of this function is "labels" (for labels of minibatch in neural net
    training), a tensor of integers. Each integer is formatted as a geometric
    tensor which will become a row of the output tensor. There are additional
    options which are the same as the options for the function "int_to_geometric_tensor"
    (these options will be passed directly to that function).

    The output is a properly formatted geometric tensor of size
    (len_labels x output_length).
    """
    # Check to make sure "labels" input is indeed a tensor
    if not type(labels) == torch.Tensor:
        raise TypeError("Input should be a tensor, not a {}.".format(type(labels)))

    # Assemble output tensor
    for entry in labels:
        try:
            output = torch.cat([output, int_to_geometric_tensor(entry.item(), output_length = output_length, geometric_parameter = geometric_parameter, mao_as_percent = mao_as_percent).unsqueeze_(0)], dim = 0)
        except:
            output = int_to_geometric_tensor(entry.item(), output_length = output_length, geometric_parameter = geometric_parameter, mao_as_percent = mao_as_percent)
            output.unsqueeze_(0)

    # Return output tensor
    return output

# Define a function for computing the custom loss of the neural network
def custom_loss_ug(output, labels, output_length = 21, geometric_parameter = 0.75, mao_as_percent = False):
    """
    This defines a custom loss function for the ultimatum game.

    It takes the following inputs:
        -"output" is a 2-D tensor. It is the output of an LSTM neural network of size
            (bs, output_size). Each of the "bs" rows of output is the neural network's
            output to a single input sequence.
        -"labels" is also a 1-D tensor. It contains simply the index for the MAO for
            each of the "bs" responders (hence it is a vector of length "bs").

    This function operates in two main parts.

        1) First, the Tensor "labels" must be formatted properly to take a difference.
                Specifically, it must be converted from a 1-D tensor to a 2-D tensor.
                The way this will be done is that every entry in the initial tensor
                "labels" will be converted to a 21-entry vector with 0s in every entry
                up to the specific index. The remaining entries will be filled with a
                shifted geometric distribution, softmaxed over the remaining entries.

        2) Next, the difference between this updated 2-D label tensor and the initial
                output tensor will be taken and this difference will be returned.

    The function will return the difference between the updated 2-D label tensor
    and the initial output tensor. The return type will always be float.
    """
    # First, verify that both inputs are tensors
    if (type(output) != torch.Tensor):
        raise TypeError("Expected inputs to be tensors. Instead, received input of unexpected type: {}.".format(type(output)))
    if (type(labels) != torch.Tensor):
        raise TypeError("Expected inputs to be tensors. Instead, received input of unexpected type: {}.".format(type(labels)))

    # Reformat labels as 2-D tensor so difference can be taken
    target = tensor_to_geometric_tensor(labels, output_length = output_length, geometric_parameter = geometric_parameter, mao_as_percent = False)

    # Take difference and return it
    loss = torch.mean((output - target) ** 2)
    return loss


"""
=======================================================
========== TESTING DIFFERENT HYPERPARAMETERS ==========
=======================================================
"""

# Initialize blank list to track how well each set of hyperparameters does
network_performance_data = []

# Setup headers to tell which performance data corresponds to which hyperparemeters
network_performance_headers = ['Network'] + list(range(1, 1001))

# Setup training constants
bs = 30
num_layers = 2

# Define hyperparameters to be tested
hidden_sizes = [200, 250, 300]
geometric_parameters = [0.6, 0.75, 0.9]
initial_lrs = [0.1, 0.5, 1.0]

# Loop through hidden sizes
for hidden_size_index in range(len(hidden_sizes)):
    hidden_size = hidden_sizes[hidden_size_index]

    # Loop through geometric parameters
    for geometric_parameter_index in range(len(geometric_parameters)):
        geometric_parameter = geometric_parameters[geometric_parameter_index]

        # Loop through initial learning rates
        for initial_lr_index in range(len(initial_lrs)):
            initial_lr = initial_lrs[initial_lr_index]

            # Initialize neural network with desired hidden size
            ug_net = generic_net(53, hidden_size, 21, num_layers = num_layers)

            # Set initial learning rate
            lr = initial_lr

            # Initialize blank list to track performance of individual network
            network_efficiency = []

            # Start timing the network and begin training
            start = time.time()
            for epoch in range(1, 1001):
                # Learning schedule
                if epoch % 80 == 0:
                    lr = lr / 2

                # Setup optimizer
                optimizer = optim.SGD(ug_net.parameters(), lr = lr)

                # Initialize descriptive stats to zeros
                running_loss = 0
                running_error = 0
                num_batches = 0
                total_winnings = 0
                total_possible = 0

                # Shuffle fictitious players to randomize training
                random.shuffle(ug_responders)

                # Create 'bs' number of groups (bs stands for "batch size")
                groups = []
                for i in range(0, bs * 10, 10):
                    # Initialize blank list for this particular random grouping of responders
                    group = []
                    for j in range(10):
                        group.append(ug_responders[i + j])
                    groups.append(group)

                for round_number in range(10):

                    # Randomly select one opponent from each group for the network to play against
                    opponents = []
                    for group in groups:
                        random.shuffle(group)
                        opponents.append(group[0])

                    # If this is the first round, set up blank round histories
                    if round_number == 0:
                        histories = torch.zeros(1, bs, 52)
                        rounds_remaining = torch.ones(1, bs, 1)
                        input_seq = torch.cat([histories, rounds_remaining], dim = 2)

                    # Setup minibatch labels
                    minibatch_label = []
                    for opponent in opponents:
                        minibatch_label.append(opponent.mao_to_tensor_index(round_number))
                    minibatch_label = torch.tensor(minibatch_label)

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
                    input_seq.size()

                    # Clear prior gradient from network
                    optimizer.zero_grad()

                    # Send data through network
                    scores, (h, c) = ug_net(input_seq, h, c)
                    offer_indices = scores.argmax(dim = 1)

                    # Compute loss on minibatch
                    loss = custom_loss_ug(scores, minibatch_label, geometric_parameter = geometric_parameter)

                    # Backward pass
                    loss.backward()

                    # Do one step of stochastic gradient descent
                    normalize_gradient(ug_net)
                    optimizer.step()

                    # Update batch statistics
                    with torch.no_grad():
                        running_loss += loss.item()
                        error = get_error(scores, minibatch_label)
                        running_error += error
                        num_batches += 1

                    # Update input sequence so it is ready for the next round
                    with torch.no_grad():
                        round_history = torch.zeros(1, bs, 22)
                        for i in range(len(opponents)):
                            offer_index = offer_indices[i].item()
                            offer = offer_index * 0.05
                            accepted = opponents[i].accept(offer, round_number)
                            if accepted:
                                round_history[0, i, 0] = 1
                                round_history[0, i, offer_index + 1] = 1
                                total_winnings += (1 - offer)
                            total_possible += (1 - opponents[i].round_maos[round_number])
                        if round_number != 9:
                            current_round_proposals = scores.view(1, bs, 21)
                            round_remaining_index = 9 - round_number - 1
                            if round_remaining_index > 0:
                                temp1 = torch.zeros(1, bs, round_remaining_index)
                                temp2 = torch.ones(1, bs, 1)
                                temp3 = torch.zeros(1, bs, 9 - round_remaining_index)
                                rounds_remaining = torch.cat([temp1, temp2, temp3], dim = 2)
                            elif round_remaining_index == 0:
                                temp1 = torch.ones(1, bs, 1)
                                temp2 = torch.zeros(1, bs, 9)
                                rounds_remaining = torch.cat([temp1, temp2], dim = 2)
                            input_seq_extension = torch.cat([current_round_proposals, round_history, rounds_remaining], dim = 2)
                            input_seq = torch.cat([input_seq, input_seq_extension], dim = 0)

                # At the end of each epoch, print summary statistics
                elapsed = time.time() - start
                avg_loss = running_loss / num_batches
                avg_error = running_error / num_batches
                print('')
                print('| EPOCH {} |'.format(epoch))
                print('='*len('| EPOCH {} |'.format(epoch)))
                print('Error: ', '{}%'.format(avg_error * 100), '\t Loss: ', avg_loss, '\t Time: ', '{} minutes'.format(elapsed / 60))
                print('Total winnings this epoch: ', total_winnings / total_possible)
                print('')
                network_efficiency.append(total_winnings / total_possible)

            # Format network efficiency data with a header and add it to master data list
            network_output = ['Initial LR - {} | Hidden Size - {} | Geometric Parameter - {}'.format(initial_lr, hidden_size, geometric_parameter)] + network_efficiency
            network_performance_data.append(network_output)

# Save training data to a .CSV file so it may be accessed later
with open('Hyperparameter Testing.csv', 'r', newline = '') as csvfile:
    csvwriter = csv.writer(csvfile, skipinitialspace = True)
    csvwriter.writerow(network_performance_headers)
    for row in network_performance_data:
        csvwriter.writerow(row)


"""
=======================================================
========== TRAINING NEURAL NETWORK OF CHOICE ==========
=======================================================
"""

"""
TRAINING THE NEURAL NETWORK OF CHOICE (so analysis can be completed)
"""
# Setup constants as they have been chosen
bs = 30
num_layers = 2
geometric_parameter = 0.75
lr = 0.1
hidden_size = 200

# Initialize neural network
ug_net = generic_net(53, hidden_size, 21, num_layers = num_layers)

# Start the timer and begin training
start = time.time()
for epoch in range(1, 1001):
    # Learning schedule
    if epoch % 80 == 0:
        lr = lr / 2

    # Setup optimizer
    optimizer = optim.SGD(ug_net.parameters(), lr = lr)

    # Initialize descriptive stats to zeros
    running_loss = 0
    running_error = 0
    num_batches = 0
    total_winnings = 0
    total_possible = 0

    # shuffle fictitious players to randomize training
    random.shuffle(ug_responders)

    # Create 'bs' number of groups (bs stands for "batch size")
    groups = []
    for i in range(0, bs * 10, 10):
        # Initialize blank list for this particular random grouping of responders
        group = []
        for j in range(10):
            group.append(ug_responders[i + j])
        groups.append(group)

    for round_number in range(10):

        # Randomly select one opponent from each group for the network to play against
        opponents = []
        for group in groups:
            random.shuffle(group)
            opponents.append(group[0])

        # If this is the first round, set up blank round histories
        if round_number == 0:
            histories = torch.zeros(1, bs, 52)
            rounds_remaining = torch.ones(1, bs, 1)
            input_seq = torch.cat([histories, rounds_remaining], dim = 2)

        # Setup minibatch labels
        minibatch_label = []
        for opponent in opponents:
            minibatch_label.append(opponent.mao_to_tensor_index(round_number))
        minibatch_label = torch.tensor(minibatch_label)

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
        input_seq.size()

        # Clear prior gradient from network
        optimizer.zero_grad()

        # Send data through network
        scores, (h, c) = ug_net(input_seq, h, c)
        offer_indices = scores.argmax(dim = 1)

        # Compute loss on minibatch
        loss = custom_loss_ug(scores, minibatch_label, geometric_parameter = geometric_parameter)

        # Backward pass
        loss.backward()

        # Do one step of stochastic gradient descent
        normalize_gradient(ug_net)
        optimizer.step()

        # Update batch statistics
        with torch.no_grad():
            running_loss += loss.item()
            error = get_error(scores, minibatch_label)
            running_error += error
            num_batches += 1

        # Update input sequence so it is ready for the next round
        with torch.no_grad():
            round_history = torch.zeros(1, bs, 22)
            for i in range(len(opponents)):
                offer_index = offer_indices[i].item()
                offer = offer_index * 0.05
                accepted = opponents[i].accept(offer, round_number)
                if accepted:
                    round_history[0, i, 0] = 1
                    round_history[0, i, offer_index + 1] = 1
                    total_winnings += (1 - offer)
                total_possible += (1 - opponents[i].round_maos[round_number])
            if round_number != 9:
                current_round_proposals = scores.view(1, bs, 21)
                round_remaining_index = 9 - round_number - 1
                if round_remaining_index > 0:
                    temp1 = torch.zeros(1, bs, round_remaining_index)
                    temp2 = torch.ones(1, bs, 1)
                    temp3 = torch.zeros(1, bs, 9 - round_remaining_index)
                    rounds_remaining = torch.cat([temp1, temp2, temp3], dim = 2)
                elif round_remaining_index == 0:
                    temp1 = torch.ones(1, bs, 1)
                    temp2 = torch.zeros(1, bs, 9)
                    rounds_remaining = torch.cat([temp1, temp2], dim = 2)
                input_seq_extension = torch.cat([current_round_proposals, round_history, rounds_remaining], dim = 2)
                input_seq = torch.cat([input_seq, input_seq_extension], dim = 0)

    # At the end of each epoch, print summary statistics
    elapsed = time.time() - start
    avg_loss = running_loss / num_batches
    avg_error = running_error / num_batches
    print('')
    print('| EPOCH {} |'.format(epoch))
    print('='*len('| EPOCH {} |'.format(epoch)))
    print('Error: ', '{}%'.format(avg_error * 100), '\t Loss: ', avg_loss, '\t Time: ', '{} minutes'.format(elapsed / 60))
    print('Total winnings this epoch: ', total_winnings / total_possible)
    print('')

# Save neural network weights so the trained net can be loaded later
torch.save(ug_net.state_dict(), 'trained_network.pt')


"""
=======================================================
========== ANALYZING PLAY OF TRAINED NETWORK ==========
=======================================================
"""

# Initialize blank list to keep track of the network's proposals
round_offers = []

# Initialize blank placeholders to keep track of network's efficiency
total_winnings = 0
total_possible = 0
efficiency_by_epoch = []
accepted_offers = 0
total_offers = 0
offer_count = 0
sum_of_offers = 0

# Setup constants of network
bs = 30
num_layers = 2
hidden_size = 200

# Instantiate neural network and load trained parameters
ug_net = generic_net(53, hidden_size, 21, num_layers = num_layers)
ug_net.load_state_dict(torch.load('trained_network.pt'))
ug_net.eval()

# Collect data on 6000 network proposals
for epoch in range(1, 21):
    # Start keeping track of winnings for this particular epoch
    epoch_winnings = 0
    epoch_possible_winnings = 0

    # Shuffle fictitious players to randomize data collection
    random.shuffle(ug_responders)

    # Create 'bs' number of groups (bs stands for "batch size")
    groups = []
    for i in range(0, bs * 10, 10):
        # Initialize blank list for this particular random grouping of responders
        group = []
        for j in range(10):
            group.append(ug_responders[i + j])
        groups.append(group)

    for round_number in range(10):

        # Randomly select one opponent from each group for the network to play against
        opponents = []
        for group in groups:
            random.shuffle(group)
            opponents.append(group[0])

        # If this is the first round, set up blank round histories
        if round_number == 0:
            histories = torch.zeros(1, bs, 52)
            rounds_remaining = torch.ones(1, bs, 1)
            input_seq = torch.cat([histories, rounds_remaining], dim = 2)

        # Initialize network memory to zeros
        h = torch.zeros(2 * num_layers, bs, hidden_size)
        c = torch.zeros(2 * num_layers, bs, hidden_size)

        # Send data through network
        scores, (h, c) = ug_net(input_seq, h, c)
        offer_indices = scores.argmax(dim = 1)

        # Keep track of offers neural network makes in each round
        for offer in offer_indices:
            try:
                round_offers[round_number].append(round(offer.item() * 0.05, 2))
            except:
                round_offers.append([round(offer.item() * 0.05, 2)])

        # Update input sequence so it is ready for the next round
        with torch.no_grad():
            round_history = torch.zeros(1, bs, 22)
            for i in range(len(opponents)):
                offer_index = offer_indices[i].item()
                offer = offer_index * 0.05
                sum_of_offers += offer
                offer_count += 1
                accepted = opponents[i].accept(offer, round_number)
                if accepted:
                    round_history[0, i, 0] = 1
                    round_history[0, i, offer_index + 1] = 1
                    total_winnings += (1 - offer)
                    epoch_winnings += (1 - offer)
                    accepted_offers += 1
                total_offers += 1
                total_possible += (1 - opponents[i].round_maos[round_number])
                epoch_possible_winnings += (1 - opponents[i].round_maos[round_number])
            if round_number != 9:
                current_round_proposals = scores.view(1, bs, 21)
                round_remaining_index = 9 - round_number - 1
                if round_remaining_index > 0:
                    temp1 = torch.zeros(1, bs, round_remaining_index)
                    temp2 = torch.ones(1, bs, 1)
                    temp3 = torch.zeros(1, bs, 9 - round_remaining_index)
                    rounds_remaining = torch.cat([temp1, temp2, temp3], dim = 2)
                elif round_remaining_index == 0:
                    temp1 = torch.ones(1, bs, 1)
                    temp2 = torch.zeros(1, bs, 9)
                    rounds_remaining = torch.cat([temp1, temp2], dim = 2)
                input_seq_extension = torch.cat([current_round_proposals, round_history, rounds_remaining], dim = 2)
                input_seq = torch.cat([input_seq, input_seq_extension], dim = 0)

    efficiency_by_epoch.append(epoch_winnings / epoch_possible_winnings)

# Check to see what percentage of the network's offers are accepted
print(accepted_offers / total_offers)

# Check to see what the network's average offer is
print(sum_of_offers / offer_count)

efficiency_by_epoch = np.array(efficiency_by_epoch)
t_score, p_value = ttest_1samp(efficiency_by_epoch, 0.6375)
print(p_value)


with open('neural_net_proposals.csv', 'w+', newline = '') as csvfile:
    csvwriter = csv.writer(csvfile, skipinitialspace = True)
    for row in round_offers:
        csvwriter.writerow(row)

# Print final network efficiency
print("Network Efficiency: ", total_winnings / total_possible)

"""
Analyzing how network's decisions change as it trains.
"""
# Setup constants as they have been chosen
bs = 30
num_layers = 2
geometric_parameter = 0.75
lr = 0.1
hidden_size = 200
offers = []
round_offers = []

# Initialize neural network
ug_net = generic_net(53, hidden_size, 21, num_layers = num_layers)

# Start the timer and begin training
start = time.time()
for epoch in range(1, 401):
    # Setup blank "suboffers" list if it is the first epoch
    if epoch == 1:
        suboffers = []
        subround_offers = {
            0: [],
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
            8: [],
            9: []
        }
    # Learning schedule
    if epoch % 80 == 0:
        lr = lr / 2

    # Setup optimizer
    optimizer = optim.SGD(ug_net.parameters(), lr = lr)

    # Initialize descriptive stats to zeros
    running_loss = 0
    running_error = 0
    num_batches = 0
    total_winnings = 0
    total_possible = 0

    # Shuffle fictitious players to randomize training
    random.shuffle(ug_responders)

    # Create 'bs' number of groups (bs stands for "batch size")
    groups = []
    for i in range(0, bs * 10, 10):
        # Initialize blank list for this particular random grouping of responders
        group = []
        for j in range(10):
            group.append(ug_responders[i + j])
        groups.append(group)

    for round_number in range(10):

        # Randomly select one opponent from each group for the network to play against
        opponents = []
        for group in groups:
            random.shuffle(group)
            opponents.append(group[0])

        # If this is the first round, set up blank round histories
        if round_number == 0:
            histories = torch.zeros(1, bs, 52)
            rounds_remaining = torch.ones(1, bs, 1)
            input_seq = torch.cat([histories, rounds_remaining], dim = 2)

        # Setup minibatch labels
        minibatch_label = []
        for opponent in opponents:
            minibatch_label.append(opponent.mao_to_tensor_index(round_number))
        minibatch_label = torch.tensor(minibatch_label)

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
        input_seq.size()

        # Clear prior gradient from network
        optimizer.zero_grad()

        # Send data through network
        scores, (h, c) = ug_net(input_seq, h, c)
        offer_indices = scores.argmax(dim = 1)
        for offer in offer_indices:
            suboffers.append(offer.item())
            subround_offers[round_number].append(offer.item())

        # Compute loss on minibatch
        loss = custom_loss_ug(scores, minibatch_label, geometric_parameter = geometric_parameter)

        # Backward pass
        loss.backward()

        # Do one step of stochastic gradient descent
        normalize_gradient(ug_net)
        optimizer.step()

        # Update batch statistics
        with torch.no_grad():
            running_loss += loss.item()
            error = get_error(scores, minibatch_label)
            running_error += error
            num_batches += 1

        # Update input sequence so it is ready for the next round
        with torch.no_grad():
            round_history = torch.zeros(1, bs, 22)
            for i in range(len(opponents)):
                offer_index = offer_indices[i].item()
                offer = offer_index * 0.05
                accepted = opponents[i].accept(offer, round_number)
                if accepted:
                    round_history[0, i, 0] = 1
                    round_history[0, i, offer_index + 1] = 1
                    total_winnings += (1 - offer)
                total_possible += (1 - opponents[i].round_maos[round_number])
            if round_number != 9:
                current_round_proposals = scores.view(1, bs, 21)
                round_remaining_index = 9 - round_number - 1
                if round_remaining_index > 0:
                    temp1 = torch.zeros(1, bs, round_remaining_index)
                    temp2 = torch.ones(1, bs, 1)
                    temp3 = torch.zeros(1, bs, 9 - round_remaining_index)
                    rounds_remaining = torch.cat([temp1, temp2, temp3], dim = 2)
                elif round_remaining_index == 0:
                    temp1 = torch.ones(1, bs, 1)
                    temp2 = torch.zeros(1, bs, 9)
                    rounds_remaining = torch.cat([temp1, temp2], dim = 2)
                input_seq_extension = torch.cat([current_round_proposals, round_history, rounds_remaining], dim = 2)
                input_seq = torch.cat([input_seq, input_seq_extension], dim = 0)

    # At the end of each epoch, print summary statistics
    elapsed = time.time() - start
    avg_loss = running_loss / num_batches
    avg_error = running_error / num_batches
    print('')
    print('| EPOCH {} |'.format(epoch))
    print('='*len('| EPOCH {} |'.format(epoch)))
    print('Error: ', '{}%'.format(avg_error * 100), '\t Loss: ', avg_loss, '\t Time: ', '{} minutes'.format(elapsed / 60))
    print('Total winnings this epoch: ', total_winnings / total_possible)
    print('')

    # If it has been 20 epochs, append list "suboffers" to larger list "offers"
    if epoch % 20 == 0:
        offers.append(suboffers)
        suboffers = []
        round_offers.append(subround_offers)
        subround_offers = {
            0: [],
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
            8: [],
            9: []
        }

# First 20
plt.hist(offers[0])
plt.show()

plt.hist(offers[1])
plt.show()

plt.hist(offers[-1])
plt.show()

# Save data showing overall distribution of offers and distribution of offers by
# round
with open("Overall Training Distribution of Offers.csv", 'w+', newline = '') as csvfile:
    csvwriter = csv.writer(csvfile, skipinitialspace = True)
    for i in range(len(offers)):
        header = ['Epochs {} - {}'.format((20 * i) + 1, (20 * i) + 20)]
        csvwriter.writerow(header + offers[i])





SPE_OFFER = 0.05
earnings = 0
possible_earnings = 0
rejections = 0
offers = 0
for responder in ug_responders:
    for round_number in range(10):
        if responder.accept(SPE_OFFER, round_number):
            earnings += 1.0
        else:
            rejections += 1
        possible_earnings += (1 - responder.round_maos[round_number])
        offers += 1


print(earnings / possible_earnings)
print(rejections)
print(offers)
print(earnings)


# Computing average expected earnings from random proposals
def expected_earnings(mao):
    """
    Takes as input a responder's MAO and returns the average earnings that the network
    could expect from making a random offer to this responder.
    """
    return (.5 - (mao - ((mao ** 2) / 2)))

total_expected_earnings = 0
total_possible_earnings = 0
for responder in ug_responders:
    for round_number in range(10):
        mao = responder.round_maos[round_number]
        total_expected_earnings += expected_earnings(mao)
        total_possible_earnings += 1 - mao

print(total_expected_earnings / total_possible_earnings)
