"""
========================================================================
========== THIS SCRIPT IS FOR ANALYZING HUMAN DECISIONS IN UG ==========
========================================================================
"""

import csv, os, math
from scipy.stats import rv_continuous, kstest, gaussian_kde
import numpy as np
import matplotlib.pyplot as plt

os.chdir(r"C:\Users\thehu\OneDrive\Documents\2019-2020\Thesis (Economics)")

round_data = []

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

def listparse(csvfilename):
    """
    Parses CSV into a list of lists.
    """
    output = []

    with open(csvfilename, 'r', newline = '') as csvfile:
        csvreader = csv.reader(csvfile, skipinitialspace = True)
        for row in csvreader:
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

# Turn empirical observed probabilitiy of acceptance into CMF
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

NUM_ROUNDS = 10

# Create a class for continuous random variable to approximate PMF
class mao_pdf(rv_continuous):
    """
    Class for creating a pdf, round-by-round, from which samples may be drawn.
    """
    def __init__(self, data):
        super(mao_pdf, self).__init__(a = 0, b = 1)
        self.kde = gaussian_kde(data, bw_method = 0.18)

    def _pdf(self, x):
        return self.kde.evaluate(x)

# Establish container list for a pdf of mao distribution for each round
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

# Find the expected value for responder MAO for each round
expected_values_by_round = {}
for i in range(len(round_pdfs)):
    expected_values_by_round[i + 1] = round_pdfs[i].expect()

# Determining efficiency of human players
estimated_total_possible = 0
total_proposer_winnings = 0
for observation in full_data_sorted:
    round_number = int(observation['round'])
    if round_number > 10:
        continue
    else:
        estimated_total_possible += (1 - expected_values_by_round[round_number])
        if int(observation['binary_for_acceptance']):
            total_proposer_winnings += 1 - (float(observation['amount_offered (%)']) / 100)

# Determining efficiency if human players were only allowed to make offers in
# increments of 0.05
total_winnings = 0
total_possible_winnings = 0
for entry in full_data:
    proposal = float(entry['amount_offered (%)'])
    round_number = int(entry['round'])
    if round_number > 10:
        continue
    if (proposal % 5) == 0:
        if int(entry['binary_for_acceptance']):
            total_winnings += (1 - (proposal / 100))
        total_possible_winnings += (1 - expected_values_by_round[round_number])
    else:
        new_proposal = 0.05 * math.ceil(proposal / 5)
        if int(entry['binary_for_acceptance']):
            total_winnings += (1 - (new_proposal))
        total_possible_winnings += (1 - expected_values_by_round[round_number])

# Print efficiency of human players
print(total_proposer_winnings / estimated_total_possible)

# Print efficiency of human proposers if restricted to offers of multiples of 0.05
print(total_winnings / total_possible_winnings)

# Determine the average offer made by human proposers
sum_of_offers = 0
sum_of_accepted_offers = 0
offer_count = 0
accepted_offer_count = 0
for entry in full_data:
    round_number = int(entry['round'])
    if round_number > 10:
        continue
    offer = float(entry['amount_offered (%)']) / 100
    sum_of_offers += offer
    offer_count += 1
    if int(entry['binary_for_acceptance']):
        sum_of_accepted_offers += offer
        accepted_offer_count += 1

# Print average offer by human proposers and average accepted offer
print(sum_of_offers / offer_count)
print(sum_of_accepted_offers / accepted_offer_count)

# Count how many human proposals are in increments of 0.05 and how
# many of these are accepted
increments = 0
total = 0
non_increments = 0
accepted_non_increments = 0
for entry in full_data:
    proposal = float(entry['amount_offered (%)'])
    if (proposal % 5) == 0:
        increments += 1
        if int(entry['binary_for_acceptance']):
            accepted_increments += 1
            total_winnings += (1 - (proposal / 100))
    else:
        if int(entry['binary_for_acceptance']):
            accepted_non_increments += 1
        non_increments += 1
    total += 1

print(1 - (increments / total))
print(accepted_non_increments / non_increments)
