"""
=====================================================================
========== THIS SCRIPT IS FOR PROPERLY FORMATTING BCG DATA ==========
=====================================================================
"""


import os, csv, math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous, gaussian_kde

BCG_DATA_STRING1 = """0.7	50	20	5	10	10	13	7	5	4	0
0.7	100	55	30	25	12	20	14	8	2	0
0.7	20	5	5	7	7	10	8	2	1	0
0.7	70	0	5	5	100	10	9	6	2	1
0.7	10	25	100	13	10	11	9	6	2	0
0.7	95	50	20	25	15	20	12	8	3	1
0.7	5	20	8	25	8	12	11	5	1	0
0.7	60	15	15	10	10	7	3	5	4	3
0.7	50	35	25	10	10	10	2	10	5	0
0.7	10	20	25	12	13	7	1	7	5	100
0.7	50	70	25	100	12	8	3	8	8	2
0.7	1	24	21	7	18	3	0	8	3	1
0.7	80	50	18	9	12	10	100	3	4	3
0.7	25	15	18	10	15	7	2	5	2	1
0.7	70	40	20	3	6	4	0	0	0	0
0.7	21	27	20	47	1	0	0	0	0	7
0.7	73	25	5	2	4	3	1	0	0	0
0.7	15	10	5	2	3	0	0	1	100	2
0.7	80	20	7	4	6	0	1	0	0	0
0.7	1	15	2	5	4	2	1	0	0	0
0.7	50	25	7	1	5	1	0	0	0	0
0.7	38	47	25	21	7	2	2	2	7	7
0.7	23	10	19	20	11	3	2	2	5	6
0.7	70	45	15	20	15	9	5	3	5	5
0.7	70	35	20	18	10	4	0	100	4	5
0.7	50	20	5	10	5	0	2	1	0	100
0.7	10	15	15	17	8	2	1	1	12	5
0.7	70	40	100	20	10	5	5	1	15	10
0.7	75	30	10	10	3	1	1	0	100	0
0.7	50	20	10	8	6	1	1	0	0	0
0.7	0	25	20	5	5	2	0	0	0	0
0.7	80	0	10	6	5	1	0	0	0	0
0.7	90	20	20	7	5	2	1	0	0	25
0.7	95	40	18	20	3	4	1	0	4	6
0.7	10	100	50	10	5	0	0	0	0	0
0.7	70	60	30	11	11	3	3	3	3	1
0.7	100	30	25	5	10	8	3	3	2	1
0.7	97	50	25	15	10	5	4	2	2	1
0.7	100	0	5	7	6	7	6	3	1	1
0.7	50	50	20	10	5	4	2	2	2	1
0.7	50	60	35	20	10	6	4	4	1	1
0.7	70	45	10	8	4	4	3	3	1	0
0.7	60	40	20	10	4	50	22	12	5	2
0.7	60	20	15	7	5	15	12	9	7	2
0.7	55	20	12	9	5	16	18	9	7	3
0.7	50	45	35	20	5	27	24	10	4	3
0.7	25	25	15	8	100	8	20	10	5	2
0.7	43	40	10	9	100	30	16	9	6	2
0.7	60	15	15	10	6	60	20	14	7	2
0.9	20	20	10	2	80	10	4	4	20	15
0.9	7	8	9	8	6	8	8	6	3	14
0.9	30	10	2	2	2	10	9	6	2	10
0.9	60	18	28	8	11	19	9	0	100	20
0.9	80	30	8	8	4	10	8	6	3	10
0.9	18	40	28	20	12	18	16	15	9	22
0.9	53	29	21	12	9	16	11	6	23	15
0.9	0	0	0	0	0	0	0	0	0	0
0.9	40	48	22	10	1	1	0	0	0	100
0.9	45	32	23	10	4	0	0	1	0	0
0.9	40	45	27	10	1	0	0	0	0	0
0.9	57	37	24	9	3	1	0	1	1	0
0.9	49	38	25	9	1	0	0	0	0	0
0.9	47	30	25	8	10	0	2	0	0	0
0.9	35	40	28	16	7	7	2	2	1	1
0.9	50	40	30	20	18	5	3	1	0	1
0.9	65	42	25	18	9	6	3	0	0	1
0.9	40	50	31	17	9	5	1	1	1	1
0.9	1	10	19	19	12	2	1	2	1	1
0.9	95	60	45	20	13	7	3	1	1	1
0.9	100	45	25	22	12	7	2	1	1	100
0.9	0	20	28	31	36	26	18	0	8	5
0.9	90	93	88	40	35	32	21	17	15	9
0.9	50	40	30	28	27	26	24	18	10	5
0.9	42	36	8	30	30	28	22	18	10	8
0.9	44	40	32	29	31	26	21	17	11	7
0.9	50	36	45	80	34	25	23	16	10	5
0.9	60	20	35	33	34	25	23	18	10	10
0.9	55	58	41	36	24	19	19	21	21	21
0.9	39	53	43	32	23	20	20	20	22	23
0.9	71	34	30	34	27	19	17	20	21	21
0.9	50	50	50	50	50	50	50	50	50	50
0.9	68	60	50	35	25	20	18	21	22	23
0.9	60	62	55	50	30	30	28	26	20	40
0.9	92	50	43	4	21	19	19	20	22	23
0.9	50	60	55	50	42	15	10	3	100	0
0.9	50	50	49	30	20	9	8	0	0	10
0.9	70	65	52	38	23	22	2	3	1	1
0.9	55	70	40	35	25	30	5	3	0	30
0.9	90	55	47	28	19	9	4	2	1	19
0.9	80	50	43	31	20	11	4	3	1	70
0.9	75	65	50	35	20	5	5	0	100	18
0.9	68	54	54	37	20	12	10	10	10	4
0.9	80	50	40	35	25	20	15	14	8	5
0.9	45	53	53	25	30	30	15	15	8	8
0.9	90	30	38	35	33	25	15	10	5	5
0.9	75	55	42	35	25	20	17	10	8	5
0.9	96	59	41	40	31	20	16	10	6	2
0.9	10	50	41	35	25	20	14	9	9	3"""

BCG_DATA_STRING2 = """0.7	68	28	25	27	23	12	8	4	1	1
0.7	33	65	27	23	19	13	13	8	3	1
0.7	35	35	60	15	7	14	10	5	4	1
0.7	40	30	15	25	25	20	8	3	3	100
0.7	20	29	56	27	30	20	10	9	3	0
0.7	7	25	30	35	30	18	18	8	3	1
0.7	7	7	50	30	20	15	7	7	2	1
0.7	28	15	100	52	35	29	24	20	18	25
0.7	60	15	20	20	18	20	25	18	23	15
0.7	0	35	25	35	27	25	25	17	28	16
0.7	38	28	12	27	27	25	25	25	25	21
0.7	21	7	16	45	22	24	13	23	18	19
0.7	45	23	20	17	35	21	18	24	16	25
0.7	32	12	100	14	100	18	100	17	100	12
0.7	20	23	25	18	10	13	16	16	16	12
0.7	50	21	23	5	8	100	20	14	12	9
0.7	35	35	35	30	10	17	23	20	17	9
0.7	7	50	25	9	9	18	22	12	10	5
0.7	31	30	15	15	90	12	15	17	15	13
0.7	35	35	14	14	9	14	20	8	8	8
0.7	30	25	25	12	10	12	24	50	20	18
0.7	34	22	24	13	25	23	35	22	23	25
0.7	88	44	19	30	22	29	17	23	20	16
0.7	50	45	60	40	100	20	50	40	20	14
0.7	23	40	40	30	21	25	25	25	15	10
0.7	53	67	60	50	40	40	30	20	20	12
0.7	50	25	60	60	45	30	25	26	20	10
0.7	50	49	28	28	22	28	25	25	22	12
0.7	5	20	35	17	10	3	3	1	50	25
0.7	33	45	20	18	6	3	2	2	14	25
0.7	80	38	25	15	5	1	2	2	10	100
0.7	1	21	27	13	17	13	5	27	20	10
0.7	35	23	14	10	6	4	3	2	13	23
0.7	35	21	16	10	6	6	6	10	100	9
0.7	60	20	15	13	7	3	10	100	15	30
0.7	35	40	21	15	15	14	14	20	25	22
0.7	50	18	12	8	18	10	100	18	13	17
0.7	21	35	27	5	18	10	12	22	11	19
0.7	7	10	17	19	14	27	70	77	37	35
0.7	90	50	15	7	18	15	15	20	20	15
0.7	5	10	14	10	16	16	15	20	21	17
0.7	75	59	8	93	31	47	23	39	21	27
0.7	49	40	61	15	14	9	53	55	47	39
0.7	88	4	26	33	48	21	32	11	28	25
0.7	80	30	45	25	15	99	30	25	30	100
0.7	88	40	15	20	20	10	21	33	33	33
0.7	25	31	24	33	14	81	29	35	25	32
0.7	66	60	10	18	12	7	40	40	36	35
0.7	40	65	45	25	9	7	100	100	100	0
0.9	10	40	50	45	39	39	36	29	17	10
0.9	50	65	52	40	41	39	30	28	20	5
0.9	40	22	45	49	35	34	31	27	18	7
0.9	50	40	36	40	43	38	36	20	18	8
0.9	80	45	38	45	45	38	30	26	13	9
0.9	80	50	49	50	45	45	37	29	15	13
0.9	21	50	70	40	41	35	33	25	20	5
0.9	90	45	45	43	36	26	30	26	25	22
0.9	81	52	44	43	33	30	40	34	30	22
0.9	10	49	50	50	35	30	33	32	29	20
0.9	18	45	46	40	37	100	25	32	29	19
0.9	7	50	45	43	35	30	40	40	25	20
0.9	50	40	45	43	40	33	35	30	28	22
0.9	70	60	55	45	40	25	35	30	27	22
0.9	60	64	50	34	27	20	15	9	2	18
0.9	49	45	35	38	34	30	25	12	99	15
0.9	50	45	42	34	29	21	15	10	6	13
0.9	62	50	40	37	32	25	20	11	5	10
0.9	22	39	44	38	37	30	19	15	12	14
0.9	58	39	42	40	35	28	18	14	8	50
0.9	38	35	45	40	34	25	20	13	5	10
0.9	74	25	57	48	52	26	31	20	13	8
0.9	50	49	45	35	25	25	23	20	15	7
0.9	46	52	45	40	35	78	18	16	12	9
0.9	47	43	40	35	27	23	25	19	13	7
0.9	49	41	43	37	29	20	23	17	8	7
0.9	3	53	43	37	30	24	27	20	12	8
0.9	80	67	41	39	31	41	26	18	15	7
0.9	65	70	68	68	60	55	50	48	40	20
0.9	88	60	58	56	55	50	40	38	30	18
0.9	20	70	70	65	52	46	53	44	32	25
0.9	49	79	69	63	58	45	38	36	33	20
0.9	13	45	55	60	56	51	45	37	37	37
0.9	9	40	65	61	55	45	37	39	28	20
0.9	65	55	60	60	58	50	60	35	20	22
0.9	38	38	54	50	45	31	31	27	23	16
0.9	33	66	54	50	45	37	29	25	23	14
0.9	53	60	55	49	47	43	39	25	20	15
0.9	51	60	53	54	44	40	37	25	16	18
0.9	77	77	57	55	50	10	39	39	30	20
0.9	100	50	55	50	45	100	32	30	20	15
0.9	75	60	57	50	43	37	32	24	19	13
0.9	88	58	48	43	38	99	35	35	36	33
0.9	27	31	93	42	30	87	46	35	41	33
0.9	42	53	53	49	43	42	39	42	44	38
0.9	74	41	37	42	40	33	42	40	40	40
0.9	16	32	36	42	38	35	40	40	40	37
0.9	66	50	45	50	40	38	50	50	42	36
0.9	31	39	41	43	40	33	47	61	33	31"""

bcg_list1 = BCG_DATA_STRING1.split('\n')
bcg_data = []
bcg_data_for_stata = [['p_value', 'guess', 'prior_guess']]
for row in bcg_list1:
    row_list = row.split('\t')
    new_row = []
    for item in row_list:
        new_row.append(float(item))
    bcg_data.append(new_row)
bcg_list2 = BCG_DATA_STRING2.split('\n')
bcg_data2 = []
for row in bcg_list2:
    row_list = row.split('\t')
    new_row = []
    for item in row_list:
        new_row.append(float(item))
    bcg_data.append(new_row)

# Add correct group numbers to each player
for j in range(0, len(bcg_data), 7):
    group_data = bcg_data[j : j + 7]
    group_number = math.floor(j / 7)
    for entry in group_data:
        entry.append(group_number)

# Set headings for .csv file
bcg_data_headings = ['p_value', 'round1', 'round2', 'round3', 'round4', 'round5', 'round6', 'round7', 'round8', 'round9', 'round10', 'group']

os.chdir(r"C:\Users\thehu\OneDrive\Documents\2019-2020\Thesis (Economics)\Beauty Contest Game data")
with open('Beauty Contest Game Data.csv', 'w+', newline = '') as csvfile:
    csvwriter = csv.writer(csvfile, skipinitialspace = True)
    csvwriter.writerow(bcg_data_headings)
    for row in bcg_data:
        csvwriter.writerow(row)

for i in range(len(bcg_data)):
    try:
        if len(bcg_data[i]) != 11:
            bcg_data.pop(i)
    except:
        pass

winning_numbers = {}
bcg_array = np.array(bcg_data)

for i in range(0, len(bcg_data), 7):
    group_winning_numbers = {}
    group = bcg_data[i : i + 7]
    group_avg = np.mean(group, axis = 0)
    p = round(group_avg[0], 2)
    for index in range(1, len(group_avg)):
        winning_number = p * group_avg[index]
        group_winning_numbers[index] = winning_number
    winning_numbers[(i / 7) + 1] = group_winning_numbers

full_stata_data = []
for i in range(0, len(bcg_data), 7):
    group = bcg_data[i : i + 7]
    group_num = (i / 7) + 1
    p = group[0][0]
    for player in group:
        for round_number in range(1, len(player)):
            try:
                data_point = [p, round_number, winning_numbers[group_num][round_number - 1], player[round_number], group_num]
            except:
                data_point = [p, round_number, '.', player[round_number], group_num]
            full_stata_data.append(data_point)

with open('Stata BCG Data.csv', 'w+', newline = '') as csvfile:
    csvwriter = csv.writer(csvfile, skipinitialspace = True)
    csvwriter.writerow(['p_value', 'round_number', 'previous_winning_number', 'guess', 'group_id'])
    for row in full_stata_data:
        csvwriter.writerow(row)

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

updated_bcg_data = listparse('New Updated BCG Data.csv')

print(updated_bcg_data[:3])

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

print(len(updated_bcg_data))

# Formatting first round responses
first_round_responses_0_7 = []
first_round_responses_0_9 = []
for i in range(1, 1961, 10):
    p_value = updated_bcg_data[i][0]
    first_round_response = float(updated_bcg_data[i][-3])
    if float(p_value) == 0.7:
        first_round_responses_0_7.append(first_round_response)
    elif float(p_value) == 0.9:
        first_round_responses_0_9.append(first_round_response)

newspaper_data = listparse('Newspaper Data.csv')
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


"""
======================================================
======== ANALYZING DECISIONS OF HUMAN PLAYERS ========
======================================================
"""
# Initiate lists with row headers to sort human responses by round
# and p-value
real_human_responses_7 = [
    ["round1", 0.7],
    ["round2", 0.7],
    ["round3", 0.7],
    ["round4", 0.7],
    ["round5", 0.7],
    ["round6", 0.7],
    ["round7", 0.7],
    ["round8", 0.7],
    ["round9", 0.7],
    ["round10", 0.7]
]
real_human_responses_9 = [
    ["round1", 0.9],
    ["round2", 0.9],
    ["round3", 0.9],
    ["round4", 0.9],
    ["round5", 0.9],
    ["round6", 0.9],
    ["round7", 0.9],
    ["round8",0.9],
    ["round9", 0.9],
    ["round10", 0.9]
]

# Sort human responses by round and p-value
for entry in updated_bcg_data[1:]:
    round_number = int(entry[1])
    if round_number > 10:
        continue
    guess = float(entry[3])
    p_value = float(entry[0])
    if p_value == 0.7:
        real_human_responses_7[round_number - 1].append(guess)
    else:
        real_human_responses_9[round_number - 1].append(guess)

# Save data to .csv file so it can be accessed from different Python script
with open('BCG - Human Responses by Round.csv', 'w+', newline = '') as csvfile:
    csvwriter = csv.writer(csvfile, skipinitialspace = True)
    for row in real_human_responses_7:
        csvwriter.writerow(row)
    for row in real_human_responses_9:
        csvwriter.writerow(row)
