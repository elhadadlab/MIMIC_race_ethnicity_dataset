import pandas as pd
import numpy as np
import argparse
import sys
# import glob
from collections import Counter
path_to_src = "./"  # noqa
sys.path.append(path_to_src)  # noqa
# path_to_src = "./modeling/src/"  # noqa
# sys.path.append(path_to_src)  # noqa
import read_annotation_files_utils as utils
# import data_utils as data_utils
# import re
# from importlib import reload
import pprint
pp = pprint.PrettyPrinter(indent=4)


def read_data_files_example(re_assignment_dir):
    """
    Description: A function to show how to read in the race and ethnicity indicator and assignment
        annotation files.
    Input:
        race_assignemt_dir (str): Directory containing all subdirectories of raw RE assignments.
        collected_data_dir (str): Directory to the collected data files.
    Output:
    """
    chatty = True
    ####################################################################
    ##################### READING IN SHARED SUBSET #####################
    ####################################################################
    print("READING SHARED SUBSET")
    shared_re_assignments, shared_assignment_df_list = utils.collect_race_assignments_shared_subset(
        re_assignemt_all_dir=re_assignment_dir,
        chatty=chatty,
        return_individual_annotations=True
    )
    print("Shared assignments shape:", shared_re_assignments.shape)
    ####################################################################
    ################### READING IN INDIVIDUAL SUBSET ###################
    ####################################################################
    print("READING INDIVIDUAL SUBSET")
    individual_re_assignments = utils.collect_race_assignments_inidividual_subset(
        re_assignemt_all_dir=re_assignment_dir,
        chatty=chatty
    )
    print("Individual assignments shape:", individual_re_assignments.shape)
    ####################################################################
    ################### COMBINE SHARED AND INDIVIDUAL ##################
    ####################################################################
    print("COMBINING SHARED AND INDIVIDUAL")
    assert np.all(
        individual_re_assignments.columns.values ==
        shared_re_assignments.columns.values
    ), "cannot concat dataframes because column names"
    all_re_assignments_df = pd.concat(
        [shared_re_assignments, individual_re_assignments],
        axis='index',
        ignore_index=True,
    )
    print("Combined assignments shape:", all_re_assignments_df.shape)

    ####################################################################
    ###################### LOADING COLLECTED DATA ######################
    ####################################################################
    collected_all_re_assignments_df = pd.read_json(
        collected_data_dir + 'all_re_assignments_df.jsonl',
        lines=True
    )  # should be identical to all_re_assignments_df
    collected_individual_re_assignments = pd.read_json(
        collected_data_dir + 'individual_re_assignments.jsonl',
        lines=True
    )  # should be identical to individual_re_assignments
    collected_shared_re_assignments = pd.read_json(
        collected_data_dir + 'shared_re_assignments.jsonl',
        lines=True
    )  # should be identical to shared_re_assignments
    collected_indicator_assignments_df = pd.read_json(
        collected_data_dir + 'indicators_assignments_df.jsonl',
        lines=True,
    )
    # all_span_length = np.array(collected_indicator_assignments_df.spans.apply(lambda x: len(x)))
    # np.sum(all_span_length != 0)

    ####################################################################
    ######################### ITERATE SPAN DATA ########################
    ####################################################################
    all_span_length = np.array(
        collected_indicator_assignments_df.spans.apply(lambda x: len(x)))
    sents_with_at_least_one_span = np.sum(all_span_length != 0)
    indicators_overall = []
    indicators_sentence = []
    indicator_spans = []
    for idx, row in collected_indicator_assignments_df.iterrows():
        curr_indicators = []
        # Each row contains a list of dictionaries with span data
        for span in row.spans:
            curr_indicators.append(span['label'])
            if "start" in span:
                span_str = row.text[span["start"]: span["end"]]
            else:
                span_str = "MISSING_TEXT_FOR_" + span['label']
            indicator_spans.append(span_str)
        indicators_overall.extend(curr_indicators)
        indicators_sentence.extend(list(set(curr_indicators)))
    print("Overall indicator counts", Counter(indicators_overall))
    print("Sentence indicator counts", Counter(indicators_sentence))
    print("Sentences with at leasrt one indicator", sents_with_at_least_one_span)
    print("Spans associated with indicators")
    pp.pprint(Counter(indicator_spans))

# re_assignment_dir = 'dataSaves/raw_race_ethnicity_assignment_annotations/'
# collected_data_dir = 'dataSaves/collected_data/'


def main():
    parser = argparse.ArgumentParser(
        description='Collects indicator and assignment annotations to one file.')
    parser.add_argument('-ra', '--race_assignemt_dir',
                        help='Folder to read RE assignments')
    parser.add_argument('-cd', '--collected_data_dir',
                        help='Path to the collected data folder.')

    args = parser.parse_args()
    collect_indicator_assignment_annotations_together(
        re_assignment_dir=args.race_assignemt_dir,
        collected_data_dir=args.collected_data_dir,
    )


if __name__ == '__main__':
    main()
