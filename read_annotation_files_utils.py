import glob
import random
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import numpy as np
import os
# from fuzzywuzzy import process
from collections import defaultdict, Counter
import re
from tqdm import tqdm
import nltk
nltk.download('punkt')


race_ethnicity_assignment_cols = [
    'RACE Native American or Alaska Native',
    'RACE Black or African American',
    'RACE Asian',
    'RACE Native Hawaiian or Other Pacific Islander',
    'RACE White',
    'RACE Not Covered',
    'RACE No Information Indicated',
    'ETH Hispanic/Latino/Latina/Latinx',
    'ETH Non-Hispanic/Non-Latino/Non-Latina/Non-Latinx',
    'ETH Not Covered',
    'ETH No Information Indicated'
]

race_assignment_cols = [
    'RACE Native American or Alaska Native',
    'RACE Black or African American',
    'RACE Asian',
    'RACE Native Hawaiian or Other Pacific Islander',
    'RACE White',
    'RACE Not Covered',
    'RACE No Information Indicated',
]

ethnicity_assignment_cols = [
    'ETH Hispanic/Latino/Latina/Latinx',
    'ETH Non-Hispanic/Non-Latino/Non-Latina/Non-Latinx',
    'ETH Not Covered',
    'ETH No Information Indicated'
]


def read_individual_assignment_annotation_df(annotation_dir):
    """
    Description: Reads in and does some safety checks for an individual assignment df
    Input:
        annotation_dir (str): Path to directory with annotated files.
    Output:
        annot_df(pandas df): A dataframe of all the concatenated jsonl dataframes.
    TODO:
        1) Consider moving this to data_utils
        2) Need check on the columns to make sure no one redordered them
    """
    annotation_path = glob.glob(
        annotation_dir + "individual_annotation/*.xlsx")
    assert len(annotation_path) == 1, "Did not find individual annotation file."
    annotation_path = annotation_path[0]
    col_names = [
        "sentence_id",
        "text",
        "RACE Native American or Alaska Native",
        "RACE Black or African American",
        "RACE Asian",
        "RACE Native Hawaiian or Other Pacific Islander",
        "RACE White",
        "RACE Not Covered",
        "RACE No Information Indicated",
        "ETH Hispanic/Latino/Latina/Latinx",
        "ETH Non-Hispanic/Non-Latino/Non-Latina/Non-Latinx",
        "ETH Not Covered",
        "ETH No Information Indicated"]
    race_cols = col_names[2:9]
    eth_cols = col_names[9:]
    dtype_dict = {
        "sentence_id": np.int64,
        "text": str,
        "RACE Native American or Alaska Native": str,
        "RACE Black or African American": str,
        "RACE Asian": str,
        "RACE Native Hawaiian or Other Pacific Islander": str,
        "RACE White": str,
        "RACE Not Covered": str,
        "RACE No Information Indicated": str,
        "ETH Hispanic/Latino/Latina/Latinx": str,
        "ETH Non-Hispanic/Non-Latino/Non-Latina/Non-Latinx": str,
        "ETH Not Covered": str,
        "ETH No Information Indicated": str,
    }
    annot_df = pd.read_excel(
        annotation_path, skiprows=[0], names=col_names, dtype=dtype_dict,
        keep_default_na=False, usecols=list(range(13)))
    column_checks_passed = check_column_validity(
        annotation_path=annotation_path)
    assert column_checks_passed, "\tFile {} does not pass column checks".format(
        annotation_path)
    checks_passed = check_assignment_annotation_validity(re_assignment_df=annot_df,
                                                         race_cols=race_cols, eth_cols=eth_cols)
    # Commenting out below for now. Will need these redone later.
    assert checks_passed, "\tFile {} does not pass checks".format(
        annotation_path)
    annot_df = convert_df_to_binary(
        annot_df=annot_df, cols2binarize=(race_cols + eth_cols))
    return(annot_df)


def collect_race_assignments_shared_subset(re_assignemt_all_dir, chatty,
                                           return_individual_annotations=False):
    """
    Description: Collects race and ethnicity assignments together. Combined using majority vote
    Input:
        re_assignment_dir (str): Directory containing all subdirectories of RE assignemtns.
        chatty (bool): True if print out while reading in data
        return_individual_annotations (bool): If true, also returns list of individual annotator
            annotations. Defaults to False
    Output:
        re_assignments_df (pandas df): Contains race ethnicity assignments
    TODO:
        1)
    """
    all_annotator_directories = glob.glob(
        re_assignemt_all_dir + "annotator_*/")
    if chatty:
        print("Found {} directories".format(len(all_annotator_directories)))
        for dir in all_annotator_directories:
            print("\t" + dir)

    assignment_df_list = []
    for dir in all_annotator_directories:
        if chatty:
            print("Reading: " + dir)
        assignments_df = collect_assignment_annotations_to_one_df(dir)
        assert assignments_df.shape == (
            5834, 13), "bad shape for returned dataframe"
        assignment_df_list.append(assignments_df)
    for my_index, assignments_df in enumerate(assignment_df_list):
        if my_index == 0:
            previous_df = assignments_df
            continue
        assert np.all(
            assignments_df.sentence_id.values == previous_df.sentence_id.values)
        previous_df = assignments_df
    re_assignments_df = determine_majority_vote_assignment(
        assignment_df_list=assignment_df_list
    )
    if return_individual_annotations:
        return(re_assignments_df, assignment_df_list)
    else:
        return(re_assignments_df)


def collect_race_assignments_inidividual_subset(re_assignemt_all_dir, chatty):
    """
    Description: Collects and concats all individual race and ethnicity assignments.
    Input:
        re_assignment_dir (str): Directory containing all subdirectories of RE assignemtns.
        chatty (bool): True if print out while reading in data
    Output:
        re_assignments_df (pandas df): Contains race ethnicity assignments
    TODO:
        1)
    """
    all_annotator_directories = glob.glob(
        re_assignemt_all_dir + "annotator_*/")
    if chatty:
        print("Found {} directories".format(len(all_annotator_directories)))
        for dir in all_annotator_directories:
            print("\t" + dir)
    individual_assignment_df_list = []
    for dir in all_annotator_directories:
        if chatty:
            print("Reading: " + dir)
        individual_assignments_df = read_individual_assignment_annotation_df(
            dir)
        individual_assignment_df_list.append(individual_assignments_df)

    re_assignments_df = pd.concat(
        individual_assignment_df_list, ignore_index=True)
    return(re_assignments_df)


def collect_assignment_annotations_to_one_df(annotation_dir):
    """
    Description: Collects all race and ethnicity assignment xlsx annotation files for a single
        annotator into one dataframe
    Input:
        annotation_dir (str): Path to directory with annotated files.
    Output:
        annot_df(pandas df): A dataframe of all the concatenated jsonl dataframes. This is sorted
            by sentence_id at the end.
    TODO:
        1) Consider moving this to data_utils
    """
    # annotation_dir = "dataSaves/race_ethnicity_assignment_annotations/annotator_8_annotations/"
    all_annotation_paths = glob.glob(annotation_dir + "*.xlsx")
    # print(all_annotation_paths)
    col_names = [
        "sentence_id",
        "text",
        "RACE Native American or Alaska Native",
        "RACE Black or African American",
        "RACE Asian",
        "RACE Native Hawaiian or Other Pacific Islander",
        "RACE White",
        "RACE Not Covered",
        "RACE No Information Indicated",
        "ETH Hispanic/Latino/Latina/Latinx",
        "ETH Non-Hispanic/Non-Latino/Non-Latina/Non-Latinx",
        "ETH Not Covered",
        "ETH No Information Indicated"]
    race_cols = col_names[2:9]
    eth_cols = col_names[9:]
    dtype_dict = {
        "sentence_id": np.int64,
        "text": str,
        "RACE Native American or Alaska Native": str,
        "RACE Black or African American": str,
        "RACE Asian": str,
        "RACE Native Hawaiian or Other Pacific Islander": str,
        "RACE White": str,
        "RACE Not Covered": str,
        "RACE No Information Indicated": str,
        "ETH Hispanic/Latino/Latina/Latinx": str,
        "ETH Non-Hispanic/Non-Latino/Non-Latina/Non-Latinx": str,
        "ETH Not Covered": str,
        "ETH No Information Indicated": str,
    }
    all_annotation_dfs_list = []
    for annotation_path in all_annotation_paths:
        print(annotation_path)
        # print("Come back and read in the df again and check that column values make sense")
        # Need to protect against ppl switching order of columns or god forbid adding in new
        # columns
        temp_df = pd.read_excel(
            annotation_path, skiprows=[0], names=col_names, dtype=dtype_dict,
            keep_default_na=False, usecols=list(range(13)))
        # perform some checks.
        # 1)    check that every row has at least one race and one ethnicity.
        column_checks_passed = check_column_validity(
            annotation_path=annotation_path)
        assert column_checks_passed, "\tFile {} does not pass column checks".format(
            annotation_path)
        checks_passed = check_assignment_annotation_validity(re_assignment_df=temp_df,
                                                             race_cols=race_cols, eth_cols=eth_cols)
        assert checks_passed, "\tFile {} does not pass checks".format(
            annotation_path)
        # removing checking for now. Put back in later.
        all_annotation_dfs_list.append(temp_df)
    annot_df = pd.concat(all_annotation_dfs_list, ignore_index=True)
    annot_df = convert_df_to_binary(
        annot_df=annot_df, cols2binarize=(race_cols + eth_cols))
    annot_df.loc[:, race_cols + eth_cols].sum()
    annot_df = annot_df.sort_values(by=["sentence_id"])
    annot_df.reset_index(inplace=True, drop=True)
    return(annot_df)


def check_column_validity(annotation_path):
    """
    Description: Some annotators added or replaced columns, so I need to make sure they're still
        in the expected order.
    Input:
        annotation_path (str): Path to annotation excel file.
    Output:
        checks_passed (bool): True if columns are well formed.
    TODO:
        1)
    """
    checks_passed = True
    expected_columns = [['Unnamed: 0'],
                        ['Unnamed: 1'],
                        ['Native American or Alaska Native'],
                        ['Black or African American'],
                        ['Asian'],
                        ['Native Hawaiian or Other Pacific Islander'],
                        ['White'],
                        ['Not Covered', "None of the above"],
                        ['No Information Indicated'],
                        ['Hispanic/Latino/Latina/Latinx'],
                        ['Non-Hispanic/Non-Latino/Non-Latina/Non-Latinx'],
                        ['Not Covered.1', "None of the above.1"],
                        ['No Information Indicated.1']
                        ]

    temp_df = pd.read_excel(
        annotation_path, skiprows=[0], usecols=list(range(13)))
    # print(temp_df.columns)
    for idx, col in enumerate(temp_df.columns):
        if col not in expected_columns[idx]:
            print("\tAnnotation file {} has unexpected column {} that does not match"
                  "expected ({})".format(annotation_path, col, expected_columns[idx]))
            checks_passed = False
    return(checks_passed)


def convert_df_to_binary(annot_df, cols2binarize):
    """
    Description: Checks to make sure that an RE assignment dataframe is valid. The checks are below
        1) check that every row has at least one race and one ethnicity. By this time, all values
        should have been checked, so anything that is not '""' is considered a hit.
    Input:
        annot_df (pandas df): Contains race and ethnicity assignment columns, sentence and
            sentence id.
    Output:
        binarized_df (pandas df): Same as above, but all values are binarized
    TODO:
        1)
    """
    binarized_df = annot_df.copy(deep=True)
    for col in cols2binarize:
        curr_col = binarized_df.loc[:, col]
        binarized_df.loc[curr_col != "", col] = 1
        binarized_df.loc[curr_col == "", col] = 0
        binarized_df.loc[:, col] = binarized_df.loc[:, col].astype(np.int64)
    return(binarized_df)


def check_assignment_annotation_validity(re_assignment_df, race_cols, eth_cols):
    """
    Description: Checks to make sure that an RE assignment dataframe is valid. The checks are below
        1) check that every row has at least one race and one ethnicity.
    Input:
        re_assignment_df (pandas df):
    Output:
        checks_passed (bool): True if the code ran without issues.
    TODO:
        1) Also check that other assignments and "no mention" are mutually exclusive
    """
    checks_passed = True
    all_sentence_ids = re_assignment_df.sentence_id.values
    all_sentences_text = re_assignment_df.text.values
    # check for at least one assignment
    for col_type, cols in zip(["race", "ethnicity"], [race_cols, eth_cols]):
        row_sums = (re_assignment_df.loc[:, cols] != "").sum(axis=1)
        at_least_one_annot_per_row = np.mean(row_sums > 0) == 1.0
        if not at_least_one_annot_per_row:
            print("\tSome rows have no {} annotations. Expected 1.0, got {}".format(
                col_type, np.mean(row_sums > 0)))
            bad_rows = np.where(row_sums == 0)[0]
            for bad_row in bad_rows:
                print("\tCheck sentence ID : {}\t{}\n\n".format(
                    all_sentence_ids[bad_row], all_sentences_text[bad_row]))
            checks_passed = False
    # check that values make sense.
    expected_values = set(["", "x", "X", "xx", "s", "x ", 'x         '])
    for col in race_cols + eth_cols:
        curr_col = re_assignment_df.loc[:, col]
        unique_vals = set(curr_col.unique())
        if not unique_vals.issubset(expected_values):
            print("\tFound unaccounted for values in column " + col)
            print("\tValues: {}".format(unique_vals))
            checks_passed = False
            break
    return(checks_passed)


def determine_majority_vote_assignment(assignment_df_list):
    """
    Description: Calculates IAA using a 1 vs all approach. The all is a majority vote on
        assignments
        and is used as groundtruth.
    Input:
        assignment_df_list (list(pandas df)): List of RE assignemtn dataframes. It is assumed
            that all dfs are of the same sentences in the same order.
    Output:
        majority_vote_df (pandas df): A single pandas df with the same sentences and columns
            as the input list, but assignments are made using majority voting. Binarized.
    TODO:
        1) Make sure this is performing correctly.
    """
    majority_vote_df = None
    for df in assignment_df_list:
        if majority_vote_df is None:
            majority_vote_df = df.copy(deep=True)
        else:
            assert majority_vote_df.shape == df.shape, "bad shape for returned dataframe"
            majority_vote_df[race_ethnicity_assignment_cols] += \
                df[race_ethnicity_assignment_cols]
    min_vote_needed = np.ceil(len(assignment_df_list) / 2)
    vote_mask = majority_vote_df[race_ethnicity_assignment_cols] >= min_vote_needed
    majority_vote_df[race_ethnicity_assignment_cols] = 0
    majority_vote_df[race_ethnicity_assignment_cols] = majority_vote_df[race_ethnicity_assignment_cols].mask(
        vote_mask, other=1)
    return(majority_vote_df)
