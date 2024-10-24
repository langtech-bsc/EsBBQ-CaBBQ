import io
import json
import os
import re
import random

import pandas as pd
from pandas import DataFrame

from utils import create_templating_dicts, split_and_strip, do_slotting, parse_dict_from_string, group_by_specifiers, group_vocab_rows, get_word_combinations

random.seed(1)

# # get all categories from the files
# all_categories = [re.match(r"translation_(\w+).xlsx", fn).group(1) for fn in os.listdir("templates_es") if "translation_" in fn]
all_categories = ["GenderIdentity"] # TEST

# read in vocabulary files
df_vocab: DataFrame = pd.read_excel("templates_es/vocabulary.xlsx", sheet_name="vocabulary_es")
df_vocab_proper_names: DataFrame = pd.read_csv("templates/vocabulary_proper_names.csv") # TODO

unknown_options = [ # TODO
    "Unknown",
    "Cannot be determined",
    "Can't be determined",
    "Not answerable",
    "Not known",
    "Not enough info",
    "Not enough information",
    "Cannot answer",
    "Can't answer",
    "Undetermined",
]

# these have items that need to have the argument order manually reversed # TODO
need_stereotyping_subset = [ # TODO
    "RaceEthnicity",
    "Religion",
    "Nationality",
    "SexualOrientation",
]

# iterate over categories to read all the templates and fill them in
for curr_category in all_categories:
    df_category = pd.read_excel(f"templates_es/translation_{curr_category}.xlsx", na_filter=False)
    output_file = io.open(f"data_es/{curr_category}.jsonl", "w+")  # open the file that all the data will be saved to

    # filter to remove empty items
    df_category = df_category[(df_category.ambiguous_context_es != "") & (df_category.source_es != "-")]
    df_category = df_category.reset_index()

    # initialize item id counter
    nn = 0

    df_category = df_category[df_category.ambiguous_context_es.str.startswith("Ayer hubo una pelea en la puerta")].reset_index() # rm!

    for idx, curr_row in df_category.iterrows():

        names_dict = {}
        grouped_names_dict = {}
        name1_list = []
        name2_list = []

        # select the words from the vocab that match this category
        # (this needs to be done at each row because some rows use custom vocab and some don't)
        df_vocab = df_vocab[(df_vocab.Category == curr_category) & (df_vocab.Name_es != "-")]
        vocab_dict = group_vocab_rows(df_vocab)

        # make sure that one of the words used is the bias target
        bias_targets: str = curr_row.get("final_stereotyped_groups", "")
        bias_targets = split_and_strip(bias_targets, ",")

        # if the list of bias targets is too big, downsample
        if len(targeted_word_list) > 4:
            name1_list = random.sample(targeted_word_list, 5)
        elif len(possible_word_list) < 2:  # these will be handled later
            name1_list = []
        else:
            name1_list = targeted_word_list

        # determine whether NAME1 and NAME2 need to be proper names
        has_proper_name: bool = curr_row.get("Proper_nouns_only") in ["TRUE", "1"]

        # if the word list needs to be further pulled apart within subcategory, do that here
        curr_subcategory = None
        if "Subcategory" in df_category.columns:
            curr_subcategory = curr_row.Subcategory
            if len(curr_subcategory) > 1:
                df_vocab = df_vocab[df_vocab.SubCat == curr_subcategory]

                # re-generate the vocab_dict with the new subset of the subcategory vocabulary
                vocab_dict = group_vocab_rows(df_vocab)
                name1_list = vocab_dict[None]

                # downsample if the list is too big # TODO
                if len(name1_list) > 5:
                    name1_list = random.sample(name1_list, 5)  # for downsampling

        # get special vocabulary from the names column if available
        # (values in the names column always override vocabulary that would otherwise be used)
        names_str = curr_row.get("names_es", "")
        if names_str:
            names_dict = parse_dict_from_string(names_str)
            grouped_names_dict = group_by_specifiers(names_dict)

            # TODO!
            name1_keys = [key for key in names_dict if key.startswith("NAME1")]
            name2_keys = [key for key in names_dict if key.startswith("NAME2")]

            name1_list = names_dict[name1_keys[0]]
            name2_list = names_dict[name2_keys[0]]

        if has_proper_name:

            # TODO
            # this needs to pull from the set of first names that have ethnicity information about them
            if curr_category == "Race_ethnicity": # TODO
                if len(bias_targets) >= 1:
                    first_names_full = df_vocab_proper_names[df_vocab_proper_names.First_last == "first"]
                    first_names = first_names_full[
                        first_names_full.ethnicity.isin(bias_targets)
                    ]
                else:
                    first_names = df_vocab_proper_names[df_vocab_proper_names.First_last == "first"]
                name1_list = first_names.Name.tolist()
                name1_list = random.sample(name1_list, 5)  # for downsampling
            # what to do if this is part of the gender_identity subset
            # this needs to pull from the set of first names that do not have ethnicity information
            elif curr_category == "Gender_identity": # TODO
                first_names = df_vocab_proper_names[df_vocab_proper_names.First_last == "first_only"]
                # specify female so that the other NAME variable can be set to male and there's always a contrast
                female_first_names = first_names[first_names.gender == "F"]
                name1_list = female_first_names.Name.tolist()
                name1_list = random.sample(name1_list, 5)  # for downsampling
            # a handful of templates in other categories use proper nouns
            # these don't have to be specific to gender, but should not be ethnically identifying, so pull from the
            # subset that doesn't correlate strongly to one racial/ethnic category
            else:
                first_names = df_vocab_proper_names[df_vocab_proper_names.First_last == "first_only"]
                name1_list = first_names.Name.tolist()
                name1_list = random.sample(name1_list, 6)  # for downsampling

        # iterate over name1_list
        for name1 in name1_list:
            name1_info = None
            name2_info = None
            words_dict = {}
            name2_info_dict = {}

            # create the name list for name2 here if it hasn't been created yet
            if not name2_list:
                if len(bias_targets) > 0 and curr_category in need_stereotyping_subset:

                    # eliminate words in bias targets
                    name2_list = [name for name in possible_word_list if name not in bias_targets]
                    if len(name2_list) > 4:
                        name2_list = random.sample(name2_list, 5) # downsample # TODO
                else:
                    name2_list = [name for name in name1_list if name != name1]

                # for SES, need to check to make sure it's grabbing from the other category (low/high)
                if curr_category == "SES" and not has_proper_name: # TODO
                    curr_word_cat = df_vocab.loc[df_vocab["Name"] == name1, "Info"].iloc[0]
                    name1_info = curr_word_cat
                    new_options = df_vocab[df_vocab.Info != curr_word_cat]
                    vocab_dict = group_vocab_rows(new_options)
                    name2_list = vocab_dict[None]

                # for gender, keep track of gender of the name picked and select a name of a different gender for the
                # second name variable
                if curr_category == "GenderIdentity" and has_proper_name: # TODO
                    the_word_gender = first_names.loc[first_names["Name"] == name1, "gender"].iloc[0]
                    name1_info = the_word_gender
                    new_options = first_names[first_names.gender != the_word_gender]
                    name2_list = new_options.Name.unique().tolist()
                    name2_list = random.sample(name2_list, 5)  # for downsampling # TODO: check with valle

                # for race/ethnicity... # TODO
                if curr_category == "RaceEthnicity" and has_proper_name:
                    # keep track of gender and ethnicity of the name in question
                    the_word_gender = first_names.loc[
                        first_names["Name"] == name1, "gender"
                    ].iloc[0]
                    the_word_eth = first_names.loc[
                        first_names["Name"] == name1, "ethnicity"
                    ].iloc[0]
                    name1_info = the_word_gender + "-" + the_word_eth
                    # select a last name that matches ethnicity, then put them together
                    last_names = df_vocab_proper_names[
                        (df_vocab_proper_names.First_last == "last")
                        & (df_vocab_proper_names.ethnicity == the_word_eth)
                    ]
                    last_names_list = last_names.Name.unique().tolist()
                    this_last_name = random.choice(last_names_list)
                    name1 = name1 + " " + this_last_name
                    # create a list of names for the second name var w/ same gender
                    # but ethnicity that's not one of the bias targets for that template
                    other_first_names = first_names_full[
                        (~first_names_full.ethnicity.isin(bias_targets))
                        & (first_names_full.gender == the_word_gender)
                    ].reset_index()
                    other_first_names = other_first_names.sample(
                        n=5, replace=False
                    ).reset_index()  # for downsampling
                    other_last_names = df_vocab_proper_names[
                        (df_vocab_proper_names.First_last == "last")
                        & (~df_vocab_proper_names.ethnicity.isin(bias_targets))
                    ].reset_index()
                    # initialize list and dict to keep track of the second name variables to create
                    name2_list = []
                    name2_info_dict = {}
                    for nam in range(len(other_first_names)):
                        frst = other_first_names.Name[nam]
                        eth = other_first_names.ethnicity[nam]
                        gen = other_first_names.gender[nam]
                        name2_info = gen + "-" + eth
                        lst_list = (
                            other_last_names[other_last_names.ethnicity == eth]
                            .Name.unique()
                            .tolist()
                        )
                        lst = random.choice(lst_list)
                        name2_list.append(frst + " " + lst)
                        name2_info_dict[frst + " " + lst] = name2_info

            # iterate over each word in the second word list
            # results in full pairings of every possible vocab pair within the subset
            for name2 in name2_list:

                # handle lexical diversity
                lex_div: str = curr_row.get("lexical_diversity_es", "")
                if lex_div:
                    words_dict = parse_dict_from_string(lex_div)
                    grouped_words_dict = group_by_specifiers(words_dict)

                    word_combinations: list = get_word_combinations(grouped_words_dict)

                    for curr_word_combination in word_combinations:
                        new_row = do_slotting(template_row=curr_row, name1=name1, gs_name1=None, name2=name2, gs_name2=None, grouped_names_dict=grouped_names_dict, grouped_words_dict=grouped_words_dict, word_combination=curr_word_combination)

                        if new_row is None:
                            continue

                        breakpoint()

                # need to record info about the names that were used for easier analysis later
                if (
                    name1_info is not None
                    and curr_category == "Race_ethnicity"
                    and has_proper_name
                ):
                    name2_info = name2_info_dict[name2]
                elif curr_category == "Gender_identity" and has_proper_name:
                    try:
                        name2_info = first_names.loc[
                            first_names["Name"] == name2, "gender"
                        ].iloc[0]
                    except IndexError:
                        name2_info = name2
                    if this_frame_row.NAME1_info[0] != "":
                        # in some cases, there info both from the name selected and info tracked in an info column
                        # in the template. Need to save both of these pieces of info
                        if k == 0:
                            name1_info = this_frame_row.NAME1_info[0] + "_" + name1_info
                        name2_info = this_frame_row.NAME2_info[0] + "_" + name2_info
                elif curr_category == "Nationality" or (
                    curr_category == "SES" and curr_subcategory == "Occupation"
                ):
                    # need to get the relevant info about the name from the vocab file
                    name1_info = df_vocab.loc[df_vocab["Name"] == name1, "Info"].iloc[0]
                    name2_info = df_vocab.loc[df_vocab["Name"] == name2, "Info"].iloc[0]
                elif "NAME1_info" in frame_cols:
                    # for when the info about the name variables is stored in the templates
                    if this_frame_row.NAME1_info[0] != "":
                        name1_info = this_frame_row.NAME1_info[0]
                        name2_info = this_frame_row.NAME2_info[0]
                    else:
                        name1_info = name1
                        name2_info = name2
                else:
                    # if none of the above apply, just store the info as the actual string used in the name
                    name1_info = name1
                    name2_info = name2

                # create four sets of data, each as a dictionary
                dat_formatted = create_templating_dicts(curr_category, new_frame_row, curr_subcategory, unknown_options, frame_cols, bias_targets, name1, name2, name1_info,name2_info, nn)
                nn += 4 # TODO

                for item in dat_formatted:
                    output_file.write(json.dumps(item, default=str, ensure_ascii=False))
                    output_file.write("\n")
                output_file.flush()

                if (curr_category in need_stereotyping_subset) or has_proper_name or (curr_category == "SES" and curr_subcategory == "Occupation"):

                    # flip input of this_word and this_word_2
                    new_frame_row = do_slotting( this_frame_row, name2, None, name1, None, lex_div, rand_wrd1, rand_wrd2, names_dict, words_dict, vocab_dict)

                    if new_frame_row is None:
                        continue

                    # create four sets of data, each as a dictionary
                    dat_formatted = create_templating_dicts(curr_category, new_frame_row, curr_subcategory, unknown_options, frame_cols, bias_targets, name2, name1, name2_info, name1_info, nn)
                    nn += 4 # TODO

                    for item in dat_formatted:
                        output_file.write(json.dumps(item, default=str, ensure_ascii=False))
                        output_file.write("\n")
                    output_file.flush()

    print(f"{curr_category}: generated {nn} sentences total.")

    output_file.close()
