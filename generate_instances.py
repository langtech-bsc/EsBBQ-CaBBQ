import argparse
import json
import os
import random
import re
# from tabulate import tabulate

import pandas as pd

from utils import (
    fill_template,
    flip_names_dict_keys,
    generate_examples,
    get_lex_div_combinations,
    group_by_specifiers,
    parse_dict_from_string,
    parse_list_from_string,
    flatten_nested_dicts
)

# set seed to avoid non-determinism
random.seed(1)

# overwrite the get() function of pandas Series to perform str.strip before returning when the value is a string
# this saves us from worrying too much about extra spaces in the Excel spreadsheets
original_get = pd.Series.get

def new_get(self, key, default=None):
    item = original_get(self, key, default)
    return item.strip() if isinstance(item, str) else item

pd.Series.get = new_get

# formats available for the output
output_format_choices = ["jsonl", "csv"]

# get list of categories from the filenames available in the templates folder
all_categories = [re.match(r"final_(\w+).xlsx", fn).group(1) for fn in os.listdir("templates_es") if fn.startswith("final_")]

# create parser for the CLI arguments
parser = argparse.ArgumentParser(prog="Generate BBQ Instances", description="This script will read the Excel files in the input folder and generate BBQ instances for all the categories.")
parser.add_argument("--categories", nargs="+", choices=all_categories, default=all_categories, help="Space-separated list of categories to process templates and generate instances. If not passed, will run for all available categories.")
parser.add_argument("--minimal", action="store_true", help="Minimize he sources of variation in instances by only taking one option from each source of variation.")
parser.add_argument("--output-formats", nargs="+", choices=output_format_choices, default=["jsonl"], help=f"Space-separated format(s) in which to save the examples. Choices are {output_format_choices}.")
parser.add_argument("--dry-run", action="store_true", help="Generate the templates and print the logs and optional statistics but don't actually save them to file.")
parser.add_argument("--variation-stats", action="store_true", help="Print a table of variation statistics per category.")

args = parser.parse_args()

# read and pre-process vocabulary files
df_vocab = pd.read_excel("templates_es/vocabulary_es.xlsx", sheet_name="vocabulary_es").fillna("")
df_proper_names = pd.read_excel("templates_es/vocabulary_proper_names_es.xlsx", sheet_name="vocabulary_proper_names_es").fillna("")

df_vocab = df_vocab[["Category", "SubCat", "Name_es", "f", "Info", "stereotyped"]].map(str.strip)
df_proper_names = df_proper_names[["Name", "gender", "ethnicity"]].map(str.strip)

# options for the unknown answer
unknown_answer_options = [
    "No hay suficiente información",
    "No se puede saber",
    "No se puede determinar"
]

if args.minimal:
    unknown_answer_options = unknown_answer_options[:1]

# # Variation stats
# _variation_stats = pd.DataFrame(index=args.categories)
# _name1_lengths = []
# _name2_lengths = []
# _lex_div_lengths = []

# iterate over categories to read all the templates and fill them in
for curr_category in args.categories:

    # read the category's Excel spreadsheet of templates
    df_category = pd.read_excel(f"templates_es/final_{curr_category}.xlsx", sheet_name="Sheet1", na_filter=False).fillna("")

    # initialize list that will contain all the examples generated for all the templates
    all_generated_examples: list[dict] = []

    # filter to remove empty items
    df_category = df_category[(df_category.ambiguous_context_es != "") & (df_category.source_es != "-")]

    print(f"[{curr_category}] Imported {len(df_category)} templates.")

    _flipping_allowed_count = 0

    # iterate over template rows to generate examples for one template at a time
    for _, curr_row in df_category.iterrows():
        """
        ROW CONFIG
        """

        # dictionaries that will store pre-processed values from the names column and the lexical diversity column
        grouped_names_dict = {}
        grouped_lex_div_dict = {}

        # options and information for NAME1 values
        name1_list = []
        name1_info = ""
        name1_info_dict = {}

        # options and information for NAME2 values
        name2_list = []
        name2_info = ""
        name2_info_dict = {}

        # pre-process the column of stated gender in the cases where the template should be used only for one gender
        stated_gender: str = curr_row.get("Stated_gender_info", "").lower()
        if "fake-" in stated_gender:
            # treat the case of "fake-m" and "fake-f" (used to differentiate from "m/f" when it's only grammatical gender)
            stated_gender = stated_gender.split("-")[1]

        assert stated_gender in ["m", "f", ""], f"Invalid value for Stated_gender_info: `{stated_gender}`"

        # store whether the row is an exceptional template where the names cannot be flipped
        flipping_allowed: bool = curr_row.get("flip_names", True) is not False
        _flipping_allowed_count += int(flipping_allowed)

        """
        NAME1 AND NAME2 VALUES
        """

        # select the words from the vocab that match the current category
        df_vocab_cat = df_vocab[(df_vocab.Category == curr_category) & (df_vocab.Name_es != "-")]

        # filter by subcategory if there is one
        curr_subcategory = curr_row.get("Subcategory")
        if curr_subcategory:
            if curr_category == "GenderIdentity":
                # skip this category because we already subverted the subcategory issue by including all the vocabulary in the names column
                pass
            else:
                # filter the vocabulary for the given subcategory only
                df_vocab_cat = df_vocab_cat[df_vocab_cat.SubCat == curr_subcategory]

        # parse the list of stereotyped groups (i.e. bias targets) that the current template refers to
        bias_targets: str = parse_list_from_string(curr_row.final_stereotyped_groups)

        # determine whether NAME1 and NAME2 need to be proper names
        proper_nouns_only: bool = bool(curr_row.get("Proper_nouns_only", ""))

        if proper_nouns_only: # rm!
            continue # rm! this is a temporary filter

        if proper_nouns_only:

            # for RaceEthnicity, generate NAME1 options using names associated with the targeted ethnicities
            if curr_category == "RaceEthnicity":
                assert bias_targets, "Category is RaceEthnicity but the template doesn't have any specified stereotyped groups!"

                df_first_names = df_proper_names[df_proper_names.First_last == "first"]
                df_first_names = df_first_names[df_first_names.ethnicity.isin(bias_targets)]

                # filter by gender if the template is set to apply to only one gender
                # (also includes names without a specific gender i.e. genderless names)
                if stated_gender:
                    df_first_names = df_first_names[df_first_names.gender.isin([stated_gender, ""])]
                
                for _, first_name_row in df_first_names.iterrows():

                    # select all possible last names with the same ethnicity
                    last_name_options = df_proper_names[(df_proper_names.First_last == "last") & (df_proper_names.ethnicity == first_name_row.ethnicity)].Name.unique().tolist()

                    for last_name in last_name_options:
                        full_name = f"{first_name_row.Name} {last_name}"
                        name_info = f"{first_name_row.gender}, {first_name_row.ethnicity}"

                        name1_list.append(full_name)
                        name1_info_dict[full_name] = name_info

            # for GenderIdentity with proper names, NAME1 is always female and NAME2 is always male
            elif curr_category == "GenderIdentity":
                # we use only white names to avoid introducing additional bias by comparing between stereotyped ethnicities
                df_first_names = df_proper_names[
                    (df_proper_names.First_last == "first") &
                    (df_proper_names.ethnicity == "blanco")
                ]

                # force NAME1 to be female
                df_female_first_names = df_first_names[df_first_names.gender == "f"]
                name1_list = df_female_first_names.Name.tolist()
                name1_info = "f"

                # force NAME2 to be male
                df_male_first_names = df_first_names[df_first_names.gender == "m"]
                name2_list = df_male_first_names.Name.unique().tolist()
                name2_info = "m"

            # In the case of other categories that use proper nouns, they can be of any gender but we don't want them to elicit any specific ethnicities so we take only typical white Spanish names
            else:
                df_first_names = df_proper_names[(df_proper_names.First_last == "first") & (df_proper_names.ethnicity == "blanco")]

                if stated_gender:
                    # if the template states a specific gender to use, restrict to this gender or to genderless names
                    df_first_names = df_first_names[df_first_names.gender.isin([stated_gender, ""])]

                name1_list = df_first_names.Name.tolist()

        else: # if not using proper names

            if curr_category == "SES" and curr_subcategory == "Occupation":
                # for SES Occupations, options for NAME1 are all the occupations listed in the vocabulary
                name1_list = df_vocab_cat.Name_es.unique().tolist()

            elif curr_category != "GenderIdentity":
                # options for NAME1 are the simply the bias targets
                name1_list = bias_targets

            if not name2_list:
                # initialize the options for NAME2 as the groups marked in the vocabulary as non-stereotyped, if any
                df_vocab_non_stereotyped = df_vocab_cat[df_vocab_cat.stereotyped == "no"]
                name2_list = df_vocab_non_stereotyped.Name_es.unique().tolist()

            # get specific vocabulary from the names column if available
            # (values in the names column always override vocabulary that would otherwise be used,
            # and they can only be used when the template is not for proper names)
            names_str = curr_row.get("names_es", "")
            if names_str:
                names_dict = parse_dict_from_string(names_str)
                grouped_names_dict = group_by_specifiers(names_dict)

                name1_list = grouped_names_dict["NAME1"][None]
                name2_list = grouped_names_dict["NAME2"][None]

                name1_info = curr_row.get("NAME1_info")
                name2_info = curr_row.get("NAME2_info")

        if args.minimal:
            name1_list = name1_list[:1]

        assert name1_list, "No names in the list of options for NAME1!"

        """
        LEXICAL DIVERSITY
        """
        lex_div_str: str = curr_row.get("lexical_diversity_es", "")
        if lex_div_str:
            lex_div_dict = parse_dict_from_string(lex_div_str)
            grouped_lex_div_dict = group_by_specifiers(lex_div_dict)

            # generate all the possible combinations of WORD1, WORD2, ..., WORD<N>
            lex_div_combinations: list = get_lex_div_combinations(grouped_lex_div_dict)

        else:
            # if there is no lexical diversity, create a list with one combination set to None so that we can still loop over the "combinations" and only generate one
            lex_div_combinations = [None]
            grouped_lex_div_dict = {}

        if args.minimal:
            lex_div_combinations = lex_div_combinations[:1]

        # _name1_lengths.append(len(name1_list))
        # _lex_div_lengths.append(len(lex_div_combinations))

        """
        NAME1 LOOP
        Iterate over the list of possible values for NAME1
        """

        for name1 in name1_list:
            # try to set info field if it hasn't been determined yet
            if not name1_info:
                if name1 in name1_info_dict:
                    name1_info = name1_info_dict[name1]
                else:
                    name1_info = curr_row.get("NAME1_info")

            # set possible values for NAME2 here when they depend on NAME1
            # (in which case they could not be determined before the NAME1 loop)
            if curr_category == "SES":
                if proper_nouns_only:
                    # for SES with proper names, take NAME2 from the remaining white names different from NAME1
                    df_other_first_names = df_proper_names[
                        (df_proper_names.First_last == "first") &
                        (df_proper_names.ethnicity == "blanco") &
                        (df_proper_names.Name != name1)
                    ]

                    if stated_gender:
                        df_other_first_names = df_other_first_names[df_other_first_names.gender.isin([stated_gender, ""])]

                    name2_list = df_other_first_names.Name.tolist()

                elif curr_subcategory == "Occupation":
                # for SES Occupations, NAME1 options can be either highSES or lowSES, so we ensure that NAME2 options are always in the opposite category
                    name1_info = df_vocab_cat.loc[df_vocab.Name_es == name1, "Info"].iloc[0]
                    name2_list = df_vocab_cat[df_vocab_cat.Info != name1_info].Name_es.unique().tolist()

            # for race/ethnicity, NAME2 options should be the same gender as NAME1 but with the non-stereotyped ethnicity
            elif curr_category == "RaceEthnicity" and proper_nouns_only:

                # find the non-stereotyped ethnicity
                non_stereotyped_groups = df_vocab[df_vocab.stereotyped == "no"].Name_es.unique().tolist()

                name1_gender = name1_info.split(",")[0]
                df_other_first_names = df_proper_names[
                    (df_proper_names.First_last == "first") &
                    (df_proper_names.ethnicity.isin(non_stereotyped_groups)) &
                    (df_proper_names.gender == name1_gender)
                ]
                df_other_last_names = df_proper_names[
                    (df_proper_names.First_last == "last") &
                    (df_proper_names.ethnicity.isin(non_stereotyped_groups))
                ]

                # iterate over the first names to match them with all last name options
                for _, first_name_row in df_other_first_names.iterrows():
                    last_name_options = df_other_last_names[df_other_last_names.ethnicity == first_name_row.ethnicity].Name.unique().tolist()

                    for last_name in last_name_options:
                        full_name = f"{first_name_row.Name} {last_name}"
                        name2_info = f"{first_name_row.gender}, {first_name_row.ethnicity}"

                        name2_list.append(full_name)
                        name2_info_dict[full_name] = name2_info

            assert name2_list, "No names in the list of options for NAME2!"

            if args.minimal:
                name2_list = name2_list[:1]

            # _name2_lengths.append(len(name2_list))

            """
            NAME2 LOOP
            Iterate over all the names in the NAME2 list
            """
            for name2 in name2_list:

                # record info
                if curr_category == "RaceEthnicity" and proper_nouns_only:
                    name2_info = name2_info_dict[name2]

                elif curr_category == "GenderIdentity" and proper_nouns_only:
                    # if there is already some info on the row, append it to the existing information
                    if curr_row.get("NAME1_info"):
                        name1_info = f"{name1_info}, {curr_row.get('NAME1_info')}"
                    if curr_row.get("NAME2_info"):
                        name2_info = f"{name2_info}, {curr_row.get('NAME2_info')}"

                elif curr_category == "SES" and curr_subcategory == "Occupation":
                    # get the info about each name from the vocab
                    name1_vocab_row = df_vocab.loc[df_vocab.Name_es == name1].iloc[0]
                    name1_info = name1_vocab_row.get("Info")
                    name2_vocab_row = df_vocab.loc[df_vocab.Name_es == name2].iloc[0]
                    name2_info = name2_vocab_row.get("Info")

                # if there is still no info, use the name itself
                if not name1_info:
                    name1_info = name1

                if not name2_info:
                    name2_info = name2

                # iterate over combinations to generate every possible version of the texts
                for curr_lex_div in lex_div_combinations:
                    new_row, values_used = fill_template(template_row=curr_row, name1=name1, gs_name1=None, name2=name2, gs_name2=None, names_dict=grouped_names_dict, lex_div_dict=grouped_lex_div_dict, lex_div_assignment=curr_lex_div, stated_gender=stated_gender, df_vocab=df_vocab_cat)

                    if new_row is None:
                        continue

                    # with the texts filled, create all possible instances that use them
                    for unknown_answer_text in unknown_answer_options:
                        new_examples: list[dict] = generate_examples(row=new_row, subcategory=curr_subcategory, unknown_answer_text=unknown_answer_text, bias_targets=bias_targets, name1=name1, name2=name2, name1_info=name1_info, name2_info=name2_info, values_used=values_used)
                        all_generated_examples.extend(new_examples)

                    # unless column `flip_names` is set to False, flip the names in NAME1 and NAME2 to generate the instances with the opposite order
                    if flipping_allowed and not args.minimal:

                        flipped_names_dict = flip_names_dict_keys(grouped_names_dict)

                        # using the same word combination, generate instances with NAME1 and NAME2 flipped
                        for curr_lex_div in lex_div_combinations:
                            new_row, values_used = fill_template(template_row=curr_row, name1=name2, gs_name1=None, name2=name1, gs_name2=None, names_dict=flipped_names_dict, lex_div_dict=grouped_lex_div_dict, lex_div_assignment=curr_lex_div, stated_gender=stated_gender, df_vocab=df_vocab_cat)
                            if new_row is None:
                                continue

                            # create all possible instances with these texts
                            for unknown_answer_text in unknown_answer_options:
                                new_examples: list[dict] = generate_examples(row=new_row, subcategory=curr_subcategory, unknown_answer_text=unknown_answer_text, bias_targets=bias_targets, name1=name1, name2=name2, name1_info=name1_info, name2_info=name2_info, values_used=values_used)
                                all_generated_examples.extend(new_examples)

    print(f"[{curr_category}] Generated {len(all_generated_examples)} sentences total.")

    # if args.variation_stats:
    #     def range_or_num(list_of_vals):
    #         if len(set(list_of_vals)) == 1:
    #             return str(list_of_vals[0])
    #         else:
    #             return f"{min(list_of_vals)}-{max(list_of_vals)}"

    #     _variation_stats.at[curr_category, "num_templates"] = len(df_category)
    #     _variation_stats.at[curr_category, "name1"] = range_or_num(_name1_lengths)
    #     _variation_stats.at[curr_category, "name2"] = range_or_num(_name2_lengths)
    #     _variation_stats.at[curr_category, "unk"] = len(unknown_answer_options)
    #     _variation_stats.at[curr_category, "lex_div"] = range_or_num(_lex_div_lengths)
    #     _variation_stats.at[curr_category, "flipped"] = f"{_flipping_allowed_count}/{len(df_category)}"
    #     _variation_stats.at[curr_category, "total_examples"] = len(all_generated_examples)

    #     _variation_stats = _variation_stats.map(str)

    # now that all the examples are generated, add the sequential IDs to each example dict
    all_generated_examples = [{"example_id": example_id, **example_dict} for example_id, example_dict in enumerate(all_generated_examples)]

    if args.minimal:
        output_fn_prefix = f"data_es/{curr_category}.minimal."
    else:
        output_fn_prefix = f"data_es/{curr_category}."

    # save as JSONL
    if "jsonl" in args.output_formats and not args.dry_run:
        output_fn = output_fn_prefix + "jsonl"

        # convert the examples to JSON strings
        json_lines: list[str] = [json.dumps(example_dict, default=str, ensure_ascii=False) for example_dict in all_generated_examples]

        # print to file
        with open(output_fn, "w+") as output_file:
            print(*json_lines, sep="\n", file=output_file)
            print(f"[{curr_category}] Saved to {output_fn}.")

    # save as CSV
    if "csv" in args.output_formats and not args.dry_run:
        output_fn = output_fn_prefix + "csv"

        # flatten all the example dictionaries because CSV can't handle nested dictionaries
        flattened_examples = [flatten_nested_dicts(example_dict) for example_dict in all_generated_examples]

        # convert example dicts to DataFrame
        df_examples = pd.DataFrame(flattened_examples)

        # save DataFrame to CSV
        df_examples.to_csv(output_fn, index=False)
        print(f"[{curr_category}] Saved to {output_fn}.")

    print()

# if args.variation_stats:
#     print("Variation stats:")
#     print(tabulate(_variation_stats, headers=_variation_stats.columns, tablefmt="psql"))
#     print()
