import argparse
import json
import os
import re

import pandas as pd
from tabulate import tabulate

from utils import (
    fill_template,
    flatten_nested_dicts,
    generate_instances,
    get_all_permutations,
    get_lex_div_combinations,
    group_by_specifiers,
    parse_dict_from_string,
    parse_list_from_string,
    validate_template
)

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
all_categories = sorted([re.match(r"esbbq_(\w+).xlsx", fn).group(1) for fn in os.listdir("templates_es") if fn.startswith("esbbq_")])

# create parser for the CLI arguments
parser = argparse.ArgumentParser(prog="Generate BBQ Instances", description="This script will read the Excel files in the input folder and generate BBQ instances for all the categories. By default, generates instances from all the templates in all the categories.")
parser.add_argument("--categories", nargs="+", choices=all_categories, default=all_categories, help="Space-separated list of categories to process templates and generate instances. If not passed, will run for all available categories.")
parser.add_argument("--minimal", action="store_true", help="Minimize the sources of variation in instances by only taking one option from each source of variation.")
parser.add_argument("--output-formats", nargs="+", choices=output_format_choices, default=output_format_choices, help="Space-separated format(s) in which to save the instances.")
parser.add_argument("--dry-run", action="store_true", help="Generate the templates and print the logs and stats but don't actually save them to file.")
parser.add_argument("--no-proper-names", action="store_true", help="Ignore the templates that require proper names in all categories contemplated.")
parser.add_argument("--save-fertility", action="store_true", help="Save an extra CSV with the fertility (instance count) of each template.")

args = parser.parse_args()

# read and pre-process vocabulary files
df_vocab = pd.read_excel("templates_es/vocabulary_es.xlsx", sheet_name="vocabulary_es").fillna("")
df_proper_names = pd.read_excel("templates_es/vocabulary_proper_names_es.xlsx", sheet_name="vocabulary_proper_names_es").fillna("")

df_vocab = df_vocab[df_vocab.include_name == ""] # only keep the rows where include_name is empty (not FALSE)

df_vocab = df_vocab[["category", "subcategory", "name", "f", "information"]].map(str.strip)
df_proper_names = df_proper_names[["proper_name", "gender", "ethnicity"]].map(str.strip)

# initialize DF for the statistics per category
df_stats = pd.DataFrame(index=args.categories)

# iterate over categories to read all the templates and fill them in
for curr_category in args.categories:

    # read the category's Excel spreadsheet of templates
    df_category = pd.read_excel(f"templates_es/esbbq_{curr_category}.xlsx", sheet_name="Sheet1", na_filter=False).fillna("")
    
    print(f"[{curr_category}] Imported {len(df_category)} templates.")

    # save the number of templates in stats
    df_stats.at[curr_category, "num_templates"] = len(set(df_category.esbbq_template_id))
    df_stats.at[curr_category, "num_rows"] = len(df_category)

    # generate all possible templates with permutations of NAME1 and NAME2
    if not args.minimal:
        df_category = get_all_permutations(df_category)

    # initialize list that will contain all the instances generated for all the templates in this category
    generated_instances: list[dict] = []

    # iterate over template rows to generate instances for one template at a time
    for _, curr_row in df_category.iterrows():
        """
        ROW CONFIG
        """

        validate_template(curr_row)

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

        # count the instances generated for this specific template, for fertility stats
        instance_count = 0

        # pre-process the column of stated gender in the cases where the template should be used only for one gender
        stated_gender: str = curr_row.get("stated_gender_info", "").lower()
        if "fake-" in stated_gender:
            # treat the case of "fake-m" and "fake-f" (used to differentiate from "m/f" when it's only grammatical gender)
            stated_gender = stated_gender.split("-")[1]

        assert stated_gender in ["m", "f", ""], f"Invalid value for stated_gender_info: `{stated_gender}`"

        # determine whether NAME1 and NAME2 need to be proper names
        proper_names_only: bool = bool(curr_row.get("proper_nouns_only", ""))

        if proper_names_only and args.no_proper_names:
            # if the option to ignore proper names was passed and the current template is for proper names, skip it
            continue

        """
        NAME1 AND NAME2 VALUES
        """

        # select the words from the vocab that match the current category
        df_vocab_cat = df_vocab[(df_vocab.category == curr_category) & (df_vocab.name != "-")]

        # filter by subcategory if there is one
        curr_subcategory = curr_row.get("subcategory")
        if curr_subcategory:
                # filter the vocabulary for the given subcategory only
                df_vocab_cat = df_vocab_cat[df_vocab_cat.subcategory == curr_subcategory]

        # parse the list of stereotyped groups (i.e. bias targets) that the current template refers to
        bias_targets: str = parse_list_from_string(curr_row.stereotyped_groups)

        if proper_names_only:

            # for RaceEthnicity, generate NAME1 options using names associated with the targeted ethnicities
            if curr_category == "RaceEthnicity":
                assert bias_targets, "Category is RaceEthnicity but the template doesn't have any specified stereotyped groups!"

                df_names = df_proper_names[df_proper_names.ethnicity.isin(bias_targets)]

                if stated_gender:
                    # if the template states a specific gender to use, restrict to this gender or to genderless names
                    df_names = df_names[df_names.gender.isin([stated_gender, ""])]
                
                # iterate over names to save them in the NAME1 list and store info
                for _, row in df_names.iterrows():
                        name = row.proper_name
                        name_info = f"{row.gender}, {row.ethnicity}"
                        name1_list.append(name)
                        name1_info_dict[name] = name_info

            # for Gender with proper names, NAME1 is always female and NAME2 is always male
            elif curr_category == "Gender":
                # we use only white names to avoid introducing additional bias by comparing between stereotyped ethnicities
                df_names = df_proper_names[df_proper_names.ethnicity == "blanco"]
                assert len(df_names)

                # force NAME1 to be female
                name1_list = df_names[df_names.gender == "f"].proper_name.unique().tolist()
                name1_info_dict.update({name : "f" for name in name1_list})

                # force NAME2 to be male
                name2_list = df_names[df_names.gender == "m"].proper_name.unique().tolist()
                name2_info_dict.update({name : "m" for name in name2_list})

            # In the case of other categories that use proper nouns, they can be of any gender but we don't want them to elicit any specific ethnicities so we take only typical white Spanish names
            else:
                df_names = df_proper_names[df_proper_names.ethnicity == "blanco"]
                df_names = df_names[df_names.gender.isin([stated_gender, ""])]

                name1_list = df_names.proper_name.tolist()

        else: # if not using proper names

            if curr_category == "SES" and curr_subcategory == "Occupation":
                # for SES Occupations, options for NAME1 are all the occupations listed in the vocabulary
                name1_list = df_vocab_cat.name.unique().tolist()

            elif curr_category != "Gender":
            # options for NAME1 are simply the bias targets
                name1_list = bias_targets

            if not name2_list:
                curr_non_stereotyped = curr_row.get("non_stereotyped_groups")
                if curr_non_stereotyped:
                    name2_list = parse_list_from_string(curr_non_stereotyped)
                else:
                    # initialize the options for NAME2 as the groups marked in the vocabulary as non-stereotyped, if any
                    df_vocab_non_stereotyped = df_vocab_cat[df_vocab_cat.information == "not-stereotyped"]
                    name2_list = df_vocab_non_stereotyped.name.unique().tolist()

            # get specific vocabulary from the names column if available
            # (values in the names column always override vocabulary that would otherwise be used,
            # and they can only be used when the template is not for proper names)
            names_str = curr_row.get("names_es", "")
            if names_str:
                names_dict = parse_dict_from_string(names_str)
                grouped_names_dict = group_by_specifiers(names_dict)

                name1_list = grouped_names_dict["NAME1"][None]
                name2_list = grouped_names_dict["NAME2"][None]

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

        """
        NAME1 LOOP
        Iterate over the list of possible values for NAME1
        """

        for name1 in name1_list:

            # set info field
            if name1 in name1_info_dict:
                name1_info = name1_info_dict[name1]
            else:
                name1_info = curr_row.get("NAME1_info")

            # set possible values for NAME2 here when they depend on NAME1
            # (in which case they could not be determined before the NAME1 loop)
            if curr_category == "SES":
                if proper_names_only:
                    # for SES with proper names, take NAME2 from the remaining white names different from NAME1
                    df_other_names = df_proper_names[(df_proper_names.ethnicity == "blanco") & (df_proper_names.proper_name != name1)]

                    if stated_gender:
                        # if the template states a specific gender to use, restrict to this gender or to genderless names
                        df_other_names = df_other_names[df_other_names.gender.isin([stated_gender, ""])]

                    name2_list = df_other_names.proper_name.tolist()

                elif curr_subcategory == "Occupation":
                # for SES Occupations, NAME1 options can be either highSES or lowSES, so we ensure that NAME2 options are always in the opposite category
                    name1_info = df_vocab_cat.loc[df_vocab.name == name1, "information"].iloc[0]
                    name2_list = df_vocab_cat[df_vocab_cat.information != name1_info].name.unique().tolist()

            # for race/ethnicity, NAME2 options should be the same gender as NAME1 but with the non-stereotyped ethnicity
            elif curr_category == "RaceEthnicity" and proper_names_only:

                # find the non-stereotyped ethnicity
                non_stereotyped_groups = df_vocab[df_vocab.information == "not-stereotyped"].name.unique().tolist()

                # filter the proper names to those of the non-stereotyped ethnicity and same gender as NAME1
                gender = name1_info.split(",")[0]
                df_other_names = df_proper_names[(df_proper_names.ethnicity.isin(non_stereotyped_groups)) & (df_proper_names.gender == gender)]

                # iterate over names to add them to the NAME2 list and save info
                for _, name_row in df_other_names.iterrows():
                    name = name_row.proper_name
                    name2_list.append(name)
                    name2_info_dict[name] = name_row.ethnicity                        

            assert name2_list, "No names in the list of options for NAME2!"

            if args.minimal:
                name2_list = name2_list[:1]

            """
            NAME2 LOOP
            Iterate over all the names in the NAME2 list
            """
            for name2 in name2_list:
                if name2 in name2_info_dict:
                    name2_info = name2_info_dict[name2]
                else:
                    name2_info = curr_row.get("NAME2_info")

                if curr_category == "RaceEthnicity" and proper_names_only:
                    name2_info = name2_info_dict[name2]

                elif curr_category == "Gender" and proper_names_only:
                    # if there is already some info on the row, append it to the existing information
                    if curr_row.get("NAME1_info"):
                        name1_info = f"{name1_info}, {curr_row.get('NAME1_info')}"
                    if curr_row.get("NAME2_info"):
                        name2_info = f"{name2_info}, {curr_row.get('NAME2_info')}"

                elif curr_category == "SES" and curr_subcategory == "Occupation":
                    # get the info about each name from the vocab
                    name1_vocab_row = df_vocab.loc[df_vocab.name == name1].iloc[0]
                    name1_info = name1_vocab_row.get("information")
                    name2_vocab_row = df_vocab.loc[df_vocab.name == name2].iloc[0]
                    name2_info = name2_vocab_row.get("information")

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
                    new_instances: list[dict] = generate_instances(row=new_row, bias_targets=bias_targets, values_used=values_used, name1_info=name1_info, name2_info=name2_info, proper_names_only=proper_names_only)
                    generated_instances.extend(new_instances)

    assert len(generated_instances), f"No instances generated for {curr_category}!"

    # Generating all possible permutations of NAME1 and NAME2 creates duplicates. Deduplicate df before saving it
    if not args.minimal:
        # Set to track the unique combinations of relevant columns
        seen = set()
        deduplicated = []
        for instance in generated_instances:
            identifier = (instance["template_id"], instance["context"], instance["question"])
            if identifier not in seen:
                deduplicated.append(instance)
                seen.add(identifier)
        generated_instances = deduplicated
    
    print(f"[{curr_category}] Generated {len(generated_instances)} sentences total.")

    # now that all the instances are generated, add sequential IDs to each instance dict
    generated_instances = [{"instance_id": instance_id, **instance_dict} for instance_id, instance_dict in enumerate(generated_instances)]

    # calculate the fertility of each template (indexed by template_id and version) by counting the unique instance IDs
    df_category_fertility = pd.DataFrame(generated_instances).groupby(["template_id", "version"])["instance_id"].count().reset_index().rename(columns={"instance_id": "instances"})

    if args.save_fertility:
        # save the fertility dict to a CSV under stats/template_fertility
        if not os.path.exists("stats/template_fertility"):
            os.makedirs("stats/template_fertility")
        fertility_fn = f"stats/template_fertility/{curr_category}.fertility.csv"
        df_category_fertility.to_csv(fertility_fn, index=False)
        print(f"[{curr_category}] Fertility saved to `{fertility_fn}`.")

    # save stats
    df_stats.at[curr_category, "total_instances"] = len(generated_instances)
    df_stats.at[curr_category, "avg_fertility"] = round(df_category_fertility.instances.mean())

    if args.minimal:
        output_fn_prefix = f"data_es/{curr_category}.minimal."
    else:
        output_fn_prefix = f"data_es/{curr_category}.full."

    # save as JSONL
    if "jsonl" in args.output_formats and not args.dry_run:
        output_fn = output_fn_prefix + "jsonl"

        # convert the examples to JSON strings
        json_lines: list[str] = [json.dumps(example_dict, default=str, ensure_ascii=False) for example_dict in generated_instances]

        # print to file
        with open(output_fn, "w+") as output_file:
            print(*json_lines, sep="\n", file=output_file)
            print(f"[{curr_category}] Instances saved to `{output_fn}`.")

    # save as CSV
    if "csv" in args.output_formats and not args.dry_run:
        output_fn = output_fn_prefix + "csv"

        # flatten all the instances because CSV can't handle nested dicts
        flattened_examples = [flatten_nested_dicts(example_dict) for example_dict in generated_instances]

        # convert instance dicts to DataFrame and save to CSV
        df_instances = pd.DataFrame(flattened_examples)
        df_instances.to_csv(output_fn, index=False)
        print(f"[{curr_category}] Instances saved to `{output_fn}`.")

    print()

print("Summary:")
print("Be careful! Avg fertility is computed among all instances, not per template.")
print(tabulate(df_stats.reset_index(), headers=["category", *df_stats.columns], tablefmt="psql"))
