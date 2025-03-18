import itertools
import re
from collections import OrderedDict
from typing import Optional

import pandas as pd
from nltk.tokenize import sent_tokenize


def flatten(input_list: list) -> list:
    return list(itertools.chain(*input_list))

def split_and_strip(s: str, sep: str) -> list[str]:
    return [item.strip() for item in s.split(sep)]

def remove_brackets(s: str) -> str:
    return re.sub(r"(\{\{?)|(\}\}?)", "", s)

def validate_template(template_row: pd.Series) -> None:
    """
    This function performs some non-exhaustive checks to attempt and identify any errors in the templates that might trickle down and cause confusing errors in the code later.
    """

    # the neg and non-neg answers must use different NAME variables
    assert ("NAME1" in template_row.answer_negative_es and "NAME2" in template_row.answer_non_negative_es) or ("NAME1" in template_row.answer_non_negative_es and "NAME2" in template_row.answer_negative_es), "Same NAME variable used in both answers!"

    # if any context uses WORD variables, the row must include the lexical diversity column
    if "WORD" in template_row.ambiguous_context_es or "WORD" in template_row.disambiguating_context_es:
        assert template_row.get("lexical_diversity_es"), "WORD variable used in context but lexical diversity is empty!"

    # except in Gender, stated gender is required in all the cases that don't use proper names
    if template_row.esbbq_category != 'Gender':
        assert not (template_row.get("proper_nouns_only") == 1 and template_row.get("stated_gender_info") == ""), "No gender info stated!"

def fill_template(
    template_row: pd.Series,
    name1: str,
    gs_name1, # TODO [intersectionals]
    name2: str,
    gs_name2, # TODO [intersectionals]
    names_dict: dict,
    lex_div_dict: dict[str, dict[str, list]],
    lex_div_assignment: dict[str, int],
    stated_gender: str,
    df_vocab: pd.DataFrame,
) -> tuple[pd.Series, dict[str, str]]:
    """
    Process all the text columns in a single template row to substitute the variables for the values given.

    Args:
        template_row (pd.Series): The row from the category DataFrame that contains text to process.
        name1 (str): Value selected for the NAME1 variable.
        gs_name1: TODO
        gs_name2: TODO
        lex_div_dict (dict[str, dict[str, list]]): Lexical diversity, pre-processed from the original DataFrame column into a standard dictionary. 
        lex_div_assignment (dict[str, int]): The assignment of which words from the lexical diversity vocabulary should be used for the current texts.
        stated_gender (str): The pre-processed value from the stated gender column, if available.
        df_vocab (pd.DataFrame): The DataFrame of vocabulary for this category.

    Returns:
        tuple[pd.Series, dict[str, str]]: A pd.Series like the original row but with the processed texts, and a dict that maps all variables that occur in the texts to the values that were used to substitute them.
    """

    # create a copy of the row where the text will be modified
    new_row = template_row.copy()

    # initialize the dictionary that maps all occurring variables to the values used to substitute them
    values_used: dict = {}

    # iterate over all the columns that might have variables to fill
    for curr_text_col in ["ambiguous_context_es", "disambiguating_context_es", "lexical_diversity_es", "question_negative_stereotype_es", "question_non_negative_es", "answer_negative_es", "answer_non_negative_es"]:
        curr_text = template_row.get(curr_text_col, "")
        assert isinstance(curr_text, str)

        new_text: str = curr_text.strip()

        # find all the variables that are used in the text
        variables: list = re.findall(r"\{\{([^\}]+)\}\}", curr_text)

        # iterate over variables and replace each one
        for variable in variables:
            if variable.startswith("NAME1") or variable.startswith("NAME2"):

                selected_name: str = name1 if variable.startswith("NAME1") else name2

                # variable will be replaced by the selected NAME1 or NAME2...
                subst: str = selected_name

                # ...unless there is a specifier, in which case it needs to be replaced by its corresponding -def or -indef
                if "-" in variable:
                    label, specifier = variable.split("-")

                    assert names_dict, f"Variables with specifiers like '{variable}' can only be used with Names but the names column is empty."

                    selected_name_idx: int = names_dict[label][None].index(selected_name)
                    desired_group: list = names_dict[label][specifier]
                    subst = desired_group[selected_name_idx]

                # switch to feminine version if the stated_gender is F and a feminine version is available
                if selected_name in df_vocab.name.tolist() and stated_gender == "f":
                    vocab_row = df_vocab[df_vocab.name == selected_name].iloc[0]
                    subst = vocab_row.get("f") if vocab_row.get("f") else selected_name

            elif variable.startswith("WORD"):
                assert lex_div_dict and lex_div_assignment is not None, f"Text contains '{variable}' but no lexical diversity was found."

                if "-" in variable:
                    label, specifier = variable.split("-")
                else:
                    label, specifier = variable, None

                curr_word_idx: int = lex_div_assignment[label]
                desired_group: list = lex_div_dict[label][specifier]
                subst: str = desired_group[curr_word_idx]

            elif variable in ["GEN1", "GEN2", "OCC1", "OCC2"]:
                # TODO[intersectionals -: this is untested and hasn't been used yet
                assert (variable.endswith("1") and gs_name1 is not None) or (variable.endswith("2") and gs_name2 is not None)
                subst = gs_name1 if variable.endswith("1") else gs_name2

            else:
                raise Exception(f"Unrecognized variable in template with template_id={template_row.esbbq_template_id} column '{curr_text_col}': '{variable}'")

            # once the substitution is defined, replace the variable with the substitution in the text
            new_text = new_text.replace("{{" + variable + "}}", subst).strip()

            # store the value that was used
            values_used[variable] = subst

        # fix capitalization
        new_text = capitalize_sents(new_text)

        # fix error with SpanishRegion group 'País Vasco'
        if "de País Vasco" in new_text:
            new_text = new_text.replace("de País Vasco", "del País Vasco")

        # replace the text with the new one in the new row's corresponding column
        new_row[curr_text_col] = new_text

    return new_row, values_used

def generate_instances(
    row: pd.Series,
    bias_targets: list[str],
    values_used: dict[str, str],
    name1_info: str,
    name2_info: str,
    proper_names_only: bool
) -> tuple[dict]:
    """
    Takes in the pre-processed template row and generates all four possible instances that use this template, by crossing ambiguous and disambiguating contexts and negative and non-negative questions.

    Args:
        row (pd.Series): The template row, with the texts already filled.
        bias_targets (list[str]): The list of bias targets, i.e. stereotyped groups.
        values_used (dict[str, str]): Mapping of all variables in the template to the values that were used to substitute them.
        name1_info (str): Information string referring to NAME1.
        name2_info (str): Information string referring to NAME2.
        stated_gender (str): The pre-processed stated gender info column from the template.

    Returns:
        tuple[dict]: A tuple of four dictionaries corresponding to the four instances.
    """

    # save basic information that will be present in all the instances
    template_id = row.get("esbbq_template_id")
    template_label = row.get("label")
    category = row.get("esbbq_category")
    subcategory = row.get("subcategory", "")
    stated_gender_info = row.get("stated_gender_info", "")
    version = row.get("version", "None")
    source = row.get("esbbq_source", "")
    source = parse_list_from_string(source)

    # contexts
    text_ambig = row.get("ambiguous_context_es")
    text_disambig = row.get("disambiguating_context_es")

    # questions
    q_neg = row.get("question_negative_stereotype_es")
    q_non_neg = row.get("question_non_negative_es")

    # answers
    ans_neg = row.get("answer_negative_es")
    ans_non_neg = row.get("answer_non_negative_es")

    # flipped
    flipped = row.get("flipped")

    # stereotype
    social_value = row.get("relevant_social_values_es")

    """
    NOTE: We have kept this format for compatibility with other BBQ versions but we don't shuffle the answers and we always put the unknown answer as "unknown" in ans2 because it will be handled at evaluation time.
    """
    answer_info = {"ans0": ["", ""], "ans1": ["", ""], "ans2": ["", ""]}

    # retrieve the final version of the names used to fill the templates
    def retrieve_value_without_specifiers(label, values_used):
        label_keys = [key for key in values_used if key.startswith(label)]
        first_key = label_keys[0]
        raw_value = values_used[first_key]
        if first_key.endswith("-def") or first_key.endswith("-indef"):
            return raw_value[raw_value.index(" ")+1:]
        else:
            return raw_value

    name1 = retrieve_value_without_specifiers("NAME1", values_used)
    name2 = retrieve_value_without_specifiers("NAME2", values_used)

    # determined whether the stereotyped group is in NAME1 or NAME2
    # (we need this in order to determine which answer has the stereotyped group
    # and to determine whether the question is pro- or anti-stereotypical)
    if any([info in bias_targets for info in split_and_strip(name1_info, ",")]):
        stereotyped_name = name1
    elif any([info in bias_targets for info in split_and_strip(name2_info, ",")]):
        stereotyped_name = name2
    else:
        raise Exception("Failed to identify stereotyped group in either NAME1 or NAME2.")

    # define whether the stereotyped group is in ans_neg or in ans_non_neg
    # (this is just to make sure that we save the stereotyped group as ans0 and the non-stereotyped as ans1)
    if stereotyped_name.lower() in ans_neg.lower():
        ans_stereotyped = ans_neg
        ans_non_stereotyped = ans_non_neg
        ans_neg_pos, ans_non_neg_pos = 0, 1
    else:
        ans_stereotyped = ans_non_neg
        ans_non_stereotyped = ans_neg
        ans_neg_pos, ans_non_neg_pos = 1, 0

    # the position of the unknown answer will always be the last
    ans_unk_pos = 2

    # set the answer_info values
    if name1.lower() in ans_neg.lower():
        # NAME1 in ans_neg and NAME2 in ans_non_neg
        answer_info[f"ans{ans_neg_pos}"] = [name1, name1_info]
        answer_info[f"ans{ans_non_neg_pos}"] = [name2, name2_info]
    if name1.lower() in ans_non_neg.lower():
        # NAME1 in ans_non_neg and NAME2 in ans_neg
        answer_info[f"ans{ans_non_neg_pos}"] = [name1, name1_info]
        answer_info[f"ans{ans_neg_pos}"] = [name2, name2_info]

    # if name2 in ans_non_neg.lower():
    #     answer_info[f"ans{ans_non_neg_pos}"] = [name2, name2_info]
    # if name2 in ans_neg.lower():
    #     answer_info[f"ans{ans_neg_pos}"] = [name2, name2_info]

    # the unknown answer is always "unknown" and always placed in "ans2"
    answer_info["ans2"] = ["unknown", "unknown"]

    # check that the answer infos have been filled
    assert answer_info["ans0"] != ["", ""] and answer_info["ans1"] != ["", ""]

    # define the base information that is common to all 4 instances
    # (the fields that will vary are set to None so that they can be updated later but keep the ordering)
    base_example_dict = OrderedDict({
        "template_id": template_id,
        "version": version,
        "template_label":template_label,
        "flipped": flipped,
        "question_polarity": None,
        "context_condition": None,
        "category": category,
        "subcategory": subcategory,
        "relevant_social_value": social_value,
        "stereotyped_groups": bias_targets,
        "answer_info": answer_info,
        # "variables": values_used,
        "stated_gender_info": stated_gender_info,
        "proper_nouns_only": proper_names_only,
        "context": None,
        "question": None,
        "ans0": ans_stereotyped,
        "ans1": ans_non_stereotyped,
        "ans2": "unknown",
        "question_type": None,
        "label": None,
        "source": source
    })

    # create instance with negative question and ambiguous context
    neg_ambig_instance = {
        **base_example_dict,
        "question_polarity": "neg",
        "context_condition": "ambig",
        "context": text_ambig,
        "question": q_neg,
        "question_type": "n/a", # ambiguous instance is neither pro- nor anti-stereo
        "label": ans_unk_pos, # answer to ambiguous instance is always unknown
    }

    # create instance with negative question and disambiguating context
    neg_disambig_instance = {
        **base_example_dict,
        "question_polarity": "neg",
        "context_condition": "disambig",
        "context": text_ambig + " " + text_disambig,
        "question": q_neg,
        "question_type": "pro-stereo" if stereotyped_name.lower() in ans_neg.lower() else "anti-stereo",
        "label": ans_neg_pos, # q_neg -> ans_neg
    }

    # create instance with non-negative question and ambiguous context
    non_neg_ambig_instance = {
        **base_example_dict,
        "question_polarity": "nonneg",
        "context_condition": "ambig",
        "context": text_ambig,
        "question": q_non_neg,
        "question_type": "n/a", # ambiguous instance is neither pro- nor anti-stereo
        "label": ans_unk_pos, # answer to ambiguous instance is always unknown
    }

    # create instance with non-negative question and disambiguating instance
    non_neg_disambig_instance = {
        **base_example_dict,
        "question_polarity": "nonneg",
        "context_condition": "disambig",
        "context": text_ambig + " " + text_disambig,
        "question": q_non_neg,  
        "question_type": "anti-stereo" if stereotyped_name.lower() in ans_non_neg.lower() else "pro-stereo",
        "label": ans_non_neg_pos, # q_non_neg -> ans_non_neg
    }

    return (neg_ambig_instance, neg_disambig_instance, non_neg_ambig_instance, non_neg_disambig_instance)

def parse_list_from_string(_string: str) -> list[str]:
    """
    Takes a string that contains a list, either as "['item', 'item', 'item']" (like a stringified Python list) or as comma-separated words ("item, item, item"), and returns the items in an actual Python list.
    """
    _string = _string.strip()

    if not _string:
        return []

    if _string.startswith("[") and _string.endswith("]"):
        if "'" in _string or '"' in _string:
            _list = eval(_string)
            _list = [item.strip() for item in _list]
            return _list
        else:
            _string = _string[1:-1]

    return split_and_strip(_string, ",")

def parse_dict_from_string(raw_string: str) -> dict:
    """
    Takes a string that conveys  a one-to-many mapping of strings, like "NAME1: [nieto]; NAME2: [abuelo]", and parses it as a Python dictionary, e.g. {"NAME1": ["nieto"], "NAME2": ["abuelo"]}.
    """
    if not raw_string:
        return {}

    raw_string = raw_string.replace('"', "")
    key_val_strings = split_and_strip(raw_string, ";")
    key_val_pairs = [split_and_strip(_string, ":") for _string in key_val_strings]
    str_list_pairs = [[remove_brackets(key), parse_list_from_string(val)] for key, val in key_val_pairs]

    return dict(str_list_pairs)

def flip_names(text: str) -> str:
    """
    Flips NAME1 and NAME2 variables in a given text.
    """
    flipped_text = text.replace("NAME1", "TMP").replace("NAME2", "NAME1").replace("TMP", "NAME2")
    return flipped_text

def get_all_permutations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates all possible permutations (4) of each row by flipping NAME1 and NAME2.
    """
    permuted_rows = []
    
    for _, row in df.iterrows():
        # Flip = None
        row['flipped'] = "original"
        permuted_rows.append(row)

        # Flip NAME1/NAME2 only in ambiguous context
        flipped_row_amb = row.copy()
        flipped_row_amb["ambiguous_context_es"] = flip_names(row["ambiguous_context_es"])
        flipped_row_amb['flipped'] = "ambig"
        permuted_rows.append(flipped_row_amb)

        # Flip NAME1/NAME2 in disambiguating context and answers
        flipped_row_disambig = row.copy()
        for col in ["disambiguating_context_es","answer_negative_es","answer_non_negative_es"]:
            flipped_row_disambig[col] = flip_names(row[col])
            flipped_row_disambig['flipped'] = "disambig"
        permuted_rows.append(flipped_row_disambig)

        # Flip NAME1/NAME2 in all columns
        flipped_row_all = row.copy()
        for col in ["ambiguous_context_es","disambiguating_context_es","answer_negative_es","answer_non_negative_es"]:
            flipped_row_all[col] = flip_names(row[col])
            flipped_row_all['flipped'] = "all"
        permuted_rows.append(flipped_row_all)

    # Create a new DataFrame with all permutations
    permuted_df = pd.DataFrame(permuted_rows)
    return permuted_df

def group_by_specifiers(original_dict: dict) -> dict:
    """
    Takes a dict like {"WORD1": [], "WORD1-indef": [], "WORD2": [], "WORD2-def": [], ...} and returns a dict where the keys are only "WORD1", "WORD2" etc. without specifiers and each key is another dict where the keys are "def", "indef" and None (meaning raw form without articles).
    """
    expanded_dict = {}
    for key, word_list in original_dict.items():
        new_key, specifier = key.split("-") if "-" in key else (key, None)

        if new_key not in expanded_dict:
            expanded_dict[new_key] = {}

        expanded_dict[new_key][specifier] = word_list

    for key, sub_dict in expanded_dict.items():
        if set(sub_dict.keys()) == {"indef", "def"}:
            sub_dict[None] = [np[np.index(" ")+1:] for np in sub_dict.get("indef", sub_dict.get("def"))]

    # for each key, make sure that all the lists have the same length
    for label in expanded_dict:
        lengths = [len(value_list) for _, value_list in expanded_dict[label].items()]
        assert len(set(lengths)) == 1, f"Lists for {label} don't match: {expanded_dict[label]}"

    return expanded_dict

def capitalize_sents(text: str) -> str:
    """
    Capitalizes the first letter of each sentence after splitting the text into sentences with the NLTK sentence tokenizer.
    """
    sents = sent_tokenize(text, language="spanish")
    new_sents = []

    for sent in sents:
        if sent[0] in ["¿", "¡"]:
            # if the sentence starts with these marks we capitalize the second character
            sent = sent[0] + sent[1].upper() + sent[2:]
        else:
            sent = sent[0].upper() + sent[1:]

        new_sents.append(sent)

    return " ".join(new_sents)

def get_lex_div_combinations(grouped_dict: dict) -> list[dict[str, int]]:
    """
    TODO[docs]
    """

    labels = list(grouped_dict.keys())
    lengths = [len(grouped_dict[label][list(grouped_dict[label].keys())[0]]) for label in labels]

    index_product = list(itertools.product(*[range(length) for length in lengths]))

    selections = [{label : idx for label, idx in zip(labels, selection)} for selection in index_product]
    return selections

def flatten_nested_dicts(original_dict: dict) -> dict:
    """
    Takes a dictionary that might have sub-dictionaries and flattens them all recursively into first-level key-value pairs with dot-separated nested keys.
    e.g. {"a": {1: {i: 0, ii: 1}}} => {"a.1.i": 0, "a.1.ii": 1}
    """
    new_dict = {}
    for key, value in original_dict.items():
        if isinstance(value, dict):
            flattened_dict = flatten_nested_dicts(value) # recursive call

            for nested_key, nested_value in flattened_dict.items():
                new_dict[f"{key}.{nested_key}"] = nested_value
        else:
            new_dict[key] = value

    return new_dict
