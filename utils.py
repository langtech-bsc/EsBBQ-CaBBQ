import itertools
import random
import re
from collections import OrderedDict

import pandas as pd
from nltk.tokenize import sent_tokenize

random.seed(1)

def flatten(input_list: list) -> list:
    return list(itertools.chain(*input_list))

def split_and_strip(s: str, sep: str) -> list[str]:
    return [item.strip() for item in s.split(sep)]

def remove_brackets(s: str) -> str:
    return re.sub(r"(\{\{?)|(\}\}?)", "", s)

def fill_template(
    template_row: pd.Series,
    name1: str,
    gs_name1, # TODO[intersectionals]
    name2: str,
    gs_name2, # TODO[intersectionals]
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
                if selected_name in df_vocab.Name_es.tolist() and stated_gender == "f":
                    vocab_row = df_vocab[df_vocab.Name_es == selected_name].iloc[0]
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
                raise Exception(f"Unrecognized variable in template with Q_id={template_row.Q_id} column '{curr_text_col}': '{variable}'")

            # once the substitution is defined, replace the variable with the substitution in the text
            new_text = new_text.replace("{{" + variable + "}}", subst).strip()

            # store the value that was used
            values_used[variable] = subst

        # fix capitalization
        new_text = capitalize_sents(new_text)

        # replace the text with the new one in the new row's corresponding column
        new_row[curr_text_col] = new_text

    return new_row, values_used

def generate_examples(
    subcategory: str,
    row: pd.Series,
    unknown_answer_text: str,
    bias_targets: list[str],
    name1: str,
    name2: str,
    name1_info: str,
    name2_info: str,
    values_used: dict[str, str],
) -> tuple[dict]:
    """
    Takes in the pre-processed template row and generates all four possible instances that use this template, by crossing ambiguous and disambiguating contexts and negative and non-negative questions.

    Args:
        subcategory (str): The current subcategory.
        row (pd.Series): The template row, with the texts already filled.
        unknown_answer_text (str): The selected text for the unknown answer.
        bias_targets (list[str]): The list of bias targets, i.e. stereotyped groups.
        name1 (str): Value selected for the NAME1 variable.
        name2 (str): Value selected for the NAME2 variable.
        name1_info (str): Information string referring to NAME1.
        name2_info (str): Information string referring to NAME2.
        values_used (dict[str, str]): Mapping of all variables in the template to the values that were used to substitute them.
        stated_gender (str): The pre-processed stated gender info column from the template.

    Returns:
        tuple[dict]: A tuple of four dictionaries corresponding to the four instances.
    """

    q_id = row.Q_id
    category = row.get("final_category") if row.get("final_category") else row.get("Category")
    text_ambig = row.ambiguous_context_es.strip()
    text_disambig = row.disambiguating_context_es.strip()
    q_neg = row.question_negative_stereotype_es.strip()
    q_non_neg = row.question_non_negative_es.strip()
    ans_neg = row.answer_negative_es.strip()
    ans_non_neg = row.answer_non_negative_es.strip()
    ans_ambig = unknown_answer_text.strip()
    stated_gender_info = row.get("Stated_gender_info", "")
    version = row.get("version", "None")
    source = row.get("source_es", "")

    # randomize the order of the answers
    answer_list = [ans_neg, ans_non_neg, ans_ambig]
    random.shuffle(answer_list)

    # need to still get the position of each answer after randomization to know correct answer label
    ans_neg_pos = answer_list.index(ans_neg)
    ans_non_neg_pos = answer_list.index(ans_non_neg)
    ans_unk_pos = answer_list.index(ans_ambig)

    answer_info = {"ans0": ["", ""], "ans1": ["", ""], "ans2": ["", ""]}

    if name1 in ans_neg:
        ans_neg_key = list(answer_info.keys())[ans_neg_pos]
        answer_info[ans_neg_key] = [name1, name1_info]

    if name1 in ans_non_neg:
        ans_non_neg_key = list(answer_info.keys())[ans_non_neg_pos]
        answer_info[ans_non_neg_key] = [name1, name1_info]

    if name2 in ans_neg:
        ans_neg_key = list(answer_info.keys())[ans_neg_pos]
        answer_info[ans_neg_key] = [name2, name2_info]

    if name2 in ans_non_neg:
        ans_non_neg_key = list(answer_info.keys())[ans_non_neg_pos]
        answer_info[ans_non_neg_key] = [name2, name2_info]

    ans_unknown_key = list(answer_info.keys())[ans_unk_pos]
    answer_info[ans_unknown_key] = [ans_ambig, "unknown"]

    # define whether the stereotyped group is in NAME1 or NAME2
    if any([info in bias_targets for info in split_and_strip(name1_info, ",")]):
        stereotyped_name = name1
    elif any([info in bias_targets for info in split_and_strip(name2_info, ",")]):
        stereotyped_name = name2
    else:
        breakpoint()
        # TODO
    
    # define the base information that is common to all 4 instances
    base_example_dict = OrderedDict({
        "question_index": q_id,
        "version": version,
        "question_polarity": None,
        "context_condition": None,
        "category": category,
        "answer_info": answer_info,
        "subcategory": subcategory,
        "stereotyped_groups": bias_targets,
        "variables": values_used,
        "stated_gender_info": stated_gender_info,
        "context": None,
        "question": None,
        "ans0": answer_list[0],
        "ans1": answer_list[1],
        "ans2": answer_list[2],
        "question_type": None,
        "label": None,
        "source": source,
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
        "question_type": "pro-stereo" if stereotyped_name in ans_neg else "anti-stereo",
        "label": ans_neg_pos,
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
        "question_type": "anti-stereo" if stereotyped_name in ans_non_neg else "pro-stereo",
        "label": ans_non_neg_pos,
    }

    return (neg_ambig_instance, neg_disambig_instance, non_neg_ambig_instance, non_neg_disambig_instance)

def parse_list_from_string(s: str) -> list[str]:
    if not s:
        return []

    if s.startswith("[") and s.endswith("]"):
        return eval(s)

    else:
        return split_and_strip(s, ",")

def parse_dict_from_string(s) -> dict:
    """
    TODO[docs]
    """
    if not s:
        return {}

    s = s.replace('"', "")
    # e.g. "NAME1-indef: [un nieto, una nieta]; NAME1-def: [el nieto, la nieta]; NAME2: [abuelo, abuela]; NAME2-def: [el abuelo, la abuela]"
    key_val_pairs: list = split_and_strip(s, ";")
    key_val_pairs: list = [split_and_strip(pair, ":") for pair in key_val_pairs]
    str_list_pairs: list = [[remove_brackets(key), split_and_strip(val.strip()[1:-1], ",")] for key, val in key_val_pairs]

    return dict(str_list_pairs)

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

    Args:
        text (str): input text

    Returns:
        str: processed text
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

def flip_names_dict_keys(names_dict: dict) -> dict:
    """
    This function receives a names dict like {"NAME1": [...], "NAME2": [...]} and flips the keys so that the value of NAME1 is moved to NAME2 and vice-versa.
    """

    flipped_names_dict = {}
    for orig_key, orig_value in names_dict.items():
        if orig_key.startswith("NAME1"):
            new_key = orig_key.replace("NAME1", "NAME2")
        elif orig_key.startswith("NAME2"):
            new_key = orig_key.replace("NAME2", "NAME1")

        flipped_names_dict[new_key] = orig_value

    return flipped_names_dict

def flatten_nested_dicts(original_dict: dict) -> dict:
    """
    Takes a dictionary that might have sub-dictionaries and flattens them all recursively into first-level key-value pairs with dot-separated nested keys.
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
