import os
import random
import re
import itertools
from collections import OrderedDict, defaultdict

import pandas as pd
from nltk.tokenize import sent_tokenize

# def return_list_from_string(inputx):
#     """
#     This function takes a string that's formatted as
#     WORD1: [word, word]; WORD2: [word, word]
#     and separates it into two iterable lists

#     Args:
#         input: string

#     Returns:
#         two lists
#     """

#     x = inputx.split(";")
#     for wrd in x:
#         if "WORD1" in wrd or "NAME1" in wrd:
#             wrd2 = (
#                 wrd.replace("WORD1:", "")
#                 .replace("{{WORD1}}:", "")
#                 .replace("NAME1:", "")
#                 .replace("{{NAME1}}:", "")
#             )
#             wrd3 = wrd2.strip()
#             wrds = wrd3.replace("[", "").replace("]", "")
#             wrds1 = wrds.split(",")
#             wrds1 = [w.strip() for w in wrds1]
#         if "WORD2" in wrd or "NAME2" in wrd:
#             wrd2 = (
#                 wrd.replace("WORD2:", "")
#                 .replace("{{WORD2}}:", "")
#                 .replace("NAME2:", "")
#                 .replace("{{NAME2}}:", "")
#             )
#             wrd3 = wrd2.strip()
#             wrds = wrd3.replace("[", "").replace("]", "")
#             wrds2 = wrds.split(",")
#             wrds2 = [w.strip() for w in wrds2]
#         else:
#             wrds2 = ""
#     return wrds1, wrds2


def do_slotting(
    template_row: pd.Series,
    name1: str,
    gs_name1, # TODO (intersectionals)
    name2: str,
    gs_name2, # TODO (intersectionals)
    grouped_names_dict: dict,
    grouped_words_dict: dict[str, dict[str, list]],
    word_combination: dict[str, int],
):
    # create a copy of the row where the text will be modified
    new_row = template_row.copy()

    # iterate over all the columns that might have variables to fill
    for curr_text_col in ["ambiguous_context_es", "disambiguating_context_es", "lexical_diversity_es", "question_negative_stereotype_es", "question_non_negative_es", "answer_negative_es", "answer_non_negative_es"]:
        curr_text = template_row.get(curr_text_col, "")
        assert isinstance(curr_text, str)

        new_text: str = curr_text

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
                    selected_name_idx: int = grouped_names_dict[label][None].index(selected_name)
                    desired_group: list = grouped_names_dict[label][specifier]
                    subst = desired_group[selected_name_idx]

            elif variable.startswith("WORD"):
                if "-" in variable:
                    label, specifier = variable.split("-")
                else:
                    label, specifier = variable, None

                curr_word_idx: int = word_combination[label]
                desired_group: list = grouped_words_dict[label][specifier]
                subst: str = desired_group[curr_word_idx]

            elif variable in ["GEN1", "GEN2", "OCC1", "OCC2"]:
                assert (variable.endswith("1") and gs_name1 is not None) or (variable.endswith("2") and gs_name2 is not None)
                subst = gs_name1 if variable.endswith("1") else gs_name2

            else:
                raise Exception(f"Unrecognized variable in template with Q_id={template_row.Q_id} column '{curr_text_col}': '{variable}'")

            # once the substitution is defined, replace the variable with the substitution in the text
            new_text = new_text.replace("{{" + variable + "}}", subst)

            # fix capitalization
            curr_text = capitalize_sents(curr_text)

            # replace the text with the new one in the new row's corresponding column
            new_row[curr_text_col] = new_text

    return new_row

        # name1_final = name1
        # name2_final = name2
        # word1_final = rand_wrd1
        # word2_final = rand_wrd2

        # if names_dict:
        #     # swap out the selected word for its respective definite or indefinite noun phrase according to what is in the template
        #     for specifier in ["def", "indef"]:
        #         if f"NAME1-{specifier}" in curr_text:
        #             for key in names_dict:
        #                 if key.startswith("NAME1") and name1 in names_dict[key]:
        #                     this_word_pos = names_dict[key].index(name1)
        #                     new_word = names_dict[f"NAME1-{specifier}"][this_word_pos]
        #                     name1_final = new_word
        #                     break
        #         if f"NAME2-{specifier}" in curr_text:
        #             for key in names_dict:
        #                 if key.startswith("NAME2") and name2 in names_dict[key]:
        #                     this_word_2_pos = names_dict[key].index(name2)
        #                     new_word_2 = names_dict[f"NAME2-{specifier}"][this_word_2_pos]
        #                     name2_final = new_word_2
        #                     break

        # if words_dict:
        #     for specifier in ["def", "indef"]:
        #         if f"WORD1-{specifier}" in curr_text:
        #             for key in words_dict:
        #                 if key.startswith("WORD1") and rand_wrd1 in words_dict[key]:
        #                     rand_wrd1_pos = words_dict[key].index(rand_wrd1)
        #                     new_rand_wrd1 = words_dict[f"WORD1-{specifier}"][rand_wrd1_pos]
        #                     word1_final = new_rand_wrd1
        #                     break

        #         if f"WORD2-{specifier}" in curr_text:
        #             for key in words_dict:
        #                 if key.startswith("WORD2") and rand_wrd2 in words_dict[key]:
        #                     rand_wrd2_pos = words_dict[key].index(rand_wrd2)
        #                     new_rand_wrd2 = words_dict[f"WORD2-{specifier}"][rand_wrd2_pos]
        #                     word2_final = new_rand_wrd2
        #                     break

        # if vocab_dict and not names_dict:
        #     for specifier in ["def", "indef"]:
        #         if f"NAME1-{specifier}" in curr_text:
        #             this_word_pos = vocab_dict[None].index(name1)
        #             new_word = vocab_dict[specifier][this_word_pos]
        #             name1_final = new_word
        #         if f"NAME2-{specifier}" in curr_text:
        #             this_word_2_pos = vocab_dict[None].index(name2)
        #             new_word_2 = vocab_dict[specifier][this_word_2_pos]
        #             name2_final = new_word_2

            # # skip some cases that don't make sense with the vocabulary in the Sexual Orientation category
            # if (
            #     ("NAME1-woman" in curr_text and name1_final == "gay") or
            #     ("NAME2-man" in curr_text and name2_final == "lesbiana") or
            #     ({name1_final, name2_final}.issubset({"gay", "lesbiana", "homosexual"}))
            # ):
            #     return None


            # curr_text = re.sub(r"\{\{NAME1-?[\w]*\}\}", name1_final, curr_text)
            # curr_text = re.sub(r"\{\{NAME2-?[\w]*\}\}", name2_final, curr_text)


        


def make_dict(
    nn,
    q_id,
    polarity,
    context_cond,
    cat,
    subcat,
    answer_info,
    bias_targets,
    version,
    notes,
    context,
    question,
    ans_list,
    ans_place,
):
    """
    Formats information into a standardized dict that can be saved to jsonl

    Args:
        nn: item number created for output
        q_id: item number from the template file
        polarity: neg/nonneg indicating what type of question is used
        context_cond: ambig/disambig indicating the type of context used
        cat: category of bias
        subcat: subcategory of bias, if defined
        answer_info: information about the answer items, provided as a list of three lists
        bias_targets: list of bias targets for that template
        version: version info from the template file
        notes: additional info provided in the template file
        context: string, full context provided for model input
        question: string, question provided for model input
        ans_list: list of answer options
        ans_place: label corresponding to which answer option is correct

    Returns:
        dictionary
    """
    this_dict = {
        "example_id": nn,
        "question_index": q_id,
        "question_polarity": polarity,
        "context_condition": context_cond,
        "category": cat,
        "answer_info": answer_info,
        "additional_metadata": {
            "subcategory": subcat,
            "stereotyped_groups": bias_targets,
            "version": version,
            "source": notes,
        },
        "context": context.strip(),
        "question": question.strip(),
        "ans0": ans_list[0],
        "ans1": ans_list[1],
        "ans2": ans_list[2],
        "label": ans_place,
    }
    return this_dict


def create_templating_dicts(
    cat,
    this_frame_row,
    subcat,
    unknown_options,
    frame_cols,
    bias_targets,
    name1,
    name2,
    name1_info,
    name2_info,
    nn,
):
    """
    takes in all the information needed to create a single set of four items for a given instance,
    calls make_dict() for formatting, and returns the four dicts

    Args:
        cat: bias category
        this_frame_row: the frame row with slots already filled
        subcat: the subcategory of bias, if provided
        unknown_options: the string meaning 'unknown'
        frame_cols: names of the dataframe columns
        bias_targets: list of bias targets associated with this example
        name1: list or string of first name variable
        name2: list or string of second name variable
        name1_info: list or string, category info associated with first name variable
        name2_info: list or string, category info associated with first name variable
        nn: item number created for output

    Returns:
        list of four dicts created by crossing context condition x polarity for this example set
    """
    # get all the info for this frame, should all be formatted now
    q_id = this_frame_row.Q_id[0]
    text_ambig = this_frame_row.ambiguous_context_es[0]
    text_disambig = this_frame_row.disambiguating_context_es[0]
    q_neg = this_frame_row.question_negative_stereotype_es[0]
    q_non_neg = this_frame_row.question_non_negative_es[0]
    ans_neg = this_frame_row.answer_negative_es[0]
    ans_non_neg = this_frame_row.answer_non_negative_es[0]
    ans_ambig = random.choice(unknown_options)
    if "version" in frame_cols:
        version = this_frame_row.version[0]
    else:
        version = "None"

    notes = this_frame_row.Notes[0]

    # for answer option order randomization
    ans_list = [ans_neg, ans_non_neg, ans_ambig]
    random.shuffle(ans_list)
    # need to still get the position of each answer after randomization to know correct answer label
    ans_neg_place = ans_list.index(ans_neg)
    ans_non_neg_place = ans_list.index(ans_non_neg)
    ans_unknown_place = ans_list.index(ans_ambig)

    answer_info = {"ans0": ["", ""], "ans1": ["", ""], "ans2": ["", ""]}

    # need to pull the answer info differently depending on whether the strings are part of a list or not
    if isinstance(name1, list):
        # this is annoying, but when one of the names is 'man', it returns true for 'woman'
        # so just add a space before it, so that the conditionals below won't trigger at 'woman'
        if name1[1] == "man":
            name1[1] = " man"
        if name2[1] == "man":
            name2[1] = " man"
        if name1[0] in ans_neg and name1[1] in ans_neg:
            ans_neg_key = list(answer_info.keys())[ans_neg_place]
            answer_info[ans_neg_key] = [name1[0] + " " + name1[1].lstrip(), name1_info]
        if name1[0] in ans_non_neg and name1[1] in ans_non_neg:
            ans_non_neg_key = list(answer_info.keys())[ans_non_neg_place]
            answer_info[ans_non_neg_key] = [
                name1[0] + " " + name1[1].lstrip(),
                name1_info,
            ]
        if name2[0] in ans_neg and name2[1] in ans_neg:
            ans_neg_key = list(answer_info.keys())[ans_neg_place]
            answer_info[ans_neg_key] = [name2[0] + " " + name2[1].lstrip(), name2_info]
        if name2[0] in ans_non_neg and name2[1] in ans_non_neg:
            ans_non_neg_key = list(answer_info.keys())[ans_non_neg_place]
            answer_info[ans_non_neg_key] = [
                name2[0] + " " + name2[1].lstrip(),
                name2_info,
            ]
        ans_unknown_key = list(answer_info.keys())[ans_unknown_place]
        answer_info[ans_unknown_key] = [ans_ambig, "unknown"]
    else:
        # same issue noted above
        if name1 == "man":
            name1 = " man"
        if name2 == "man":
            name2 = " man"
        if name1 in ans_neg:
            ans_neg_key = list(answer_info.keys())[ans_neg_place]
            answer_info[ans_neg_key] = [name1.lstrip(), name1_info]
        if name1 in ans_non_neg:
            ans_non_neg_key = list(answer_info.keys())[ans_non_neg_place]
            answer_info[ans_non_neg_key] = [name1.lstrip(), name1_info]
        if name2 in ans_neg:
            ans_neg_key = list(answer_info.keys())[ans_neg_place]
            answer_info[ans_neg_key] = [name2.lstrip(), name2_info]
        if name2 in ans_non_neg:
            ans_non_neg_key = list(answer_info.keys())[ans_non_neg_place]
            answer_info[ans_non_neg_key] = [name2.lstrip(), name2_info]
        ans_unknown_key = list(answer_info.keys())[ans_unknown_place]
        answer_info[ans_unknown_key] = [ans_ambig, "unknown"]

    dict1 = make_dict(
        nn,
        q_id,
        "neg",
        "ambig",
        cat,
        subcat,
        answer_info,
        bias_targets,
        version,
        notes,
        text_ambig,
        q_neg,
        ans_list,
        ans_unknown_place,
    )
    nn += 1
    dict2 = make_dict(
        nn,
        q_id,
        "neg",
        "disambig",
        cat,
        subcat,
        answer_info,
        bias_targets,
        version,
        notes,
        "%s %s" % (text_ambig.strip(), text_disambig),
        q_neg,
        ans_list,
        ans_neg_place,
    )
    nn += 1
    dict3 = make_dict(
        nn,
        q_id,
        "nonneg",
        "ambig",
        cat,
        subcat,
        answer_info,
        bias_targets,
        version,
        notes,
        text_ambig,
        q_non_neg,
        ans_list,
        ans_unknown_place,
    )
    nn += 1
    dict4 = make_dict(
        nn,
        q_id,
        "nonneg",
        "disambig",
        cat,
        subcat,
        answer_info,
        bias_targets,
        version,
        notes,
        "%s %s" % (text_ambig.strip(), text_disambig),
        q_non_neg,
        ans_list,
        ans_non_neg_place,
    )

    return [dict1, dict2, dict3, dict4]

split_and_strip = lambda s, sep: [item.strip() for item in s.split(sep)]
remove_brackets = lambda s: re.sub(r"(\{\{?)|(\}\}?)", "", s)

def parse_dict_from_string(s) -> OrderedDict:
    if not s:
        return {}

    s = s.replace('"', "")
    # e.g. "NAME1-indef: [un nieto, una nieta]; NAME1-def: [el nieto, la nieta]; NAME2: [abuelo, abuela]; NAME2-def: [el abuelo, la abuela]"
    key_val_pairs: list = split_and_strip(s, ";")
    key_val_pairs: list = [split_and_strip(pair, ":") for pair in key_val_pairs]
    str_list_pairs: list = [[remove_brackets(key), split_and_strip(val.strip()[1:-1], ",")] for key, val in key_val_pairs]

    return OrderedDict(str_list_pairs)

def group_by_specifiers(original_dict: dict) -> dict:
    """
    Takes a dict like {"WORD1": [], "WORD1-indef": [], "WORD2": [], "WORD2-def": [], ...} and returns a dict where the keys are only "WORD1", "WORD2" etc. without specifiers and each key is another dict where the keys are "def", "indef" and None (meaning raw form without articles).
    """
    expanded_dict = {}
    for key, word_list in original_dict.items():
        if "-" in key:
            new_key, specifier = key.split("-")

            if not new_key in expanded_dict:
                expanded_dict[new_key] = {}

            expanded_dict[new_key][specifier] = word_list

        else:
            expanded_dict[key] = {}
            expanded_dict[key][None] = word_list

    # for each key, make sure that all the lists have the same length
    for label in expanded_dict:
        lengths = [len(value_list) for _, value_list in expanded_dict[label].items()]
        assert len(set(lengths)) == 1, f"Lists for {label} don't match: {expanded_dict[label]}"

    return expanded_dict

def capitalize_sents(s: str) -> str:
    sents = sent_tokenize(s, language="spanish")
    new_sents = []

    for sent in sents:
        if sent[0] in ["¿", "¡"]:
            sent = sent[0] + sent[1].upper() + sent[2:]
        else:
            sent = sent[0].upper() + sent[1:]

        new_sents.append(sent)

    return " ".join(new_sents)

def group_vocab_rows(df_vocab) -> dict[str, list]:
    """
    Process a DF with two columns, "Name_es" and "Info", and convert them to a dictionary where the keys are the values of "Info" ("def" or "indef") and the values are all available names in the Names column. Additionally, creates a group with the None key where the values are the names without the "el/la/un/una" articles in the beginning.
    """

    grouped = defaultdict(list)
    for term, info in df_vocab[["Name_es", "Info"]].values.tolist():
        grouped[info].append(term)

    if set(grouped.keys()) == {"indef", "def"}:
        grouped[None] = [np[np.index(" ")+1:] for np in grouped["indef"]]

    return dict(grouped)

def get_word_combinations(grouped_dict: dict) -> list[dict[str, int]]:
    """
    TODO
    """
    labels = list(grouped_dict.keys())
    lengths = [len(grouped_dict[label][list(grouped_dict[label].keys())[0]]) for label in labels]

    index_product = list(itertools.product(*[range(length) for length in lengths]))
    
    selections = [{label : idx for label, idx in zip(labels, selection)} for selection in index_product]
    return selections
