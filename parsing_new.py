import re
import os
from tqdm import tqdm 
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import traceback 
import random

from transformers import BertTokenizerFast, BertModel, RobertaTokenizerFast
from parsing import missing_words_in_lambda_terms, simplest

TOKENIZER_MULTILINGUAL = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased") #
TOKENIZER_BASE = BertTokenizerFast.from_pretrained("bert-base-uncased")
TOKENIZER_ROBERTA = RobertaTokenizerFast.from_pretrained("FacebookAI/roberta-base")
TOKENIZER = {"multilingual_bert": TOKENIZER_MULTILINGUAL, 
             "bert_base": TOKENIZER_BASE,
             "roberta_base":TOKENIZER_ROBERTA}
SEP_ID = {"multilingual_bert": 102, 
             "bert_base": 102,
             "roberta_base": 2}

SPEC_SENTENCE_INDICES = [8075, 8076, 132621, 62223, 62224, 132626, 85395, 126867, 85397, 85396, 126868, 8344, 85398, 8345, 91423, 85408, 91425, 91424, 85409, 91426, 23206, 23207, 62120, 62121, 70189, 70190, 85435, 85436, 23357, 23358, 18878, 23359, 18879, 18883, 18884, 126917, 126918, 107888, 107889, 129525, 129526]
def clean_lambda_tokens(text):
    # Add spaces around lambda calculus symbols
    text = re.sub(r'([()λ])', r' \1 ', text)  # Space around parens and lambda
    text = re.sub(r'\s+', ' ', text)  # Clean up multiple spaces
    text = text.strip()
    return text

class LambdaTerm:
    pass

class Variable(LambdaTerm):
    def __init__(self, name):
        self.name = name
        self.applications = 0
        self.abstractions = 0

    def __repr__(self):
        return self.name
    
    def __len__(self):
        return 0

class Abstraction(LambdaTerm):
    def __init__(self, variable, body, applications=0, abstractions=0):
        self.variable = variable
        self.body = body
        self.applications = applications
        self.abstractions = 1 + abstractions

    def __repr__(self):
        return f"λ{self.variable}"
    
    def __len__(self):
        return self.applications + self.abstractions

class Application(LambdaTerm):
    def __init__(self, function, argument, applications=0, abstractions=0):
        self.function = function
        self.argument = argument
        self.applications = 1 + applications
        self.abstractions = abstractions

    def __repr__(self):
        return "@"

    def __len__(self):
        return self.applications + self.abstractions
    
def application(stack):
    args = [] 
    while stack and stack[-1] != '(': # eat until you find the opening bracket
        args.append(stack.pop())
    if not stack:
        raise ValueError("Invalid lambda term: mismatched brackets")
    stack.pop()  # Remove '('
    if len(args) == 1:
        stack.append(args[0])
    else:
        # Build the application tree
        func = args[-1]
        for arg in reversed(args[:-1]):
            func = Application(func, arg, applications=(func.applications if type(func) is not str else 0) + (arg.applications if type(arg) is not str else 0), abstractions=(func.abstractions if type(func) is not str else 0) + (arg.abstractions if type(arg) is not str else 0))
        stack.append(func)
    return stack

def preprocess_sent(sentence):
    words = " ".join(sentence).replace("...}"," ...}").replace("{..","{. .").replace("NP.","NP .").replace("NP—","NP —").replace(",}"," ,}").\
    replace("'re"," 're").replace("'s"," 's").replace("'ve}"," 've}").replace("!}"," !}").replace("?}"," ?}").replace("n't"," n't").\
    replace("'m}"," 'm}").replace("{. ..","{...").replace("{——}","{— —}").replace("{--—}","{- -—}").replace("St.", "St").strip().split()
    
    words = [word[:-1].strip("\u200e") if word[-1] == "}" else word for i, word in enumerate(words) if i % 2 != 0] # -1 to get rid of the } at the end. sentences in PTB tokenized form with POS tagging - {Tag word}

    for i, word in enumerate(words):
        if "." in word and len(list(set(word))) != 1: words[i] = words[i].replace(".", "")
        # tiny preprocessing change λ to µ 
        if word == "λ": words[i] = "µ"

    tokens = TOKENIZER[os.environ["BERT_TYPE"]](" ".join(words), add_special_tokens=True, return_tensors="pt", return_offsets_mapping=True)
    return tokens, words

def preprocess_sent(sentence):
    words = " ".join(sentence).replace("...}"," ...}").replace("{..","{. .").replace("NP.","NP .").replace("NP—","NP —").replace(",}"," ,}").\
    replace("'re"," 're").replace("'s"," 's").replace("'ve}"," 've}").replace("!}"," !}").replace("?}"," ?}").replace("n't"," n't").\
    replace("'m}"," 'm}").replace("{. ..","{...").replace("{——}","{— —}").replace("{--—}","{- -—}").replace("St.", "St").strip().split()
    
    words = [word[:-1].strip("\u200e") if word[-1] == "}" else word for i, word in enumerate(words) if i % 2 != 0] # -1 to get rid of the } at the end. sentences in PTB tokenized form with POS tagging - {Tag word}

    for i, word in enumerate(words):
        if "." in word and len(list(set(word))) != 1: words[i] = words[i].replace(".", "")
        # tiny preprocessing change λ to µ 
        if word == "λ": words[i] = "µ"

    tokens = TOKENIZER[os.environ["BERT_TYPE"]](" ".join(words), add_special_tokens=True, return_tensors="pt", return_offsets_mapping=True)
    return tokens, words

def parse_lambda_term_1(term, term_map, term_indices, return_offset=False):
    stack = []
    i = 0
    
    close_brackets = term.count(")") != 0
    tokenizer = TOKENIZER[os.environ["BERT_TYPE"]]

    while i < len(term):
        if term[i] == '(':
            stack.append('(')
            i += 1
        elif term[i] == 'λ':
            # Parse the abstraction: λx.body
            var = term[i+1]
            i += 2  # Skip '.'
            # Parse the body of the abstraction
            body, offset = parse_lambda_term_1(term[i:], lambda x: term_map(x+i), term_indices, return_offset=True)
            # if type(body) is str:
            #     print(body, stack, term)
            #what came just before is a bracket we dont need that anymore
            stack.pop()
            stack.append(Abstraction(Variable(var), body, applications=body.applications if type(body) is not str else 0, abstractions=body.abstractions if type(body) is not str else 0))
            while not close_brackets and len(stack) > 1 and stack[-2] != "(" and "(" in stack: # keep eating until you find an opening bracket
                stack = application(stack)
            i += offset  # Move the index forward by the length of the body
        elif term[i] == ')':
            # Handle closing bracket: build the application or abstraction
            stack = application(stack)
            i += 1
        else:
            # Parse variables or constants
            join_vars = type(stack[-1]) is Variable and "specvar" not in stack[-1].name and "specvar" not in term[i] and (term_indices[term_map(i)] == term_indices[term_map(i-1)])

            if not close_brackets and len(stack) > 1 and "(" in stack and join_vars:
                var = stack[-1].name+ term[i]
                stack[-1] = Variable(var)
            else:
                var = term[i]
                stack.append(Variable(var))

            join_vars_next = "specvar" not in term[i] and i < len(term)-1 and term[i+1] not in "(λ" and "specvar" not in term[i+1] and (term_indices[term_map(i)] == term_indices[term_map(i+1)])

            while not close_brackets and len(stack) > 1 and stack[-2] != "(" and "(" in stack and not join_vars_next: # keep eating until you find an opening bracket
                stack = application(stack)

            i+=1
        if return_offset and stack.count("(") == 0:
            break # this lambda term is over

    if not stack:
        raise ValueError("Invalid lambda term: empty term")
    if return_offset:
        return stack[0], i
    return stack

def make_lambda_term_list(words, lambda_term):
    #ptb tokenize the lambda term 
    lambda_term_list = []
    acc = ""
    # assert len(re.findall(r"<w_\d+>", lambda_term)) == 0, f"Invalid lambda term {lambda_term}\n{words}"
    weird_dots = re.findall(r"<w_\d+>", lambda_term)
    for i, dot in enumerate(weird_dots):
        number = int(dot[3:-1])
        lambda_term = lambda_term.replace(dot, f"{words[number]}_{number}")
    var_logs = []
    var_dict={}
    for char in lambda_term:
        if char in "( )":
            if acc != "":
                #matches lambdaNP_4?
                if re.match(r"λ\w+_\d+.", acc):
                    # split it up into variable and number:
                    lambda_term_list.append("λ")
                    acc = acc[1:-1]
                    assert acc not in var_dict
                    var_dict[acc] = f"specvar{len(var_dict)}"
                lambda_term_list.append(acc)
                acc = ""
            if char != " ": lambda_term_list.append(char)
        else: 
            acc += char
    if acc != "": lambda_term_list.append(acc)
    lambda_term_list = [w.replace("\u200e", "") for w in lambda_term_list]

    #variable renaming:
    var_pattern = re.compile(r"(S_\d+|NP_\d+|N_\d+|PP_\d+)")

    i = 0
    while i < len(lambda_term_list):
        term = lambda_term_list[i]
        if var_pattern.match(term) and term in var_dict:
            lambda_term_list[i] = var_dict[term]
        i += 1
    #remove ) brackets:
    lambda_term_list = [term for term in lambda_term_list if term != ")"]
    return lambda_term_list

def get_lambda_word_mapping(lambda_term_list, words, replacement="J"):
    term_to_word_index = {}
    replace_copy = lambda_term_list.copy()
    #at this point we have replace copy like: ['(', '(', 'is_15', '(', 'an_16', '(', 'American_17', ...]

    # words[3] = "lamda"
    for w, word in enumerate(words):
        #find if this constitutes an entity in the lambda term
        #entities are of the form words_dddd
        if word == "(": word = "LRB"
        elif word == ")": word = "RRB"
        elif word == "[": word = "LSB"
        elif word == "]": word = "RSB"
        elif word == "{": word = "LCB"
        elif word == "}": word = "RCB"

        if "λ" in word: word = word.replace("λ", "")
        # if word not in lambda_term and word != "lamda": continue
        if word not in " ".join(lambda_term_list): continue
        #traverse the lambda_term 
        min_indx, min_no = 500000, 500000
        for i, term in enumerate(lambda_term_list):
            if re.findall(r"_\d+", term) and term[: term.rfind(re.findall(r"_\d+", term)[0])] == word and "specvar" not in term and "_" in term:
                no = int(term[term.rfind(re.findall(r"_\d+", term)[0])+1:])
                if no < min_no:
                    min_no = no
                    min_indx = i
        if min_indx == 500000: 
            continue
        #replace with something random
        lambda_term_list[min_indx] = replacement*len(word)
        term_to_word_index[min_indx] = w # index in the term to the index in the word list 
    lambda_term_list = replace_copy
    return lambda_term_list, term_to_word_index

def lambda_term_tokens(lambda_term_list, tokenized,
                        words_to_token_indices, lambda_term_indices):
    tokenizer = TOKENIZER[os.environ["BERT_TYPE"]]
    new_lambda_term_list = []
    lambda_lambda_mapping = {}
    for term_index, item in enumerate(lambda_term_list):
        if item in "(λ" or "specvar" in item:
            lambda_lambda_mapping[len(new_lambda_term_list)] = term_index
            new_lambda_term_list.append(item)
            continue
        word_index = lambda_term_indices[term_index]
        tokens = [tokenized[t] for t in words_to_token_indices[word_index]]
        for i in range(len(tokens)):
            lambda_lambda_mapping[len(new_lambda_term_list) + i] = term_index
        new_lambda_term_list.extend([tokenizer.decode([t]) for t in tokens])
    return new_lambda_term_list, lambda_lambda_mapping

def get_word_token_mapping(words, tokens, replacement="J"):
    word_mapping = tokens.words()
    word_mapping[0] = -1
    word_mapping[-1] = -1
    offset_mapping = tokens.offset_mapping[0]

    new_word_mapping = []

    # print(lambda_term)
    prev_end=1000
    how_many=0
    repl_words = words.copy()
    acc=""
    for i, (start, end) in enumerate(offset_mapping):
        if start ==0 and end == 0: 
            if acc != "": new_word_mapping.extend([words.index(acc)]*how_many)
            new_word_mapping.append(-1)
        else: 
            if start > prev_end and acc in words:
                new_word_mapping.extend([words.index(acc)]*how_many)
                words[words.index(acc)] = replacement * len(acc)
                acc = ""
                how_many=0   
            acc += " ".join(words)[start:end]
            how_many +=1
            prev_end = end
    

    words = repl_words

    word_token_mapping = {}
    for i in range(len(new_word_mapping)):
        if new_word_mapping[i] == -1: continue
        word_token_mapping[new_word_mapping[i]] = word_token_mapping.get(new_word_mapping[i], []) + [i]

    return words, word_token_mapping

def specvar_checker_multi(lambda_term_list):
    new_lambda_term_list = []
    next_is_var_index = False
    sp, ec = False, False
    for item in lambda_term_list:
        if next_is_var_index:
            next_is_var_index = False
            new_lambda_term_list[-1] = new_lambda_term_list[-1] + item
            new_lambda_term_list[-1] = new_lambda_term_list[-1].replace("#", "")
            continue
        if item == "sp":
            sp = True
            new_lambda_term_list.append(item)
        elif item == "##ec":
            if sp: ec = True
            new_lambda_term_list.append(item)
        elif item == "##var":
            if sp and ec: 
                new_lambda_term_list = new_lambda_term_list[:-2] + [new_lambda_term_list[-2]+new_lambda_term_list[-1]+item]
                next_is_var_index = True
                sp, ec = False, False
            else:
                new_lambda_term_list.append(item)
        else:
            new_lambda_term_list.append(item)
    return new_lambda_term_list

def specvar_checker_roberta(lambda_term_list):
    new_lambda_term_list = []
    next_is_var_index = False
    sp = False
    for item in lambda_term_list:
        if item == " ": continue
        if next_is_var_index:
            next_is_var_index = False
            new_lambda_term_list[-1] = new_lambda_term_list[-1] + item
            new_lambda_term_list[-1] = new_lambda_term_list[-1].replace("#", "")
            continue
        if item == " spec":
            sp = True
            new_lambda_term_list.append(item)
        elif item == "var":
            if sp: 
                new_lambda_term_list = new_lambda_term_list[:-1] + [new_lambda_term_list[-1]+item]
                next_is_var_index = True
                sp = False
            else:
                new_lambda_term_list.append(item)
        else:
            new_lambda_term_list.append(item)
    return new_lambda_term_list

def specvar_checker_bert_base(lambda_term_list):
    new_lambda_term_list = []
    next_is_var_index = False
    sp = False
    for item in lambda_term_list:
        if next_is_var_index:
            next_is_var_index = False
            new_lambda_term_list[-1] = new_lambda_term_list[-1] + item
            new_lambda_term_list[-1] = new_lambda_term_list[-1].replace("#", "")
            continue
        if item == "spec":
            sp = True
            new_lambda_term_list.append(item)
        elif item == "##var":
            if sp: 
                new_lambda_term_list = new_lambda_term_list[:-1] + [new_lambda_term_list[-1]+item]
                next_is_var_index = True
                sp = False
            else:
                new_lambda_term_list.append(item)
        else:
            new_lambda_term_list.append(item)
    return new_lambda_term_list


def get_marked_lambda_term_list(lambda_term_list, lambda_term_indices,
                                words, words_to_token_indices, tokens, replacement="J"):
    if lambda_term_indices is None:
        assert type(lambda_term_list) is str
        lambda_term_list_og = make_lambda_term_list(words, lambda_term_list)
        lambda_term_list_og, lambda_term_indices = get_lambda_word_mapping(lambda_term_list_og, 
                                                      words,
                                                      replacement=replacement)
        lambda_term_list, lambda_lambda_indices = lambda_term_tokens(lambda_term_list_og, tokens, words_to_token_indices, lambda_term_indices)
        return lambda_term_list, lambda x: lambda_lambda_indices[x], lambda_term_indices
    else:
        jack_map_index = [w for w in words_to_token_indices if len(words_to_token_indices[w]) == 1][0]
        tokens_indices_to_words = {}
        lambda_term_list = SPEC_VAR_CHECKER[os.environ["BERT_TYPE"]](lambda_term_list)
        if lambda_term_list[-1] in ("[SEP]", "</s>"): 
            lambda_term_list = lambda_term_list[:-1]
            lambda_term_indices = lambda_term_indices[:-1]
        for word in words_to_token_indices:
            ts = words_to_token_indices[word]
            for t in ts:
                tokens_indices_to_words[t] = word
        lambda_term_indices_for_word = {}
        for i, t_idx in enumerate(lambda_term_indices):
            if t_idx == -1: continue
            if tokens[t_idx.item()] == SEP_ID[os.environ["BERT_TYPE"]]:
                lambda_term_indices_for_word[i] = jack_map_index
                continue
            lambda_term_indices_for_word[i] = tokens_indices_to_words[t_idx.item()]
        return lambda_term_list, lambda x: x, lambda_term_indices_for_word

def get_subtrees(tree):
    if isinstance(tree, Variable):
        return [tree]
    elif isinstance(tree, Abstraction):
        return [tree] + get_subtrees(tree.body)
    elif isinstance(tree, Application):
        return [tree] + get_subtrees(tree.function) + get_subtrees(tree.argument)
    else:
        assert type(tree) is str, f"Invalid tree {tree}"
        return []

def get_subtree_list(forest):
    subtrees = []
    for tree in forest:
        if type(tree) is str: #TODO: gonna have to figure out something for this
            continue
        subtrees += get_subtrees(tree)
    return subtrees

def fake_eval(true_tree, inf_forest, count_abstractions=False, count_applications=False):
    true_subtrees = get_subtree_list([true_tree])
    inf_subtrees = get_subtree_list(inf_forest)

    if len(true_subtrees) == 0:
        return []

    if len(inf_subtrees) == 0:
        return None
    
    checker = (lambda x, y: x.applications == y.applications) if not count_abstractions \
          else (lambda x, y: (x.applications == y.applications if count_applications else True) & (x.abstractions == y.abstractions))
    recall = len(set([t_t for t_t in true_subtrees for i_t in inf_subtrees if checker(t_t, i_t)]))/len(true_subtrees)
    precision = len(set([i_t for i_t in inf_subtrees for t_t in true_subtrees if checker(t_t, i_t)]))/len(inf_subtrees)
    return 2*(precision * recall)/(precision + recall) if (precision+recall > 0) else 0 , precision, recall

def fake_eval_no_recount(true_tree, inf_forest, 
                         count_abstractions=False, count_applications=False):
    true_subtrees = get_subtree_list([true_tree])
    inf_subtrees = get_subtree_list(inf_forest)

    if len(true_subtrees) == 0:
        return []

    if len(inf_subtrees) == 0:
        return None
    
    true_count_dict, inf_count_dict = {}, {}
    index_tuple = (lambda x: ("app", x.applications)) if not count_abstractions \
          else (lambda x: (("app", x.applications,) if count_applications else ()) + ("abs", x.abstractions,))
    for subtree in true_subtrees:
        true_count_dict[index_tuple(subtree)] = true_count_dict.get(index_tuple(subtree), 0) + 1
    for subtree in inf_subtrees:
        inf_count_dict[index_tuple(subtree)] = inf_count_dict.get(index_tuple(subtree), 0) + 1
    
    recall = sum([min(true_count_dict.get(key_tup, 0), 
                      inf_count_dict.get(key_tup, 0)) for key_tup in true_count_dict])/len(true_subtrees)
    precision = sum([min(true_count_dict.get(key_tup, 0),
                          inf_count_dict.get(key_tup, 0)) for key_tup in inf_count_dict])/len(inf_subtrees)
    return 2*(precision * recall)/(precision + recall) if (precision+recall > 0) else 0 , precision, recall

def print_tree(nodes, prefix="", is_last=True):
    if type(nodes) is not list:
        nodes = [nodes]
    for node in nodes:
        if isinstance(node, Application):
            print(prefix + ("└── " if is_last else "├── ") + str(node))
            print_tree(node.function, prefix + ("    " if is_last else "│   "), False)
            print_tree(node.argument, prefix + ("    " if is_last else "│   "), True)
        elif isinstance(node, Abstraction):
            print(prefix + ("└── " if is_last else "├── ") + str(node))
            print_tree(node.body, prefix + ("    " if is_last else "│   "), True)
        elif isinstance(node, Variable):
            print(prefix + ("└── " if is_last else "├── ") + str(node))

def find_height_tree(tree):
    if isinstance(tree, Variable):
        return 0
    elif isinstance(tree, Abstraction):
        return 1 + find_height_tree(tree.body)
    elif isinstance(tree, Application):
        return 1 + max(find_height_tree(tree.function), find_height_tree(tree.argument))
    else:
        raise ValueError("Invalid tree structure")
    
def tokenize_lambda_term_list(lambda_term_list):
    new_list = []
    tok = TOKENIZER[os.environ["BERT_TYPE"]]
    for item in lambda_term_list:
        if item in "(λ": 
            new_list.append(item)
        elif re.match(r"specvar\d+", item):
            new_list.append(item)
        else:
            _index = item.index("_")
            item = item[:_index]
            new_list.extend([tok.decode()])
    return new_list


def filter_words_by_lambda_term(lambda_term_list, words,replacement="J"):
    missing_words, lambda_term_list = missing_words_in_lambda_terms(words,
                                                                        lambda_term_list,
                                                                        None,
                                                                        replacement=replacement
                                                                        )
    words = [w for w_idx, w in enumerate(words) if w_idx not in missing_words]
    tokens = TOKENIZER[os.environ["BERT_TYPE"]](" ".join(words), add_special_tokens=True, return_tensors="pt", return_offsets_mapping=True)
    return words, tokens
 
def brackets_remove_tokens(tokens):
    tokenizer = TOKENIZER[os.environ["BERT_TYPE"]]
    replace = {"(": tokenizer.encode("[", add_special_tokens=False)[0],
               ")": tokenizer.encode("]", add_special_tokens=False)[0]}
    for i, token in enumerate(tokens):
        if tokenizer.decode([token]) in "()":
            tokens[i] = replace[tokenizer.decode([token])]
    return tokens        

SPEC_VAR_CHECKER = {"multilingual_bert": specvar_checker_multi, 
             "bert_base": specvar_checker_bert_base,
             "roberta_base": specvar_checker_roberta}

if __name__ == "__main__":
    import argparse
    import pickle
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_file")
    parser.add_argument("--csv_file")
    parser.add_argument('--skip_file')
    parser.add_argument("--name", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--simplest", action="store_true")
    parser.add_argument("--filtered", action="store_true")
    parser.add_argument("--norecounts", action="store_true")
    # parser.add_argument("--mode")
    args = parser.parse_args()
    
    # assert args.mode in ("app", "abs", "both")
    name = args.name
    f = open(args.data_path+"dataset_splits.pkl", "rb")
    data_split = pickle.load(f)
    f.close()

    skip_file = open(args.skip_file, "rb")
    skips = pickle.load(skip_file)
    skip_file.close()

    main_df = pd.read_csv(args.data_path+"input_sentences.csv", header=None, names=["sentence", "tags", "path"])
        
    split = "test" if "test" in args.out_file else ("valid" if "valid" in args.out_file else "train")
    _, _, split_numbers = data_split
    with open(args.out_file, "rb") as f:
        out_data = pickle.load(f)
    
    df = pd.read_csv(args.csv_file, header=None, names=["true_term", "inf_term"])

    baddies = []
    f_scores_both, precisions_both, recalls_both = [], [], []
    f_scores_app, precisions_app, recalls_app = [], [], []
    f_scores_abs, precisions_abs, recalls_abs = [], [], []
    forest_sizes = []
    forests = []

    study_precisions = []

    double_tree_terms = []
    skip_count = 0
    perfect_counts = 0

    assert len(out_data) == len(df)
    skips = list(map(lambda x: x - skips.index(x), skips))
    for i, (out, out_indices) in tqdm(enumerate(out_data), total=len(out_data)):
        if skips and i >= skips[0]:
            skip_count += 1
            skips.pop(0)

        # true_term_list = row["true_term"].split()
        sent_index = split_numbers[i+skip_count]
        while sent_index in SPEC_SENTENCE_INDICES:
            skip_count+=1
            sent_index = split_numbers[i+skip_count]
        true_sentence = eval(main_df.iloc[sent_index].tags)
        
        with open(args.data_path+"/".join(main_df.iloc[sent_index].path.split("/")[2:])) as f1:
            true_term_list = f1.readlines()[0].strip() if not args.simplest else simplest(f1.readlines(), gen_sent = eval(main_df.iloc[sent_index, 1]))

        if "[UNK]" in df.iloc[i]["true_term"]: continue
        
        tokens, words = preprocess_sent(true_sentence)
        #clean_up
        weird_dots = re.findall(r"<w_\d+>", true_term_list)
        for i, dot in enumerate(weird_dots):
            number = int(dot[3:-1])
            true_term_list = true_term_list.replace(dot, f"{words[number]}_{number}")
        
        if "JJ" in words: 
            continue # due to a tiny error on my part that makes such sentences unreliable
        replacement = "J"
        while replacement in words or replacement*2 in words or replacement*3 in words:
            replacement = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[random.randint(0, 25)]

        if args.filtered:
            words, tokens = filter_words_by_lambda_term(make_lambda_term_list(words, true_term_list), words, replacement=replacement)
        
        words, word_to_token_indices = get_word_token_mapping(words, tokens, replacement=replacement)

        tokens = tokens.input_ids.squeeze(0).tolist()
        # true_term_list = make_lambda_term_list(words, true_term_list)
        # true_term_list = tokenize_lambda_term_list(true_term_list)
        
        tokens = brackets_remove_tokens(tokens)
        
        true_term_list, true_mid_map, true_term_list_word_indices = get_marked_lambda_term_list(true_term_list, None,
                                    words, word_to_token_indices, tokens, replacement=replacement)
        
        lambda_term_list, lambda_mid_map, lambda_term_list_word_indices= get_marked_lambda_term_list(out, out_indices, 
                                    words, word_to_token_indices, tokens, replacement=replacement)
        
        try:
            trees = parse_lambda_term_1(lambda_term_list, 
                                        lambda_mid_map,
                                        lambda_term_list_word_indices)
            forests.append(trees)
            forest_sizes.append(len(trees))
            if len(trees) == 2:
                double_tree_terms.append((" ".join(lambda_term_list), " ".join(words), " ".join(true_term_list)))
                
        except:
            baddies.append(i+skip_count)
            continue
        # try:
        true_tree = parse_lambda_term_1(true_term_list,
                                        true_mid_map,
                                        true_term_list_word_indices)
        # print_tree(true_tree)
        assert len(true_tree) == 1, f"True tree is not a single tree: {true_tree}, {true_term_list}"
        true_tree = true_tree[0]
        # except Exception as e:
        #     print(e)
        #     print(traceback.format_exc())
        #     raise Exception("This should not have happened. Lamda term is ", true_term_list)#, "\n with words\n", make_lambda_term_list(words, tt_true_term_list))
        for option in ("app", "abs", "both"):
            evl = fake_eval if not args.norecounts else fake_eval_no_recount
            x = evl(true_tree, trees, 
                    count_abstractions=option in ("abs", "both"), count_applications=option in ("app", "both"))
            if x is None:
                baddies.append(i+skip_count)
            elif x == []:
                continue
            elif option == "app":
                f_score, precision, recall = x
                f_scores_app.append(f_score)
                precisions_app.append(precision)
                recalls_app.append(recall)
            elif option == "abs":
                f_score, precision, recall = x
                f_scores_abs.append(f_score)
                precisions_abs.append(precision)
                recalls_abs.append(recall)
            elif option == "both":
                f_score, precision, recall = x
                f_scores_both.append(f_score)
                precisions_both.append(precision)
                recalls_both.append(recall)
                if precision == 1 and recall == 1:
                    perfect_counts += 1

            # if 0.3 <= precision < 0.4:
            #     study_precisions.append((row["inf_term"], row["true_term"], true_sentence))

    
    print("Proportion of bad trees: ", len(baddies)/len(df))
    print("Number of perfect trees: ", perfect_counts)

    #pickle forests
    with open(f"{name}_{split}_forests.pkl", "wb") as f:
        pickle.dump(forests, f)

    #write csv file of double trees;
    with open(f"{name}_{split}_double_trees.csv", "w") as f:
        f.write("inf_term,true_term,sentence\n")
        for term in double_tree_terms:
            f.write(f"{term[0]},{term[2]},{term[1]}\n")

    print("Mean forest size: ", sum(forest_sizes)/len(forest_sizes))
    print("Median forest size: ", sorted(forest_sizes)[len(forest_sizes)//2])
    print("Number of sentences with size of 1", forest_sizes.count(1))
    print("-----")
    
    # suffix = args.mode
    # Create histogram using seaborn for better default styling
    plt.figure(figsize=(10, 10))
    sns.histplot(forest_sizes, bins=max(forest_sizes))
    
    # Add labels and title
    plt.xlabel('Forest sizes')
    plt.ylabel('Count')
    plt.title('Well-formedness of lambda terms in {} set of length {}'.format(split, len(out_data)))
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    #restrict the x-axis to 50
    plt.xlim(0, 50)
    plt.savefig(f"{name}_{split}_forest_sizes.png")

    plt.clf()

    for i, (f_scores, precisions, recalls) in enumerate(zip(
        [f_scores_both, f_scores_app, f_scores_abs],
        [precisions_both, precisions_app, precisions_abs],
        [recalls_both, recalls_app, recalls_abs]
        )):
        suffix = {0:"both", 1: "app", 2: "abs"}[i]
        #create new histogram for precision, recall, and f-score
        plt.figure(figsize=(10, 10))
        sns.histplot(f_scores, bins=20)
        plt.xlabel('F1 scores')
        plt.ylabel('Count')
        plt.title('F1 scores of lambda terms in {} set of length {}'.format(split, len(out_data)))
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{name}_{split}_f_scores_{suffix}.png")
        
        plt.clf()
        plt.figure(figsize=(10, 10))
        sns.histplot(precisions, bins=20)
        plt.xlabel('Precisions')
        plt.ylabel('Count')
        plt.title('Precisions of lambda terms in {} set of length {}'.format(split, len(out_data)))
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{name}_{split}_precisions_{suffix}.png")

        plt.clf()
        plt.figure(figsize=(10, 10))
        sns.histplot(recalls, bins=20)
        plt.xlabel('Recalls')
        plt.ylabel('Count')
        plt.title('Recalls of lambda terms in {} set of length {}'.format(split, len(out_data)))
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{name}_{split}_recalls_{suffix}.png")

        print("--------")
        print(f"Average F-score {suffix}: ", sum(f_scores)/len(f_scores))
        print(f"Average Precision {suffix}: ", sum(precisions)/len(precisions))
        print(f"Average Recall {suffix}: ", sum(recalls)/len(recalls))
        print("--------")

    #save study precisions
    # with open(f"{name}_{split}_study_precisions.csv", "w") as f:
    #     f.write("inf_term,true_term,sentence\n")
    #     for term in study_precisions:
    #         f.write(f"{term[0]},{term[1]},{term[2]}\n")


# #test cases:
# # term2 = ["(", "(", "λ", "x", "x", ")", "(", "(", "λ", "y", "y", ")", "5", ")", ")"]
# term2 = "(Where_0 ((did_1 (two_2 (chemical_3 plants_4))) (λNP_4.(collapse_5 NP_4))))"
# sentence = ['{S/S', 'Where}', '{(S/(S\\NP))/NP', 'did}', '{NP/NP', 'two}', '{NP/NP', 'chemical}', '{NP', 'plants}', '{S\\NP', 'collapse}', '{?', '?}']
# # sentence = ['{NP/NP', 'Chengdu}', '{NP/NP', 'Shuangliu}', '{NP', 'Airport}', '{S\\NP', 'reopened}', '{((S\\NP)\\(S\\NP))/((S\\NP)\\(S\\NP))', 'later}', '{((S\\NP)\\(S\\NP))/NP', 'on}', '{NP/N', 'the}', '{N', 'evening}', '{(NP\\NP)/NP', 'of}', '{NP/N', 'May}', '{N', '12}', '{,', ',}', '{((S\\NP)\\(S\\NP))/NP', 'offering}', '{NP/NP', 'limited}', '{NP', 'service}', '{(((S\\NP)\\(S\\NP))\\((S\\NP)\\(S\\NP)))/S', 'as}', '{NP/N', 'the}', '{N', 'airport}', '{(S\\NP)/(S\\NP)', 'began}', '{(S\\NP)/(S\\NP)', 'to}', '{(S\\NP)/(S\\NP)', 'be}', '{(S\\NP)/PP', 'used}', '{PP/NP', 'as}', '{NP/N', 'a}', '{N/N', 'staging}', '{N', 'area}', '{(NP\\NP)/NP', 'for}', '{NP/NP', 'relief}', '{NP', 'operations}', '{.', '.}']
# # term2 = "(((later_4 (λS_14.(λNP_12.((((as_15 ((began_18 (λNP_52.((to_19 (λNP_56.((be_20 (λNP_60.((used_21 (as_22 ((for_26 (relief_27 operations_28)) (a_23 (staging_24 area_25))))) NP_60))) NP_56))) NP_52))) (the_16 airport_17))) (λS_40.(λNP_38.(((offering_12 (limited_13 service_14)) (λNP_30.(S_40 NP_30))) NP_38)))) (λNP_42.(((on_5 ((of_8 (May_9 12_10)) (the_6 evening_7))) (λNP_16.(S_14 NP_16))) NP_42))) NP_12)))) (λNP_8.(reopened_3 NP_8))) (Chengdu_0 (Shuangliu_1 Airport_2)))"
# _, words = preprocess_sent(sentence)
# term2 = make_lambda_term_list(words, term2)
# term2 = [t for t in term2 if t != ")"]
# term2 = ['(', 'Where_0', '(', '(', 'did_1', '(', 'two_2', '(', 'chemical_3', 'plants_4', '(', 'λ', 'NP_4', '(', 'collapse_5', 'NP_4', "?"]

# term2 = ['(', 'Who', '(', 'λ', 'x1', '(', '(', 'was_1', '(', 'λ', 'x2', '(', '(', 'responsible_2', '(', 'for_3', '(', 'λ', 'x3', '(', '(', 'changing_4',  '(', '(', "'s_6", 'Prussia_5', '(', 'internal_7' , 'borders_8', 'x3', 'x2', 'x1']
# print("Mutilated term 2 ", term2)
# tree2 = parse_lambda_term(term2)
# # print("Tree 2:")
# print_tree(tree2)


# --csv_file outputs/bert_simplest/simplest_nodup_test_all_lev.csv --out_file outputs/bert_simplest/simplest_stop_nodup_ret_test_all_lev.pkl --skip_file outputs/bert_simplest/simplest_nodup_test_all_lev_skips.pkl --name test_Test_del --data_path /w/nobackup/436/lambda/simplestlambda/data/ --mode both 

# --out_file outputs/bert_base_7_filt/bert_base_7_filt_stop_nodup_ret_test_all_lev.pkl --csv_file outputs/bert_base_7_filt/bert_base_7_filt_stop_nodup_test_all_lev.csv --skip_file outputs/bert_base_7_filt/bert_base_7_filt_stop_nodup_test_all_lev_skips.pkl --name bert_base_7_filt_stop_nodup --data_path /w/nobackup/436/lambda/bert_base_7_filtered/data/ --filtered