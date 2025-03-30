import re
# from tokenization import preprocess_sent
from tqdm import tqdm 
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import traceback 

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

def parse_lambda_term_1(term, return_offset=False):
    stack = []
    i = 0
    
    close_brackets = term.count(")") != 0

    while i < len(term):
        if term[i] == '(':
            stack.append('(')
            i += 1
        elif term[i] == 'λ':
            # Parse the abstraction: λx.body
            var = term[i+1]
            i += 2  # Skip '.'
            # Parse the body of the abstraction
            body, offset = parse_lambda_term_1(term[i:], return_offset=True)
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
            var = term[i]
            stack.append(Variable(var))
            
            while not close_brackets and len(stack) > 1 and stack[-2] != "(" and "(" in stack : # keep eating until you find an opening bracket
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

    for char in lambda_term:
        if char in "( )":
            if acc != "":
                #matches lambdaNP_4?
                if re.match(r"λ\w+_\d+.", acc):
                    # split it up into variable and number:
                    lambda_term_list.append("λ")
                    acc = acc[1:-1]
                lambda_term_list.append(acc)
                acc = ""
            if char != " ": lambda_term_list.append(char)
        else: 
            acc += char
    if acc != "": lambda_term_list.append(acc)
    lambda_term_list = [w.replace("\u200e", "") for w in lambda_term_list]

    #variable renaming:
    var_pattern = re.compile(r"(S_\d+|NP_\d+|N_\d+|PP_\d+)")

    var_dict = {}
    i = 0
    while i < len(lambda_term_list):
        term = lambda_term_list[i]
        if var_pattern.match(term):
            if term not in var_dict:
                var_dict[term] = f"spec_var{len(var_dict)}"
            lambda_term_list[i] = var_dict[term]
        i += 1
    #remove ) brackets:
    lambda_term_list = [term for term in lambda_term_list if term != ")"]
    return lambda_term_list

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

if __name__ == "__main__":

    files = ["/w/150/lambda_squad/lambdaBERT/outputs/test_all_lev.csv"]#["/w/150/lambda_squad/lambdaBERT/outputs/train_all_lev.csv", "/w/150/lambda_squad/lambdaBERT/outputs/valid_all_lev.csv", "/w/150/lambda_squad/lambdaBERT/outputs/test_all_lev.csv"]
    
    f = open("/w/150/lambda_squad/lambdaBERT/data/dataset_splits.pkl", "rb")
    data_split = pickle.load(f)
    f.close()

    main_df = pd.read_csv("/w/150/lambda_squad/lambdaBERT/data/input_sentences.csv", header=None, names=["sentence", "tags", "path"])
    for t, file in enumerate(files):
        
        split_numbers = data_split[t]

        split = file.split("/")[-1].split("_")[0]
        df = pd.read_csv(file, header=None, names=["true_term", "inf_term"])
        baddies = []
        f_scores, precisions, recalls = [], [], []
        forest_sizes = []
        forests = []

        study_precisions = []

        double_tree_terms = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            lambda_term_list = row["inf_term"].split()
            # true_term_list = row["true_term"].split()
            sent_index = split_numbers[i]
            true_sentence = eval(main_df.iloc[sent_index].tags)
            f1 = open("/".join(main_df.iloc[sent_index].path.split("/")[1:]))
            true_term_list = f1.readlines()[0].strip()
            f1.close()
            
            _, words = preprocess_sent(true_sentence)
            true_term_list = make_lambda_term_list(words, true_term_list)
            
            try:
                trees = parse_lambda_term(lambda_term_list)
                forests.append(trees)
                forest_sizes.append(len(trees))
                if len(trees) == 2:
                    double_tree_terms.append((" ".join(lambda_term_list), " ".join(words), " ".join(true_term_list)))
                    
            except:
                baddies.append(i)
                continue
            try:
                true_tree = parse_lambda_term(true_term_list)
                assert len(true_tree) == 1, f"True tree is not a single tree: {true_tree}, {true_term_list}"
                true_tree = true_tree[0]
            except Exception as e:
                print(e)
                print(traceback.format_exc())
                raise Exception("This should not have happened. Lamda term is ", true_term_list)
            
            x = fake_eval(true_tree, trees, count_abstractions=False, count_applications=True)
            if x is None:
                baddies.append(i)
            elif x == []:
                continue
            else:
                f_score, precision, recall = x
                f_scores.append(f_score)
                precisions.append(precision)
                recalls.append(recall)

                if 0.3 <= precision < 0.4:
                    study_precisions.append((row["inf_term"], row["true_term"], true_sentence))

        
        print("Proportion of bad trees: ", len(baddies)/len(df))

        #pickle forests
        with open(f"{split}_forests.pkl", "wb") as f:
            pickle.dump(forests, f)

        #write csv file of double trees;
        with open(f"{split}_double_trees.csv", "w") as f:
            f.write("inf_term,true_term,sentence\n")
            for term in double_tree_terms:
                f.write(f"{term[0]},{term[2]},{term[1]}\n")

        print("Mean forest size: ", sum(forest_sizes)/len(forest_sizes))
        print("Median forest size: ", sorted(forest_sizes)[len(forest_sizes)//2])
        print("Number of sentences with size of 1", forest_sizes.count(1))
        print("-----")
        
        suffix = "app"
        # Create histogram using seaborn for better default styling
        plt.figure(figsize=(10, 10))
        sns.histplot(forest_sizes, bins=max(forest_sizes))
        
        # Add labels and title
        plt.xlabel('Forest sizes')
        plt.ylabel('Count')
        plt.title('Well-formedness of lambda terms in {} set of length {}'.format(split, len(df)))
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3)
        #restrict the x-axis to 50
        plt.xlim(0, 50)
        plt.savefig(f"{split}_forest_sizes.png")

        plt.clf()

        #create new histogram for precision, recall, and f-score
        plt.figure(figsize=(10, 10))
        sns.histplot(f_scores, bins=20)
        plt.xlabel('F1 scores')
        plt.ylabel('Count')
        plt.title('F1 scores of lambda terms in {} set of length {}'.format(split, len(df)))
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{split}_f_scores_{suffix}.png")
        
        plt.clf()
        plt.figure(figsize=(10, 10))
        sns.histplot(precisions, bins=20)
        plt.xlabel('Precisions')
        plt.ylabel('Count')
        plt.title('Precisions of lambda terms in {} set of length {}'.format(split, len(df)))
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{split}_precisions_{suffix}.png")

        plt.clf()
        plt.figure(figsize=(10, 10))
        sns.histplot(recalls, bins=20)
        plt.xlabel('Recalls')
        plt.ylabel('Count')
        plt.title('Recalls of lambda terms in {} set of length {}'.format(split, len(df)))
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{split}_recalls_{suffix}.png")

        print("--------")
        print("Average F-score: ", sum(f_scores)/len(f_scores))
        print("Average Precision: ", sum(precisions)/len(precisions))
        print("Average Recall: ", sum(recalls)/len(recalls))
        print("--------")

        #save study precisions
        with open(f"{split}_study_precisions.csv", "w") as f:
            f.write("inf_term,true_term,sentence\n")
            for term in study_precisions:
                f.write(f"{term[0]},{term[1]},{term[2]}\n")


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