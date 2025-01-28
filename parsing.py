import re
from tokenization import preprocess_sent
from tqdm import tqdm 
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

class LambdaTerm:
    pass

class Variable(LambdaTerm):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

class Abstraction(LambdaTerm):
    def __init__(self, variable, body):
        self.variable = variable
        self.body = body

    def __repr__(self):
        return f"λ{self.variable}"

class Application(LambdaTerm):
    def __init__(self, function, argument):
        self.function = function
        self.argument = argument

    def __repr__(self):
        return "@"
    
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
            func = Application(func, arg)
        stack.append(func)
    return stack

def parse_lambda_term(term, return_offset=False):
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
            body, offset = parse_lambda_term(term[i:], return_offset=True)
            #what came just before is a bracket we dont need that anymore
            stack.pop()
            stack.append(Abstraction(Variable(var), body))
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
    return lambda_term_list

def parseeval(true_tree, inf_tree);
    pass

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

if __name__ == "__main__":

    files = ["/w/150/lambda_squad/lambdaBERT/train_all_lev.csv", "/w/150/lambda_squad/lambdaBERT/valid_all_lev.csv", "/w/150/lambda_squad/lambdaBERT/test_all_lev.csv"]
    for file in files:
        split = file.split("/")[-1].split("_")[0]
        df = pd.read_csv(file, header=None, names=["true_term", "inf_term"])
        baddies = []
        forest_sizes = []
        forests = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            lambda_term_list = row["inf_term"].split()
            try:
                trees = parse_lambda_term(lambda_term_list)
                forests.append(trees)
                forest_sizes.append(len(trees))
            except:
                baddies.append(i)
        print("Propotion of bad trees: ", len(baddies)/len(df))

        #pickle forests
        with open(f"{split}_forests.pkl", "wb") as f:
            pickle.dump(forests, f)

        # Create histogram using seaborn for better default styling
        plt.figure(figsize=(10, 10))
        sns.histplot(forest_sizes, bins=30, kde=True)
        
        # Add labels and title
        plt.xlabel('Forest sizes')
        plt.ylabel('Count')
        plt.title('Well-formedness of lambda terms in {} set of length {}'.format(split, len(df)))
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{split}_forest_sizes.png")

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
    # print("Mutilated term 2 ", term2)
    # tree2 = parse_lambda_term(term2)
    # # print("Tree 2:")
    # print_tree(tree2)