import re
import pdb

def extract_first_user_comment(text):
    # Define the regular expression pattern
    pattern = r'User\'s Comment:\s*"(.*?)"'
    
    # Search for the pattern in the text
    match = re.search(pattern, text)
    
    # If a match is found, return the first captured group (the text inside the double quotes)
    if match:
        return match.group(1)
    else:
        # If no match is found, return None or a custom message
        return None
    
def find_quoted_text_after_3rd_output(text):
    # Define a regular expression pattern to match "Output:" followed by quoted text
    pattern = r'Output:\s+"(.*?)"'
    
    # Find all matches of the pattern in the text
    matches = re.findall(pattern, text)
    
    # Check if there are at least 3 matches
    if len(matches) >= 3:
        # Return the quoted text after the 3rd "Output:"
        return matches[2]
    
    # Return None if there are fewer than 3 matches
    return None


def get_answer(prompt, lang_model, tokenizer, device):
    input = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    input_ids = input.input_ids
    attention_mask = input.attention_mask
    outputs = lang_model.generate(input_ids, attention_mask=attention_mask, max_length=2048, temperature = 0.7, top_k = 10, top_p=0.9, do_sample=True,)
    
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    #print(answer)
    #pdb.set_trace()
    #print(answer)
    #parsed = [re.findall(r'Output:[\s\S]+?"(.*?)"', a)[2] for a in answer]
    parsed =  [re.search(r'###.*?Output:(.*?)(?:$|\n)', a, re.DOTALL).group(1) for a in answer]
    print(parsed)
    return parsed

def create_zeroshot_prompt(arg):
    return """### You are an intelligent SQL Code assistant who effectively translates the intent and logic of the SQL queries into natural language that is easy to understand. Convert the given SQL query into a clear and concise natural language query.  Ensure that the request accurately represents the actions specified in the SQL query and is easy to understand for someone without technical knowledge of SQL. Only generate proper English.\n### Input: {0}\nOutput""".format(arg)

def create_cot_prompt(arg):
    return """### You are an intelligent SQL Code assistant who effectively translates the intent and logic of the SQL queries into natural language that is easy to understand. Convert the given SQL query into a clear and concise natural language query.  Ensure that the request accurately represents the actions specified in the SQL query and is easy to understand for someone without technical knowledge of SQL. Only generate proper English. Lets think Step by Step. \n### Input: {0}\nOutput""".format(arg)

def create_justcode_prompt(arg):
    return arg

def create_incontext_prompt(*args):
    return """
    ### System:  You are an intelligent SQL Code assistant who effectively translates the intent and logic of the SQL queries into natural language that is easy to understand.
    ### User: Convert the given SQL query into a clear and concise natural language query.  Ensure that the request accurately represents the actions specified in the SQL query and is easy to understand for someone without technical knowledge of SQL. 
    ### Examples: 
    Input: "{0}"
    Output: "{1}"
    Input: "{2}"
    Output: "{3}"
    ###
    Input: "{4}"
    Output: """.format(*args)

def create_incontext_prompt2(*args):
    if len(args) % 2 == 0:
        raise ValueError("The number of arguments must be odd.")
    
    # Initialize an empty string to accumulate the formatted text
    formatted_text = ""
    # Iterate through pairs of arguments
    for i in range(0, len(args)-1, 2):
        input_str = args[i]
        output_str = args[i + 1]
        # Format the input and output into the desired format
        formatted_text += 'Input: {0}\nOutput: {1}\n###\n'.format(input_str, output_str)
    formatted_text += 'Input: {0}\nOutput:'.format(args[-1])
    return formatted_text