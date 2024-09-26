from accelerate import Accelerator
from accelerate.utils import gather_object
import pandas as pd
import torch, time
from tqdm import tqdm
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer

#from util.chatgpt import run_chatgpt
from run_prompts import run_prompts

from util.measure import measure
from pathlib import Path

accelerator = Accelerator()
models = {
    "codellama-7b": ("codellama/CodeLlama-7b-hf", 4096),
    "zephy7B": ("HuggingFaceH4/zephyr-7b-beta", 32768),
    "gemma-7B": ("google/gemma-7b", 8192),
    "Qwen-2-7B": ("Qwen/Qwen2-7B-Instruct", 32768),
    "mistral-7B": ("mistralai/Mistral-7B-v0.1", 2048),
    "openchat-8B": ("openchat/openchat-3.6-8b-20240522", 8192),
    "WizardLM-2-7B": ("lucyknada/microsoft_WizardLM-2-7B", 32768),
    "gpt-j-6b": ("EleutherAI/gpt-j-6B", 2048)
}

def get_filename(input_csv):
    # Get the file name without directory and extension
    filename = Path(input_csv).stem
    return filename

def generate_text(model, tokenizer, max_tokens, prompt):
        prompt_tokenized = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_tokens).to("cuda")
        output_tokenized = model.generate(
            **prompt_tokenized, 
            temperature=1,       # Control the randomness of the output
            top_p=1,             # Use nucleus sampling
            top_k=50,              # Use top-k sampling
            num_return_sequences=1, 
            max_new_tokens=100
            )[0]

        input_length = len(prompt_tokenized['input_ids'][0])
        generated_text = tokenizer.decode(output_tokenized[input_length:], skip_special_tokens=True) 
        generated_text = generated_text.split('#')[0]

        return generated_text


def run_prompts(model_name, prompts_all, references, output_csv): 
    
    model_config = models[model_name]

    model = AutoModelForCausalLM.from_pretrained(
        model_config[0],    
        device_map={"": accelerator.process_index},
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config[0])   
    
    # sync GPUs and start the timer
    accelerator.wait_for_everyone()
    start=time.time()
    if accelerator.is_main_process:
        pbar=tqdm(total=len(prompts_all))    
    # divide the prompt list onto the available GPUs 
    with accelerator.split_between_processes(prompts_all) as prompts:
        # store output of generations in dict
        results=dict(outputs=[], num_tokens=0)
        #print(len(prompts))
        #model = model.to(device)
        # have each GPU do inference, prompt by prompt
        for prompt in prompts:
            result = generate_text(model, tokenizer, model_config[1], prompt)
            #generated_text = generated_text.split('#')[0]

            results["outputs"].append(result)
            results["num_tokens"] += len(result)
           
            time.sleep(0.1)
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                pbar.update( accelerator.num_processes )

        results=[ results ] # transform to list, otherwise gather_object() will not collect correctly

    # collect results from all the GPUs
    results_gathered=gather_object(results)

    if accelerator.is_main_process:
        timediff=time.time()-start
        num_tokens=sum([r["num_tokens"] for r in results_gathered ])

        df = pd.DataFrame(results_gathered[0]['outputs'])
        df.to_csv(output_csv, index=False)
        print(f"{output_csv}, tokens/sec: {num_tokens//timediff}, time {timediff}, total tokens {num_tokens}, total prompts {len(prompts_all)}")
        file_path = 'EXPERIMENTS_SUMMARY.txt'
        with open(file_path, 'a') as file:
            file.write(f"{output_csv}, tokens/sec: {num_tokens//timediff}, time {timediff}, total tokens {num_tokens}, total prompts {len(prompts_all)}\n")
            measure(output_csv, timediff, references)


def run_transformer(args, df):
    if args.limit > 0:
        df = df[0:args.limit]

    prompts_all = df['prompt1'].to_numpy().flatten()
    references = df['ref'].tolist()

    output_csv = f"{args.output_dir}/{get_filename(args.input_csv)}.{args.model_name}.csv"
    if (args.model_name == "openai/chatgptxxxx"):
        #run_chatgpt(args.langmodel_name_model, prompts_all, output_file)
        exit(-1)
    else:
        run_prompts(args.model_name, prompts_all, references, output_csv)
        
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--method', type=str, default="icl-top")
    parser.add_argument('--output_dir', type=str, default='/home/ali.lawati/mol-incontext/output')
    parser.add_argument('--model_name', type=str, default="gpt-j-6b") 
    parser.add_argument('--input_csv', type=str, default='/home/ali.lawati/gnn-incontext/input/cosql-icl-top-2.csv')
    parser.add_argument('--n_clusters', type=int, default=5)
    parser.add_argument('--limit', type=int, default=-1) 
    
    args = parser.parse_args() 
    print("Manual Input CSV")
    df = pd.read_csv(args.input_csv)
    prompts = df['prompt1'].values
    ref = df['ref'].values
    #run_transformer(args, df)
    output_csv = f"{args.output_dir}/{get_filename(args.input_csv)}.{args.model_name}.csv"
    measure(output_csv, 0, ref)
