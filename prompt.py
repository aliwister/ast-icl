from accelerate import Accelerator
from accelerate.utils import gather_object
import pandas as pd
import torch, time
from tqdm import tqdm
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTJForCausalLM
from multi_measure import multi_measure

#from util.chatgpt import run_chatgpt
import os
from pathlib import Path

accelerator = Accelerator()
models = {
    "llama-3-8B": ("meta-llama/Meta-Llama-3-8B-Instruct", 8192),
    "zephy7B": ("HuggingFaceH4/zephyr-7b-beta", 32768),
    "gemma-7B": ("google/gemma-7b", 8192),
    "Qwen-2-7B": ("Qwen/Qwen2-7B-Instruct", 32768),
    "mistral-7B": ("mistralai/Mistral-7B-v0.1", 2048),
    "openchat-8B": ("openchat/openchat-3.6-8b-20240522", 8192),
    "WizardLM-2-7B": ("lucyknada/microsoft_WizardLM-2-7B", 32768),
    "gpt-j-6b": ("EleutherAI/gpt-j-6B", 2048),
    "code-llama": ("codellama/CodeLlama-7b-hf", 2048)
}

def get_filename(input_csv):
    # Get the file name without directory and extension
    filename = Path(input_csv).stem
    return filename

def prepare_prompts(prompts, tokenizer, batch_size=4):
    batches=[prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
    batches_tok=[]
    tokenizer.padding_side="left"     
    for prompt_batch in batches:
        batches_tok.append(
            tokenizer(
                prompt_batch, 
                return_tensors="pt", 
                padding='longest', 
                truncation=False, 
                pad_to_multiple_of=8,
                add_special_tokens=False).to("cuda") 
            )
    tokenizer.padding_side="right"
    return batches_tok

BATCH_SIZE = 4
def main(args):
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    accelerator = Accelerator()

    df = pd.read_csv(args.input_csv)
    if  args.limit > 0:
        df = df[0:args.limit]
    prompts_all = df['prompt'].tolist()
    #references = df['ref'].tolist()

    # load a base model and tokenizer
    model_path=models[args.model_name][0]

    if (args.model_name == "gpt-j-6b"):
        model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B",
                                                 revision="float16", low_cpu_mem_usage=True).cuda()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,    
            device_map={"": accelerator.process_index},
            torch_dtype=torch.bfloat16,
            cache_dir=f"./llm/{args.model_name}"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path) #, cache_dir=f"../data/{model_name}")   
    tokenizer.pad_token = tokenizer.eos_token

    # sync GPUs and start the timer
    accelerator.wait_for_everyone()    
    start=time.time()
    if accelerator.is_main_process:
        pbar=tqdm(total=len(prompts_all))    
    # divide the prompt list onto the available GPUs 
    with accelerator.split_between_processes(prompts_all) as prompts:
        print (len(prompts))
        results=dict(outputs=[], num_tokens=0)

        # have each GPU do inference in batches
        prompt_batches=prepare_prompts(prompts, tokenizer, batch_size=BATCH_SIZE)
        for prompts_tokenized in prompt_batches:

            outputs_tokenized=model.generate(
                **prompts_tokenized, 
                temperature=1,       # Control the randomness of the output
                top_p=1,             # Use nucleus sampling
                top_k=50,              # Use top-k sampling
                num_return_sequences=1,
                max_new_tokens=350,
                pad_token_id=tokenizer.eos_token_id)
            # remove prompt from gen. tokens
            outputs_tokenized=[ tok_out[len(tok_in):] 
                for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized) ] 

            # count and decode gen. tokens 
            num_tokens=sum([ len(t) for t in outputs_tokenized ])
            #pdb.set_trace()
            outputs=tokenizer.batch_decode(outputs_tokenized, skip_special_tokens=True)
            processed_outputs = [output.split('###')[0].replace('"', '').strip() for output in outputs]
            results["outputs"].extend(processed_outputs)
            results["num_tokens"] += num_tokens
            time.sleep(0.1)
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                pbar.update( accelerator.num_processes * BATCH_SIZE )

        results=[ results ] # transform to list, otherwise gather_object() will not collect correctly

    results_gathered=gather_object(results)

    if accelerator.is_main_process:
        #print(len(results_gathered))
        timediff=time.time()-start
        num_tokens=sum([r["num_tokens"] for r in results_gathered ])
        #results_all = [item.split("#")[0] for r in results_gathered for item in r["outputs"]]
        results_all = [item for r in results_gathered for item in r["outputs"]]

        print(f"tokens/sec: {num_tokens//timediff}, time elapsed: {timediff}, num_tokens {num_tokens}")

        # Specify the path to the file
        file_path = args.input_csv
        parent_directory = os.path.dirname(file_path)
        dataset_model = os.path.basename(os.path.dirname(parent_directory))

        output_csv = f"{args.output_dir}/{dataset_model}/{get_filename(args.input_csv)}.{args.model_name}.csv"
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df = pd.DataFrame({
            'pred': results_all,
            #'ref': references
            })
        df.to_csv(output_csv, index=False)
        print(f"{output_csv}, tokens/sec: {num_tokens//timediff}, time {timediff}, total tokens {num_tokens}, total prompts {len(prompts_all)}")
        file_path = 'EXPERIMENTS_SUMMARY.txt'
        with open(file_path, 'a') as file:
            file.write(f"{output_csv}, tokens/sec: {num_tokens//timediff}, time {timediff}, total tokens {num_tokens}, total prompts {len(prompts_all)}\n")
            multi_measure(args.dataset, output_csv, timediff, args.limit)
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--method', type=str, default="icl-top")
    parser.add_argument('--output_dir', type=str, default=f'{Path.home()}/output/')
    parser.add_argument('--model_name', type=str, default="gpt-j-6b") 
    parser.add_argument('--input_csv', type=str, default=f'{Path.home()}/input/*.csv')
    parser.add_argument('--dataset', type=str, default='spider')
    parser.add_argument('--n_clusters', type=int, default=5)
    parser.add_argument('--limit', type=int, default=-1) 
    
    args = parser.parse_args() 
    main(args)
