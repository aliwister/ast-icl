# Semantic Captioning
[Benchmark Dataset and Graph-Aware Few-Shot In-Context Learning for SQL2Tex](https://aclanthology.org/2025.coling-main.536/)


[<img src="assets/teaser.png" width="250"/>](assets/teaser.png)


## Train and create prompt file
```
python assemble.py --dataset <cosql|spider|sparc> --method <icl-top|icl-cluster|random|BM25|zero> --model <GCN|SAGE|GAT> --num_examples <number of demonstrations>
```
Input files will be created in the `./input` directory

## Prompt and Measure
```
accelerate launch --main_process_port <PORT>  --num_processes <NUM_GPUs> prompt.py --input_csv <assembed file path> --limit <how many prompts to run> --model_name <gpt-j-6b|mistral-7B|codellama-7b> 
```

Results will be logged in `EXPERIMENTS.txt` file




## If you find this code useful, please consider citing our work:
```
@inproceedings{al-lawati-etal-2025-semantic,
    title = "Semantic Captioning: Benchmark Dataset and Graph-Aware Few-Shot In-Context Learning for {SQL}2{T}ext",
    author = "Al Lawati, Ali  and
      Lucas, Jason  and
      Mitra, Prasenjit",
    year = "2025",
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics",
    pages = "8026--8042"
}
```