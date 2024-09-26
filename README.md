# Semantic Captioning
Benchmark Dataset and Graph-Aware Few-Shot In-Context Learning for SQL2Tex


## Train and create prompt file
```
python train.py --dataset <cosql|spider|sparc> --method <icl-top|icl-cluster|random|BM25|zero> --output_csv <output filename> --num_examples <number of demonstrations>
```

## Prompt and Measure
```
accelerate --input_csv <prompt filename> --limit <how many prompts to run> --model_name <gpt-j-6b|mistral-7B|llama-3-8B> 
```

![Model](assets/teaser.png)
