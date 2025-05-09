# LongForm_Hallucination
Code for Reference-Free Hallucination Detection in Open-Domain Long-Form Generation


## Setup

### 1. Create conda environment and install dependencies
```
conda env create -f environment.yml
conda activate knowledge
```


### 2. Train the model for reference-free hallucination detection

```
cd LLaMA-Factory
llamafactory-cli train examples/train_lora/atomized_claim_classifier_sft_cot.yaml
llamafactory-cli export examples/merge_lora/atomized_claim_classifier_sft_cot_merge.yaml
```


### 3. Evaluate the trained model
```
cd Knowledge-Finetune
python ProbTrueandFalse_lora_sft.py 
```
