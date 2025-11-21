# Geneformer-Distilled 4M

This is a part of our seminar at the instite of statistics at LMU Munich. 

Here we are trying to distill a 4M parameter model from 10M parameter model which is the geneformer, considered one of the seminal work in network biology. We have managed to compress the model from 10M to 4M around 2.5x reduction with around 1/25th of the data and 1/100th of the training with with identical things. It was abit challenging to replicate this paper because there were a lot of things missings like 

1. Even to download files from hugging face is a mess
2. Needed to create a custom data collator due to variable length sampling $\to$ to save computing
3. Due to variable sequence length failed to use `torch.compile()` which builds a dynamic CUDA graph. 
4. Compute challenges, the data is massive with around 27 Mn rows and 500 tokens per sequences.  
5. Data was pretokenized that helped but also it was the most important part of the paper, needed to build our own collator. 
6. General Clarity Needed on what is the BERT masking strategy.
7. The dataset for the V2 models are not being provided 104 M
8. Working with a 27 Mn rows with 500 tokens amounts to 10B tokens approx, really difficult to work with it. Major optimizations in dataset.py

In general it proved difficult to replicate but not impossible. 

We have the following training metric 
![Training](notebooks/training_metrics.png) 

and the weights of the distilled models can be found in the outputs/checkpoints we are going to use the model_best.pt

To run the evaluations you need to do : 

```python
    python3 -m src.evals.quick
```

As we have been training a language model with masking we are evaluating it on 

1. Accuracy : Of the masking of the tokens
2. Perplexity : Measure of how confident the model is while predicting the masked tokens

The current numbers suggests that everything is working just we need to train our model more for more duration because currently it is only being trainined on 1.2 Mn rows instead of 27 Mn rows so it has not seen enough data. 

```bash
============================================================
Metric          | Teacher (Target)   | Student (Yours)    | Gap       
MLM Accuracy    | 0.2984             | 0.1853             | -0.1130
Perplexity      | 15.94              | 44.92              | +28.98
============================================================
```

This is an ongoing project, we will be improving it further. 

