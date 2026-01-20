# Constraint Satisfaction and Explicit Termination in Diffusion Language Models.
The goal of the code in this repository is to analyze different models on 50 prompts ranging across a variety of tasks and score on how proficiently each model maintains required structural constraints.


## To run this code:
Due to the fact that wedlm utilizes flash attention and the other models do not, I have broken the code up to properlyaccount for this. You must run the flash-attention and non-flash-attention models in seperate envs. So to be clear, if the code has --target noflash then you run it in a non-flash-attention environment, if it has --target wedlm then you run it in the flash-attention environment.


To import these environments you can use:


    conda env create -f environment.yml


Where environment.yml is either the flash or non-flash environment file in these repository.


## Full command by command code:


### All Models Excluding 8B Parameter DAG Models:


python benchmark.py --target noflash --prompts prompts.jsonl --out results.jsonl --append


python benchmark.py --target wedlm --prompts prompts.jsonl --out results.jsonl --append


### To Run 8B Parameter DAG:

- AR to WeDLM:
    

    python benchmark.py --target noflash --dag_mode make_buffer --dag_dir dag_buffers --buf dag_buffers/ar8b_to_wedlm8b_buf.jsonl


    python benchmark.py --target wedlm --dag_mode consume_buffer --buf dag_buffers/ar8b_to_wedlm8b_buf.jsonl --out results_wedlm_dag.jsonl


- WeDLM to AR:


    python benchmark.py --target wedlm --dag_mode make_buffer --dag_dir dag_buffers --buf dag_buffers/wedlm8b_to_ar8b_buf.jsonl


    python benchmark.py --target noflash --dag_mode consume_buffer --buf dag_buffers/wedlm8b_to_ar8b_buf.jsonl --out results_ar_dag.jsonl
