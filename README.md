# Constraint Satisfaction and Explicit Termination
The goal of the code in this repository is to analyze different models on 50 prompts ranging across a variety of tasks and score on how proficiently each model maintains required structural constraints.


To do this, we utilize a static prompt segment that is appended to each task-specific prompt. This is fed to four model types: AR, diffusion, hybrid (DAG, AR to Diff and Diff to AR), and iterative-improvement (evaluated independently for each model). We do this for a family of 0.6B parameter models and a family of 8B parameter models.


## To run this code:
Due to the fact that wedlm utilizes flash attention and the other models do not, I have broken the code up to properly account for this. You must run the flash-attention and non-flash-attention models in separate envs. So to be clear, if the code has --target noflash then you run it in a non-flash-attention environment, if it has --target wedlm then you run it in the flash-attention environment.


To import these environments you can use:


    conda env create -f no-flash_environment.yml
    

    conda env create -f flash_environment.yml




After importing the environments, the  non-flash-attention environment can be run using:

    conda activate noflash


And the flash-attention environment can be ran using:

    conda activate wedlm



## Full command by command code:


### All Models Excluding the Two-Stage DAG 8B Hybrid Models:


python benchmark.py --target noflash --prompts prompts.jsonl --out results.jsonl --append


python benchmark.py --target wedlm --prompts prompts.jsonl --out results.jsonl --append


### To Run 8B Parameter DAG:

- AR to WeDLM:
    

    python benchmark.py --target noflash --dag_mode make_buffer --dag_dir dag_buffers --buf dag_buffers/ar8b_to_wedlm8b_buf.jsonl


    python benchmark.py --target wedlm --dag_mode consume_buffer --buf dag_buffers/ar8b_to_wedlm8b_buf.jsonl --out results_wedlm_dag.jsonl


- WeDLM to AR:


    python benchmark.py --target wedlm --dag_mode make_buffer --dag_dir dag_buffers --buf dag_buffers/wedlm8b_to_ar8b_buf.jsonl


    python benchmark.py --target noflash --dag_mode consume_buffer --buf dag_buffers/wedlm8b_to_ar8b_buf.jsonl --out results_ar_dag.jsonl


## After Running

After running, all data analysis can be done through the relevant Data Analysis notebook. The file results.jsonl will contain all of the model output for the prompts in prompts.jsonl for all models ran. By default, the above commands will output the 8B parameter DAG to specific output files but the data from these files can either be appended to the main results.jsonl or the commands can be changed to force all output in results.jsonl.

