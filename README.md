# Geography-Aware Large Language Models for Next POI Recommendation





## Install

1. Clone this repository to your local machine:

```bash
git clone https://github.com/hugh2009hugh/GA-LLM.git
cd GA-LLM
```

2. Install the environment by running:

```
conda env create -f environment.yml
```

3. Download the model from: [https://huggingface.co/Yukang/Llama-2-7b-longlora-32k-ft](https://huggingface.co/Yukang/Llama-2-7b-longlora-32k-ft), Save the downloaded model to the appropriate directory (e.g., models/).





## Data Preparation and Processing

1. **Download the raw datasets** from [datasets](https://www.dropbox.com/scl/fi/teo5pn8t296joue5c8pim/datasets.zip?rlkey=xvcgtdd9vlycep3nw3k17lfae&st=qd21069y&dl=0).

2. **Copy the contents** of the extracted `dataset` folder to the `GA-LLM/data-preprocess/data` directory, and then execute the `data-preprocess/generate_ca_raw.py` script.

3. **Run the `data-preprocess/run.py` script** with the following required argument:

   ```
   python data-preprocess/run.py -f best_conf/ca.yml
   ```

4. **Run the `data-preprocess/traj_qk.py` script** with the following required argument:

   ```
   python data-preprocess/traj_qk.py -dataset_name ca
   ```

5. **Run the `data-preprocess/process_projector_gps_stage1.py` script** with the following required arguments:

   ```
   python data-preprocess/process_projector_gps_stage1.py -dataset_name ca -use_sim False
   ```

6. **Run the `data-preprocess/process_llm_gps.py` script** with the following required arguments:

   ```
   python data-preprocess/process_llm_gps.py -dataset_name ca -use_sim False
   ```



## Main Performance

*(Example commands for GCIM execution. Adjust parameters according to your environment)*



### train

**Stage 1: GCIM Module Training**

```bash
CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 --master_port=29501 finetune_GCIM.py  \
--model_name_or_path model_llama2-7b-longlora/ \
--bf16 True \
--lora_r 256 \
--lora_alpha 512 \
--output_dir outputs/ca/stage1_r256_alpha512_GCIM \
--gps_mapping_path /share/home/liuzh368/demo/LLM4POI/datasets/ca/preprocessed/ca_gps_mapping.csv \
--model_max_length 32768 \
--use_flash_attn True \
--data_path /share/home/liuzh368/demo/LLM4POI/datasets/ca/preprocessed/train_qa_pairs_kqt_200items_ca_projector_gps_without_sim_stage1.json \
--low_rank_training True \
--num_train_epochs 3  \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2  \
--gradient_accumulation_steps 1  \
--evaluation_strategy "no"  \
--save_strategy "steps"  \
--save_steps 1000  \
--save_total_limit 2  \
--learning_rate 2e-5  \
--weight_decay 0.0  \
--warmup_steps 20  \
--lr_scheduler_type "constant_with_warmup"  \
--logging_steps 1  \
--deepspeed "ds_configs/stage2.json"  \
--tf32 True > runs/ca/train_stage1_r256_alpha512_GCIM.txt 2>&1 &
```

**Stage 2: LLM Fine-tuning**

```bash
CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 --master_port=29501 finetune_LLM_with_GCIM.py \
--model_name_or_path /share/home/liuzh368/demo/LLM4POI/model_llama2-7b-longlora \
--bf16 True \
--output_dir outputs/ca/llm_GCIM_r256_alpha512 \
--model_max_length 32768 \
--use_flash_attn True \
--data_path /share/home/liuzh368/demo/LLM4POI/datasets/ca/preprocessed/train_qa_pairs_kqt_200items_ca_llm_without_sim_gps.json \
--gps_mapping_path /share/home/liuzh368/demo/LLM4POI/datasets/ca/preprocessed/ca_gps_mapping.csv \
--geoencoder_path /share/home/liuzh368/demo/LLM4POI/outputs/ca/stage1_r256_alpha512_GCIM/109122/geo_encoder_merged.pth \
--low_rank_training True \
--num_train_epochs 3 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 1000 \
--save_total_limit 2 \
--learning_rate 2e-5 \
--weight_decay 0.0 \
--warmup_steps 20 \
--lr_scheduler_type "constant_with_warmup" \
--logging_steps 1 \
--deepspeed "ds_configs/stage2.json" \
--tf32 True > runs/ca/train_llm_GCIM_r256_alpha512.txt 2>&1 &
```



### test

**Checkpoint Path**：`outputs/ca/llm_GCIM_r256_alpha512/checkpoint-109122`

1. **Convert Checkpoint**：

```bash
python outputs/ca/llm_GCIM_r256_alpha512/checkpoint-109122/zero_to_fp32.py outputs/ca/llm_GCIM_r256_alpha512/checkpoint-109122 outputs/ca/llm_GCIM_r256_alpha512/checkpoint-109122/pytorch_model.bin
```

2. **Run Evaluation**：

```bash
CUDA_VISIBLE_DEVICES=0 nohup python test_next_gps.py \
--batch_size 8 \
--base_model /share/home/liuzh368/demo/LLM4POI/model_llama2-7b-longlora \
--output_dir /share/home/liuzh368/demo/LLM4POI/outputs/ca/llm_GCIM_r256_alpha512/checkpoint-109122 \
--cache_dir ./cache \
--seq_len 32768 \
--context_size 32768 \
--flash_attn True \
--model_path /share/home/liuzh368/demo/LLM4POI/model_llama2-7b-longlora \
--data_path /share/home/liuzh368/demo/LLM4POI/datasets \
--dataset_name ca \
--test_file test_qa_pairs_kqt_200items_ca_llm_without_sim_gps.txt \
--test_type llm \
--gps_mapping_path /share/home/liuzh368/demo/LLM4POI/datasets/ca/preprocessed/ca_gps_mapping.csv \
--geoencoder_path /share/home/liuzh368/demo/LLM4POI/outputs/ca/stage1_r256_alpha512_GCIM/109122/geo_encoder_merged.pth \
--use_random_projector False > runs/nyc/eval_llm_gps_r256_alpha512.txt 2>&1 &
```





## Acknowledgement



This code is developed based on [LLM4POI](https://github.com/neolifer/LLM4POI?tab=readme-ov-file) and [LongLoRA](https://github.com/dvlab-research/LongLoRA?tab=readme-ov-file).


### Appendix



### Hyperparameter study of GA-LLM (NYC & CA Dataset)
Regarding hierarchical n-grams, we conducted ablation experiments by varying the n-gram depth to examine its impact under different dataset densities. We observed that performance improves as $n$ increases from small values, indicating that deeper hierarchical encoding better captures coarse-to-fine spatial dependencies. However, beyond a certain point, overly deep hierarchies introduce redundancy and noise, leading to diminished returns.

Across two datasets, we found that $n = 6$ consistently achieves the best performance, providing a good balance between spatial expressiveness and robustness. This effect is particularly pronounced in sparse datasets, where multi-level abstraction is crucial for generalization. We have added these results and corresponding analysis to the appendix in the GitHub repository.

We evaluated multiple values of $L$ across two datasets and observed that $L = 25$ provides the best overall trade-off between spatial resolution and robustness.

| **Parameter** | **NYC Acc@1** | **NYC Acc@5** | **NYC MRR@5** | **CA Acc@1** | **CA Acc@5** | **CA MRR@5** |
|---------------|---------------|---------------|---------------|--------------|--------------|--------------|
| **n-gram**    |               |               |               |              |              |              |
| n = 4         | 0.3476        | 0.5249        | 0.4107        | 0.2198       | 0.3986       | 0.2924       |
| n = 5         | 0.3794        | 0.5982        | 0.4451        | 0.2473       | 0.4327       | 0.3185       |
| n = 6         | 0.3988        | 0.6337        | 0.4663        | 0.2566       | 0.4614       | 0.3340       |
| n = 7         | 0.3763        | 0.5894        | 0.4386        | 0.2371       | 0.4213       | 0.3110       |
| **L**         |               |               |               |              |              |              |
| L = 24        | 0.3782        | 0.5965        | 0.4438        | 0.2423       | 0.4347       | 0.3191       |
| L = 25        | 0.3988        | 0.6337        | 0.4663        | 0.2566       | 0.4614       | 0.3340       |
| L = 26        | 0.3734        | 0.5851        | 0.4357        | 0.2408       | 0.4223       | 0.3116       |
| L = 30        | 0.3546        | 0.5303        | 0.4220        | 0.2219       | 0.4012       | 0.2957       |

*Table caption: Hyperparameter study of GA-LLM on Acc@1, Acc@5 and MRR@5 performance NYC dataset across different n-gram and L. (Label: ablation_study)*




### Performance comparison between different LLM backbone size (Qwen2.5-3B vs Llama2-7B)
For LLM backbone size，we evaluated smaller but more effective LLM backbone (Qwen 2.5-3B) in our framework and observed a consistent performance increase compared to the default backbone. Meanwhile, even with smaller LLMs, our approach still outperforms existing baselines, indicating that the proposed architecture does not rely on a specific large backbone to be effective.
Due to hardware and resource constraints, we were unable to fully fine-tune significantly larger LLM backbones within the revision cycle. However, the observed trend across smaller models suggests that increasing model capacity is likely to further improve recommendation performance, which we acknowledge as an important direction for future work.


| **Model**                | **NYC Acc@1** | **NYC ACC@5** | **NYC MRR@5** |
|--------------------------|---------------|---------------|---------------|
| **LLM Backbone Comparison** |               |               |               |
| GA-LLM (7B)              | 0.3988        | 0.6337        | 0.4663        |
| GA-LLM (3B)              | 0.4070        | 0.6448        | 0.4994        |
| **Fusion with Other Models** |              |               |               |
| *MTNet*                  | 0.2620        | 0.5381        | 0.3855        |
| GA-LLM-*MTNet*           | 0.3988        | 0.6337        | 0.4663        |
| *STHGCN*                 | 0.2734        | 0.5361        | 0.3915        |
| GA-LLM-*STHGCN*          | 0.3950        | 0.6256        | 0.4558        |
| *ROTAN*                  | 0.3106        | 0.5281        | 0.4104        |
| GA-LLM-*ROTAN*           | 0.3921        | 0.6162        | 0.4493        |

*Table caption: Performance comparison between different LLM backbone size, Qwen2.5-3B and our Llama2-7B.*  
*Table label: token_comparison*








































