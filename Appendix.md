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

