## VulDIAC: Vulnerability Detection and Interpretation based on Augmented CFG and Causal Attention Learning

This is the repository of VulDIAC，

> Vulnerability detection in software source code is essential for ensuring system security. Recently, deep learning methods have gained significant attention in this domain, leveraging structured information extracted from source code, and employing Graph Neural Networks (GNNs) to enhance detection performance through graph representation learning. However, conventional code graph structures exhibit limitations in capturing the comprehensive semantics of source code, and the presence of spurious features may result in incorrect correlations, which undermines the robustness and explainability of vulnerability detection models. In this paper, we propose VulDIAC, a novel framework for Vulnerability Detection and Interpretation that integrates an Augmented Control Flow Graph (ACFG) and a multi-task Causal attention learning module based on Relational Graph Convolutional Networks, referred to as RGCN-CAL. The ACFG incorporates additional relational edges, such as reaching-define and dominator relationships, to better capture the control flow logic and data flow information within the code. The RGCN-CAL module emphasizes causal features while learning multi-relational graph representations. This approach enhances detection accuracy and provides fine-grained, line-level explanations. Experimental evaluations on two public datasets demonstrate that VulDIAC significantly outperforms baseline methods, achieving F1-Score improvements of 27.16% and 53.59%, respectively. Additionally, VulDIAC provides superior Top-k accuracy than LineVul in line-level vulnerability detection, further verifying its effectiveness and interpretability.

## Dataset

To investigate the effectiveness of VulDIAC, we adopt two vulnerability datasets from:

- FFmpeg+Qemu [1]: https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF
- Big-Vul [2]: https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view?usp=sharing


## Requiremnet

Our code is based on Python 3.10

- dgl==2.4.0+cu121
- networkx==3.4.2
- numpy==2.1.3
- pandas==2.2.3
- prettytable==3.12.0
- PyYAML==6.0.2
- scikit-learn==1.5.2
- torch==2.3.1+cu121
- torchaudio==2.3.1+cu121
- torchvision==0.18.1+cu121
- tqdm==4.67.1
- transformers==4.46.3

## Preprocess

Normalization

```
python ./scripts/normalization.py -i 'dataset_dir'  (file *.c)
```

Genarate joern graph

```
# get bins
python joern_graph_gen.py -i 'input_dir' -o 'output_dir' -t parse
# get cpg
python joern_graph_gen.py -i 'input_dir(bins)' -o 'output_dir' -t export -r cpg
```

Get ACFG

```
python get_cfg_graph.py -i 'input_dir' -o 'output_dir'
python get_dgl_graph_with_bert.py -i 'input_dir' -o 'output_dir'
```

Train the model

```
python vul_graph.py -i ./dgl/bigvul_cfg/ -m bigvul -d cuda:0
```

### References

[1] Yaqin Zhou, Shangqing Liu, Jingkai Siow, Xiaoning Du, and Yang Liu. 2019. Devign: Effective vulnerability identification by learning comprehensive program semantics via graph neural networks. In Advances in Neural Information Processing Systems. 10197–10207.

[2] Jiahao Fan, Yi Li, Shaohua Wang, and Tien Nguyen. 2020. A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries. In The 2020 International Conference on Mining Software Repositories (MSR). IEEE.
