# IL-NER

- This annotated corpora and models have been developed under the Bhashini project funded by Ministry of Electronics and Information Technology (MeitY), Government of India. We thank MeitY for funding this work. 

- This dataset and models are licensed under Creative Commons Attribution 4.0 (CC-BY-4.0) license. The details of the dataset are given below. This dataset was developed by three partnering institutes, IIIT Hyderabad, CDAC Noida, and IIIT Bhubaneshwar. 

| Language | Train | Test | Dev |
|----------|-------|------|-----|
| Hindi    | 11076 | 1389 | 1389|
| Urdu     | 8720  | 1096 | 1094|
| Odia     | 12109 | 1519 | 1517|
| Telugu   | 2993  | 384  | 384 |
- The models can be downloaded from the below huggingface repository under the Creative Commons Attribution 4.0 International (CC-BY-4.0). 
  1. Urdu Model - https://huggingface.co/Sankalp-Bahad/Monolingual-Urdu-NER-Model
  2. Odia Model - https://huggingface.co/Sankalp-Bahad/Monolingual-Odia-NER-Model
  3. Odia Model - https://huggingface.co/Sankalp-Bahad/Monolingual-Telugu-NER-Model
  4. Hindi Model - https://huggingface.co/Sankalp-Bahad/Monolingual-Hindi-NER-Model
  5. Multilingual Model - https://huggingface.co/Sankalp-Bahad/Multilingual-NER-Model
- The NER models have been developed by fine-tuning the XLM-Roberta-Base model on the annotated datasets. NER is modeled as a token classification task where a Softmax classifier is applied on the pooled layer of XLM-Roberta-base.
  - The following hyperparameters are used during training:
      * learning_rate: 5e-05
      * train_batch_size: 4
      * eval_batch_size: 4
      * optimizer: Adam
      * lr_scheduler_type: linear
      * num_epochs: 10.0

- The models have been benchmarked on Hi-NER, Naamapadam datasets and Dev, Test datasets in this annotated data.

- Package Versions

    * Transformers 4.38.2
    * Pytorch 1.9.1
    * Datasets 2.14.6
    * Tokenizers 0.15.0

- To use this dataset, cite the paper as
```
    @inproceedings{bahad-etal-2024-fine,
    title = "Fine-tuning Pre-trained Named Entity Recognition Models For Indian Languages",
    author = "Bahad, Sankalp  and 
    Mishra, Pruthwik  and
    Krishnamurthy, Parameswari  and
    Sharma, Dipti",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 4: Student Research Workshop)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-srw.9",
    doi = "10.18653/v1/2024.naacl-srw.9",
    pages = "75--82",
    }
