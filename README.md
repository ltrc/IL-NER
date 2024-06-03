# IL-NER

- This annotated corpora has been developed under the Bhashini project funded by Ministry of Electronics and Information Technology (MeitY), Government of India. We thank MeitY for funding this work. 

- This dataset is licensed under Creative Commons Attribution 4.0 (CC-BY-4.0) license. The details of the dataset are given below. This dataset was developed by three partnering institutes, IIIT Hyderabad, CDAC Noida, and IIIT Bhubaneshwar. 

| Language | Train | Test | Dev |
|----------|-------|------|-----|
| Hindi    | 11076 | 1389 | 1389|
| Urdu     | 8720  | 1096 | 1094|
| Odia     | 12109 | 1519 | 1517|
| Telugu   | 2993  | 384  | 384 |

- The NER models have been developed by fine-tuning the XLM-Roberta-Base model on the annotated datasets. NER is modeled as a token classification task where a Softmax classifier is applied on the pooled layer of XLM-Roberta-base.
  - The following hyperparameters are used during training:
      * learning_rate: 5e-05
      * train_batch_size: 4
      * eval_batch_size: 4
      * optimizer: Adam
      * lr_scheduler_type: linear
      * num_epochs: 10.0

- Package Versions

    * Transformers 4.38.2
    * Pytorch 1.9.1
    * Datasets 2.14.6
    * Tokenizers 0.15.0

- To use this dataset, cite the paper as

      @misc{bahad2024finetuning,
            title={Fine-tuning Pre-trained Named Entity Recognition Models For Indian Languages}, 
            author={Sankalp Bahad and Pruthwik Mishra and Karunesh Arora and Rakesh Chandra Balabantaray and Dipti Misra Sharma and Parameswari Krishnamurthy},
            year={2024},
            eprint={2405.04829},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
      }
