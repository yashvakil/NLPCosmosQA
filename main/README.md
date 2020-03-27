# Cosmos QA: Machine Reading Comprehension with Contextual Commonsense Reasoning (EMNLP'2019)

This repository includes the source code and data for Cosmos QA.

To run code of distilbert, few changes are necessary,
1. Import model file as modelling_distil to import BertMultiwayattention model in run_multiway_att.py which is same as bert
2. Download tar file for distil bert from [here](https://drive.google.com/file/d/10gLGcxPNaNorV2hF8Msr_XF4O7N8pgN5/view?usp=sharing) and set path in modelling file to run it.

## Requirements

Python3, Pytorch1.0, tqdm, boto3, requests
