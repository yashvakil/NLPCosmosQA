# NLPCosmosQA

### Progress
1. Read [CosmosQA](https://arxiv.org/abs/1909.00277) paper to get deep understanding of paper concepts.
2. Run [Wilburone's model](https://github.com/samkit27/NLPCosmosQA/tree/master/wilburone/cosmosqa-master) as a baseline model
3. Read Papers of DistilBERT and RoBERTa and implemented them.
4. Run DistilBERT model sucessfully.
5. Implemented Roberta Large Multiway attention which gave 79.22% results with text simillarity module to extract important knowledge from the given context.
6. Error Analysis is distributed among team members and plan to seperate observations accroding to error analysis type provided in basic analysis section.
7. [Results](https://github.com/yashvakil/NLPCosmosQA/tree/master/results) and [Analysis](https://github.com/yashvakil/NLPCosmosQA/tree/master/analysis) included.
8. Download [trained](https://drive.google.com/file/d/1U95I1Msy153N33pn40itfutGFJSL6wBS/view?usp=sharing) model and evaluate it using
python run_roberta.py --task_name "commonsenseqa" --do_eval --load_model --do_lower_case --roberta_model roberta-large --data_dir data/ --max_seq_length 220  --gradient_accumulation_steps=10  --output_dir output_path_bin_file --seed 7 --fp16



### Contribution Info 

#### Samkit Shah : 
1. Read paper of [CosmosQA](https://arxiv.org/abs/1909.00277) , [BERT](https://arxiv.org/abs/1810.04805) to get deep understanding of paper concepts
2. Get information of basic methods used in comprehension and Question answering based problems and stated it with error analysis in basic analysis section.
3. Run [Wilburone's model](https://github.com/samkit27/NLPCosmosQA/tree/master/wilburone/cosmosqa-master) as a baseline model in the local system as well as on [Google Colab](https://colab.research.google.com/drive/1hJrzJutH7bKQ9r-wVQ9BqVxiLUE9qZOQ) by trying various batchsize, learning rates and epochs.
4. Modified the code and Run Distilbert model and saved results in Result folder.
5. Now understand the logic behind using multiway attention model provided in [CosmosQA](https://arxiv.org/abs/1909.00277) and plan to implement it with more advanced models.
6. Worked to understand Roberta Multiway attention and trained model and Extract Analysis where both bert and Roberta goes wrong .
7.  Worked on Knowldege infusion via finetune BERT on socialiqa dataset and then run cosmosqa on top of it.

#### Avatar Jaykrushna :
1. Read the paper of [CosmosQA](https://arxiv.org/abs/1909.00277) to learn more about the dataset and [BERT](https://arxiv.org/abs/1810.04805) to get a ground level understanding of the BERT model.
2. Ran the Bert base code given in the project resources to understand the implementation of BERT.  
3. Explored a new possibility of Implementing K adapters in the Bert Pretrained Model by reading the K-Adapters Model.
4. Tried to run the DistilBert code on local system but due to system limitations could not get the code to work.
5. Ran the DistilBERT code on Google Colab with different settings.
6. Trying to implement RoBerta on Google Colab.
7. Implemented the RoBERTa-large model on google colab.
8. Researched about generative models for implementation but did not move forward due to complications.
9. Read about text fooler and tried to make it work on implementation level but did not succeed.
10. Implemented Query based Text Similarity to summarize context which led to minor improvement in the performance of model and tried to implement complete Text Summarization of the context. 

#### Yash Vakil :
1. Read paper of [CosmosQA](https://arxiv.org/abs/1909.00277) , [BERT](https://arxiv.org/abs/1810.04805) to get an overall understanding of the base model of our implementation.
2. Ran the DistilBERT code on local system.
3. Read the [LSTMJump](https://arxiv.org/pdf/1704.06877.pdf), and [SkimRNN](https://arxiv.org/abs/1711.02085) to find ways to augment the information used as the input to the BERT model.
4. Formualized idea to make changes in DistilBERT using the analysis of Base BERT model to reach conclusion faster.
5. Currently conducting Error Analysis on the output of the BERT and DistilBERT model .
6. Planning to implement multiway attention model of [CosmosQA](https://arxiv.org/abs/1909.00277) for DistilBERT
7. Looking to make changes in the model apart from data augmentation to increase accuracy.
8. Completed my part of the the Error Analysis on Roberta-large's predictions.
9. Implemented Text Similarity and Summarization code and analyzed a threshold value for context truncation.

#### Pushparajsinh Zala:
1. Read following papers
    - CosmosQA
    - BERT, RoBERTa
    - Natural Language QA Approaches using Reasoning with External Knowledge (Chitta Baral, Pratyay Banerjee, Kuntal Kumar Pal, Arindam Mitra)
    - ReClor: A Reading Comprehension Dataset Requiring Logical Reasoning
2. Other preparations
    - PyTorch basics
    - huggingface: transformers, tokenizers code walkthrough
3. Understanding dataset and project setup for base model implementation(wilburOne/CosmosQA)
4. Run Wilburone repo base implementation [Wilburone's model](https://github.com/samkit27/NLPCosmosQA/tree/master/wilburone/cosmosqa-master)
    - Code changes to remove various error & adapting local machine
    - First successful Run after many code changes
    - Tried different hyper-parameters for optimized run selection for local configuration we have(limited GPU)
5. Error analysis done for BERT vs DistilBERT base model run
6. Ideation for various other method to increase model knowledge. some papers for ideas:
    - Structural Scaffolds for Citation Intent Classification in Scientific Publications - https://arxiv.org/abs/1904.01608 (here authours are adding two extra tasks(scaffolds) apart from main task on top of BiLSTM-Attn. similar techniqes can be applied here with adding more features on top of BERT.)
7. Error analysis done for BERT vs DistilBERT base model run
8. Model implementations with changes


### Basic analysis and Literature Review 
COSMOS QA: Machine Reading Comprehension with Contextual Commonsense Reasoning

Model and Approach Analysis:
There are two types of Baseline models. Reading Comprehensions and modifications on it and pretrained models, which are used as a general approaches for this problems.
1.	Sliding Windows
2.	Stanford Active Reader
3.	Gated Attention Reader
4.	Co Matching
5.	Common Sense RC
6.	GPT-FT
7.	GPT-FT
8.	BERT-FT
9.	DMCN

---

In Reading comprehension approach, semantic correlatedness is important factor to choose an answer from given answers where it infers from the given contextual paragraph about semantic correlations. 

In COSMOS-QA dataset, it contains 83% of answers which is not in reading comprehension context so semantic relatedness factor is not important here to infer the correct answer as it requires common sense to infer answer, while pretrained methods improves the scenarios further while apply finetuning on BERT. Further, more accurate results can be achieved by performing attention and finetuning on context paragraph, Question and Answer.

Ablation is also one of the important parts of the study where ablation of question didn't affect much in prediction result while ablation of question and context affects significantly on the result and got drop in accuracy

---

#### Knowledge Transfer Learning ###

Knowledge transfer and finetuning on various datasets of same kind help a lot to improve inference. Authors have proposed two datasets RACE and SWAG which contains multiple choice questions, and fine tuning of BERT on both Race + SWAG and Cosmos. BERT-FT on SWAG provides good result while including with BERT FT RACE+SWAG which gives 68.7 percent test accuracy

---

#### Error Analysis

* Complex context understanding 

It require cross sentence interpretation and reasoning and need to combine the context information to infer the real answer. Here model need to learn from complex context analysis
to infer the choice A. 


*	Inconsistent with Human Common Sense

In 33% of the errors, the model mistakenly select the choice which is not consistent with human common sense. So here answer might be right but not match with human commonsense.

* Multi-turn Common Sense Inference

19% errors due to this where there are multiple inferences present in sentence where model need to choose with proper inference.

*	Unanswerable Questions

14% of the errors are from unanswerable questions so model cannot infer from given multiple answers.


#### Generative models to infer answers ###

Fine tuning on GPT and GPT-2 can provide better understanding and provide more accurate answer which can be consider as one of the different approach.
