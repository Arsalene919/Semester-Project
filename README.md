# Semester-Project
Analyzing Causal Relations in Socio-political Events Extracted from Text: A Comparative Study of Large Language Models and Pretrained Language Models

We used this repository to try T5 model: https://github.com/idiap/cncsharedtask/tree/main

We did two experiments for subtask2, one with the original dataset and the second one with the augmented dataset, for data augmentation we used prompt engineering technique using the LLM Zephyr-7B

#results with the original dataset
Recall:0.5364741641337386

Precision:0.5487219935105742

F1:0.5399806175939386

Accuracy:0.49040043499731256

Number:409

Cause_Recall:0.5060240963855422

Cause_Precision:0.5060240963855422

Cause_F1:0.5060240963855422

Cause_Number:249

Effect_Recall:0.46987951807228917

Effect_Precision:0.46987951807228917

Effect_F1:0.46987951807228917

Effect_Number:249

Signal_Recall:0.6875

Signal_Precision:0.7378691983122362

Signal_F1:0.7019202898550724

Signal_Number:160

#results with the augmented data

Recall:0.5319148936170213

Precision:0.5403125243550775

F1:0.5341058763931105

Accuracy:0.4937153814134951

Number:409

Cause_Recall:0.5060240963855422

Cause_Precision:0.5060240963855422

Cause_F1:0.5060240963855422

Cause_Number:249

Effect_Recall:0.46987951807228917

Effect_Precision:0.46987951807228917

Effect_F1:0.46987951807228917

Effect_Number:249

Signal_Recall:0.66875

Signal_Precision:0.7032852564102564

Signal_F1:0.6777604166666666

Signal_Number:160

#Explanation of the metrics

Overall Metrics: These metrics aggregate the model's performance across all types of spans, including cause, effect, and signal spans.

Cause Metrics: These metrics specifically measure the model’s ability to identify cause spans.

Effect Metrics: These metrics specifically measure the model’s ability to identify effect spans.

Signal Metrics: These metrics specifically measure the model’s ability to identify signal spans.



