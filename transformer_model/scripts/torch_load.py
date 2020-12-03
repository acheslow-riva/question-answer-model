import torch
from transformers import AutoTokenizer

loaded_model = torch.jit.load("traced_roberta.pt")
loaded_model.eval()

model_name = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

question = "How many languages are the models available in?"
text = r"""
🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
TensorFlow 2.0 and PyTorch.
"""

inputs = tokenizer(question, text, return_tensors="pt", add_special_tokens=True)
input_ids = inputs["input_ids"].tolist()[0]

x = [inputs['input_ids'], inputs['attention_mask']]
outputs = loaded_model(*x)

answer_start_scores = outputs[0]
answer_end_scores = outputs[1]
answer_start = torch.argmax(
    answer_start_scores
)  # Get the most likely beginning of answer with the argmax of the score
answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))