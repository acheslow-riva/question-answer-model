import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

model_name = "deepset/roberta-base-squad2" # Model for QA
# model_name = "sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english" # Tiny, untrained model for sentiment analysis

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
text = r"""
🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
TensorFlow 2.0 and PyTorch.
"""
questions = [
    "How many pretrained models are there available in 🤗 Transformers?",
    "What is it that does 🤗 Transformers provide?",
    "🤗 Transformers provides interoperability between which frameworks?",
]
max_length = 0
for question in questions:
    inputs = tokenizer(question, text, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"].tolist()[0]
    
    # Get longest question to test out saved model. We can pad shorter answers. Probably better than trimming longer ones
    l = len(input_ids)
    if l > max_length:
        longest = (inputs['input_ids'], inputs['attention_mask'])
        max_length = l

    decoded_inputs = tokenizer.decode(inputs["input_ids"][0])
    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    outputs = model(**inputs)

    answer_start_scores = outputs[0]
    answer_end_scores = outputs[1]
    answer_start = torch.argmax(
        answer_start_scores
    )  # Get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    print(f"Question: {question}")
    print(f"Answer: {answer}")


# Creating the trace
# model.eval()
# traced_model = torch.jit.trace(model, longest)
# torch.jit.save(traced_model, "traced_roberta.pt")

# from transformers import pipeline
# nlp = pipeline(model=model, tokenizer=tokenizer)
# context = r"""
# Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a
# question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune
# a model on a SQuAD task, you may leverage the examples/question-answering/run_squad.py script.
# """
# result = nlp(question="What is extractive question answering?", context=context)

