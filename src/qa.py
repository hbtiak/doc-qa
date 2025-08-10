from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

_qa_model = None
_qa_tokenizer = None

def load_qa_model(name: str = "distilbert-base-cased-distilled-squad"):
    global _qa_model, _qa_tokenizer
    if _qa_model is None or _qa_tokenizer is None:
        _qa_tokenizer = AutoTokenizer.from_pretrained(name)
        _qa_model = AutoModelForQuestionAnswering.from_pretrained(name)
    return _qa_model, _qa_tokenizer

def answer_question_from_context(question: str, contexts, model, tokenizer, max_len: int = 384):
    # Iterate contexts and pick the highest score answer
    best_answer = ""
    best_score = -1e9
    best_ctx = ""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for ctx in contexts:
        inputs = tokenizer.encode_plus(question, ctx, add_special_tokens=True, return_tensors="pt", truncation=True, max_length=max_len)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        # pick best span
        start = torch.argmax(start_scores)
        end = torch.argmax(end_scores)
        if end >= start:
            answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][start:end+1]))
            score = start_scores[0][start].item() + end_scores[0][end].item()
            if score > best_score and answer.strip():
                best_score = score
                best_answer = answer
                best_ctx = ctx
    if not best_answer:
        best_answer = "No confident answer found."
    return best_answer, best_ctx
