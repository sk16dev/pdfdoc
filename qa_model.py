from transformers import T5Tokenizer, T5ForConditionalGeneration

class QAModel:
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')

    def get_answer(self, question, context):
        input_text = f"question: {question}  context: {context}"
        inputs = self.tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)
        
        outputs = self.model.generate(inputs, max_length=150, num_beams=2, early_stopping=True)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return answer
