from pdf_processor import extract_text_from_pdf
from qa_model import QAModel

def main():
    pdf_path = 'llm.pdf'
    
    # Extract text from the PDF
    context = extract_text_from_pdf(pdf_path)
    
    # Initialize the QA model
    qa_model = QAModel()
    
    while True:
        # Get the question from the user
        question = input("Ask the model: ")
        if question.lower() == 'exit':
            break
        
        # Get the answer from the QA model
        answer = qa_model.get_answer(question, context)
        
        # Print the question and answer
        print(f"Question: {question}")
        print(f"Answer: {answer}\n")

if __name__ == "__main__":
    main()
