from dotenv import load_dotenv
from app.loader import load_and_split_pdf
from app.vectorstore import create_vectorstore
from app.qa_chain_hf import create_qa_chain

def main():
    load_dotenv()

    documents = load_and_split_pdf("sample_policy.pdf")
    vectorstore = create_vectorstore(documents)
    qa_chain = create_qa_chain(vectorstore)

    while True:
        query = input("\nAsk something (or type 'exit'): ")
        if query.lower() == "exit":
            break
        print("\nAnswer:", qa_chain.run(query))

if __name__ == "__main__":
    main()
