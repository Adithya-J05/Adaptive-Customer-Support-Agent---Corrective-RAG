import argparse
import os
import sys

from config import load_env
from ingestion import build_retriever
from app import build_graph


def main():
    parser = argparse.ArgumentParser(description="Cheetah RAG Agent")
    parser.add_argument("question", nargs="?", help="Question to ask the agent")
    parser.add_argument(
        "--doc", "-d",
        default="./docs",
        help="Path to a PDF, MD, TXT file or a directory of documents (default: ./docs)",
    )
    parser.add_argument(
        "--rebuild", "-r",
        action="store_true",
        help="Force rebuild the vector store from scratch",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start an interactive Q&A session",
    )
    args = parser.parse_args()

    load_env()

    source = args.doc
    if not os.path.exists(source):
        print(f"Document path not found: {source}")
        sys.exit(1)

    print("Initializing agent (vectorizing documents)...")
    retriever = build_retriever(source, force_rebuild=args.rebuild)
    agent = build_graph(retriever)
    print("Agent ready.\n")

    if args.interactive or not args.question:
        print("Interactive mode. Type 'exit' to quit.\n")
        while True:
            q = input("> ")
            if q.lower() in ("exit", "quit"):
                break
            if q.strip():
                result = agent.invoke({"question": q})
                print(f"\n{result['generation']}\n")
    else:
        result = agent.invoke({"question": args.question})
        print(f"\n{result['generation']}")


if __name__ == "__main__":
    main()
