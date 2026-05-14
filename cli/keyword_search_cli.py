import argparse
from pathlib import Path
import math


from lib.indexer import InvertedIndex
from utils.load_data import load_movies


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build and save index")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency")
    idf_parser.add_argument("term", type=str, help="Term")

    tf_idf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score")
    tf_idf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_idf_parser.add_argument("term", type=str, help="Term")

    args = parser.parse_args()

    match args.command:
        case "search":
            invIdx = InvertedIndex()
            invIdx.load(path=CACHE_DIR)

            search_query = args.query
            results = []
            for term in search_query.lower().split():
                term_results = invIdx.get_documents(term)
                results.extend(term_results)

                if len(results) >= 5:
                    break

            for ids in results[:5]:
                print(invIdx.docmap[ids])
            pass

        case "build":
            invIdx = InvertedIndex()
            movies = load_movies(DATA_DIR / "movies.json")

            invIdx.build(movies=movies)
            invIdx.save(path=CACHE_DIR)
            pass

        case "tf":
            doc_id = args.doc_id
            term = args.term

            invIdx = InvertedIndex()
            invIdx.load(path=CACHE_DIR)

            tf = invIdx.get_tf(doc_id=doc_id, term=term)
            if tf == 0:
                print("0")
            else:
                print(f"Term Frequency of '{term}' in document {doc_id}: {tf}")

        case "idf":
            term = args.term
            invIdx = InvertedIndex()
            invIdx.load(path=CACHE_DIR)
            idf = invIdx.get_idf(term=term)

            print(f"Inverse Document Frequency of '{term}': {idf:.2f}")
            pass

        case "tfidf":
            doc_id = args.doc_id
            term = args.term
            invIdx = InvertedIndex()
            invIdx.load(path=CACHE_DIR)
            tf = invIdx.get_tf(doc_id=doc_id, term=term)
            idf = invIdx.get_idf(term=term)
            tf_idf = tf * idf
            print(f"TF-IDF score of '{term}' in document {doc_id}: {tf_idf:.2f}")
            pass

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
