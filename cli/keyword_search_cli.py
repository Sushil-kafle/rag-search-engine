import argparse
from pathlib import Path


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

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
