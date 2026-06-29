from __future__ import annotations

import argparse

from agents import generate_report, retrieve_papers, save_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a concise deepfake research report from the local knowledge base."
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="deepfake detection methods using transformers and GANs",
        help="Research topic or question to analyze.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieved chunks to use as context.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="research_report.md",
        help="Where to save the final markdown report.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=700,
        help="Maximum tokens to ask Groq to generate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("DEEPFAKE RESEARCH REPORT GENERATOR")
    print("=" * 70)
    print(f"Topic: {args.topic}")
    print(f"Retrieving top {args.top_k} chunks from the local knowledge base...")

    docs = retrieve_papers(args.topic, k=args.top_k)

    if not docs:
        print("No relevant documents found. Build the knowledge base first.")
        report = (
            f"# Research Report: {args.topic}\n\n"
            "No relevant papers were found in the local knowledge base. "
            "Run `python build_knowledge_base.py` first."
        )
        output_path = save_report(report, args.output)
        print(f"Saved empty report template to {output_path}")
        return

    print(f"Found {len(docs)} relevant chunks. Generating report...")
    report = generate_report(args.topic, docs, max_tokens=args.max_tokens)
    output_path = save_report(report, args.output)

    print(f"Saved report to {output_path}")
    print("\n" + "=" * 70)
    print(report[:2500])
    if len(report) > 2500:
        print("\n...output truncated in console, full report is in the markdown file.")
    print("=" * 70)


if __name__ == "__main__":
    main()
