"""
Author: L. Saetta
Date last modified: 2026-09-01
License: MIT
Description: Extracts verified CV facts with a local Qwen model and builds train and evaluation JSONL datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from pypdf import PdfReader
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "You are a CV assistant. Answer only with professional information explicitly "
    "present in the candidate's CV. Do not infer or add details."
)
CONTACT_PATTERNS = (
    re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"(?:\+?\d[\d .()/-]{7,}\d)"),
    re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE),
)


@dataclass(frozen=True)
class AtomicFact:
    """A verified CV fact with its page-level source location.

    Attributes:
        fact_id: Stable identifier derived from the page and fact text.
        text: Verbatim fact text validated against the extracted PDF page.
        page_number: One-based page number in the input PDF.
    """

    fact_id: str
    text: str
    page_number: int


def normalise_text(value: str) -> str:
    """Return whitespace-normalised text for deterministic source comparisons.

    Args:
        value: Text to normalise.

    Returns:
        Lowercase text with consecutive whitespace collapsed.
    """

    return " ".join(value.casefold().split())


def contains_contact_detail(value: str) -> bool:
    """Return whether text contains a direct contact detail.

    Args:
        value: Candidate fact text.

    Returns:
        True when an email address, phone-like number, or URL is found.
    """

    return any(pattern.search(value) for pattern in CONTACT_PATTERNS)


def make_fact_id(page_number: int, fact_text: str) -> str:
    """Create a stable fact identifier.

    Args:
        page_number: One-based source page number.
        fact_text: Verified fact text.

    Returns:
        A short deterministic identifier.
    """

    digest = hashlib.sha256(normalise_text(fact_text).encode("utf-8")).hexdigest()[:12]
    return f"p{page_number:02d}-{digest}"


def parse_json_payload(model_output: str) -> Any:
    """Parse the first JSON object or array emitted by the model.

    Args:
        model_output: Raw generated model text.

    Returns:
        The decoded JSON payload.

    Raises:
        ValueError: If no JSON object or array can be decoded.
    """

    decoder = json.JSONDecoder()
    for index, character in enumerate(model_output):
        if character not in "[{":
            continue
        try:
            payload, _ = decoder.raw_decode(model_output[index:])
            return payload
        except json.JSONDecodeError:
            continue
    raise ValueError("The local model did not return a JSON object or array.")


def extract_pdf_pages(pdf_path: Path) -> list[str]:
    """Extract non-empty text from every page of a text-based PDF.

    Args:
        pdf_path: Path to the CV PDF.

    Returns:
        One text string per PDF page.

    Raises:
        FileNotFoundError: If the PDF path does not exist.
        ValueError: If no readable text is extracted.
    """

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text(extraction_mode="layout") or "" for page in reader.pages]
    if not any(page.strip() for page in pages):
        raise ValueError(
            "No readable text was extracted from the PDF. This may be a scanned PDF; "
            "run OCR locally first and provide a text-based PDF."
        )
    return pages


def chunk_page_text(page_text: str, maximum_characters: int = 2_400) -> list[str]:
    """Split extracted page text on line boundaries for reliable local generation.

    Args:
        page_text: Extracted text from one PDF page.
        maximum_characters: Maximum characters supplied to one model request.

    Returns:
        Non-empty source text chunks in original order.
    """

    chunks: list[str] = []
    current_lines: list[str] = []
    current_length = 0
    for line in page_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if current_lines and current_length + len(line) + 1 > maximum_characters:
            chunks.append("\n".join(current_lines))
            current_lines = []
            current_length = 0
        current_lines.append(line)
        current_length += len(line) + 1
    if current_lines:
        chunks.append("\n".join(current_lines))
    return chunks


def select_device(requested_device: str) -> torch.device:
    """Select a compute device according to the project's MPS contract.

    Args:
        requested_device: One of ``auto``, ``mps``, or ``cpu``.

    Returns:
        The selected torch device.

    Raises:
        RuntimeError: If MPS is explicitly requested but unavailable.
        ValueError: If the requested device is unsupported.
    """

    if requested_device not in {"auto", "mps", "cpu"}:
        raise ValueError("--device must be one of: auto, mps, cpu.")
    if requested_device == "cpu":
        return torch.device("cpu")
    if requested_device == "mps":
        if not torch.backends.mps.is_built():
            raise RuntimeError("MPS was requested, but this PyTorch build has no MPS support.")
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but it is unavailable in this runtime.")
        return torch.device("mps")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    print("Warning: MPS is unavailable. Falling back to CPU; generation will be slower.")
    return torch.device("cpu")


def generate_local_text(
    model: AutoModelForCausalLM,
    tokenizer: Any,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
) -> str:
    """Generate one non-thinking response using the local Qwen model.

    Args:
        model: Loaded causal language model.
        tokenizer: Matching tokenizer.
        device: Target torch device.
        prompt: User instruction passed through the Qwen chat template.
        max_new_tokens: Maximum generation length.

    Returns:
        Decoded generated text without the prompt.
    """

    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = tokenizer(chat_text, return_tensors="pt")
    model_inputs = {name: tensor.to(device) for name, tensor in model_inputs.items()}
    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    response_ids = generated_ids[0, model_inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip()


def extract_verified_facts(
    page_text: str,
    page_number: int,
    model: AutoModelForCausalLM,
    tokenizer: Any,
    device: torch.device,
    include_contact_details: bool,
) -> list[AtomicFact]:
    """Use the local model to extract source-verifiable atomic facts from one page.

    Args:
        page_text: Extracted text for one PDF page.
        page_number: One-based source page number.
        model: Loaded local Qwen model.
        tokenizer: Matching tokenizer.
        device: Target torch device.
        include_contact_details: Whether direct contact details may be retained.

    Returns:
        Deduplicated verified atomic facts from the page.
    """

    if not page_text.strip():
        return []
    facts: list[AtomicFact] = []
    seen: set[str] = set()
    for chunk_index, page_chunk in enumerate(chunk_page_text(page_text), start=1):
        prompt = f"""Extract atomic professional facts from this CV page.

Return JSON only in exactly this shape:
{{"facts": [{{"fact": "verbatim text copied from the page"}}]}}

Rules:
- Return 10 to 30 short, useful facts when the page contains that many.
- Every fact must be copied verbatim from the page; do not paraphrase, merge, infer, or add facts.
- Focus on work experience, roles, achievements, skills, education, certifications, projects, and languages.
- Exclude email addresses, phone numbers, postal addresses, URLs, and social-media handles.
- Do not include headings, page numbers, or repeated boilerplate as facts.

CV page {page_number}:
---
{page_chunk}
---"""
        try:
            payload = parse_json_payload(generate_local_text(model, tokenizer, device, prompt, 2_048))
        except ValueError as error:
            print(
                f"Warning: page {page_number}, fragment {chunk_index} fact extraction was skipped: {error}"
            )
            continue
        entries = payload.get("facts", []) if isinstance(payload, dict) else []
        source_text = normalise_text(page_chunk)
        for entry in entries:
            candidate = entry.get("fact", "") if isinstance(entry, dict) else ""
            candidate = " ".join(str(candidate).split())
            candidate_key = normalise_text(candidate)
            if (
                len(candidate) < 8
                or len(candidate) > 450
                or candidate_key in seen
                or candidate_key not in source_text
                or (not include_contact_details and contains_contact_detail(candidate))
            ):
                continue
            seen.add(candidate_key)
            facts.append(AtomicFact(make_fact_id(page_number, candidate), candidate, page_number))
    return facts


def deduplicate_facts(facts: Iterable[AtomicFact]) -> list[AtomicFact]:
    """Deduplicate facts across pages while retaining the first source location.

    Args:
        facts: Extracted facts in source order.

    Returns:
        Facts with globally unique normalised text.
    """

    unique_facts: list[AtomicFact] = []
    seen: set[str] = set()
    for fact in facts:
        key = normalise_text(fact.text)
        if key not in seen:
            seen.add(key)
            unique_facts.append(fact)
    return unique_facts


def split_facts(
    facts: list[AtomicFact], eval_fraction: float, seed: int
) -> tuple[list[AtomicFact], list[AtomicFact]]:
    """Split facts deterministically before question generation.

    Args:
        facts: Verified facts to split.
        eval_fraction: Fraction reserved for evaluation.
        seed: Random seed for stable split assignment.

    Returns:
        Train and evaluation fact lists with no shared fact IDs.

    Raises:
        ValueError: If too few facts or an invalid fraction is supplied.
    """

    if len(facts) < 2:
        raise ValueError("At least two verified facts are required to create train and evaluation splits.")
    if not 0 < eval_fraction < 1:
        raise ValueError("--eval-fraction must be greater than 0 and less than 1.")
    shuffled = list(facts)
    random.Random(seed).shuffle(shuffled)
    eval_count = max(1, min(len(shuffled) - 1, round(len(shuffled) * eval_fraction)))
    return shuffled[eval_count:], shuffled[:eval_count]


def generate_questions(
    facts: list[AtomicFact],
    model: AutoModelForCausalLM,
    tokenizer: Any,
    device: torch.device,
    questions_per_fact: int,
) -> dict[str, list[str]]:
    """Generate factual questions for verified facts in small local-model batches.

    Args:
        facts: Facts assigned to one dataset split.
        model: Loaded local Qwen model.
        tokenizer: Matching tokenizer.
        device: Target torch device.
        questions_per_fact: Maximum questions requested for each fact.

    Returns:
        Mapping from fact ID to unique generated questions.
    """

    results: dict[str, list[str]] = {fact.fact_id: [] for fact in facts}
    for start in range(0, len(facts), 4):
        batch = facts[start : start + 4]
        fact_payload = [{"fact_id": fact.fact_id, "fact": fact.text} for fact in batch]
        prompt = f"""Create factual CV questions from the verified facts below.

Return JSON only in exactly this shape:
{{"items": [{{"fact_id": "id from input", "questions": ["question 1"]}}]}}

Rules:
- Return up to {questions_per_fact} distinct questions for every input fact.
- Each question must be answerable only by its associated fact.
- Do not copy the fact verbatim into the question, do not ask for contact details, and do not introduce facts not present in the input.
- Use clear, natural professional English.
- Return no answer text, explanation, or Markdown.

Verified facts:
{json.dumps(fact_payload, ensure_ascii=False)}"""
        try:
            payload = parse_json_payload(generate_local_text(model, tokenizer, device, prompt, 1_024))
        except ValueError as error:
            print(f"Warning: question generation batch {start // 4 + 1} was skipped: {error}")
            continue
        entries = payload.get("items", []) if isinstance(payload, dict) else []
        valid_ids = {fact.fact_id for fact in batch}
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("fact_id") not in valid_ids:
                continue
            questions = entry.get("questions", [])
            if not isinstance(questions, list):
                continue
            for candidate in questions:
                candidate = " ".join(str(candidate).split())
                if (
                    candidate.endswith("?")
                    and 12 <= len(candidate) <= 300
                    and candidate not in results[entry["fact_id"]]
                    and not contains_contact_detail(candidate)
                ):
                    results[entry["fact_id"]].append(candidate)
    for fact in facts:
        results[fact.fact_id] = complete_question_set(
            fact, results[fact.fact_id], questions_per_fact
        )
    return results


def complete_question_set(
    fact: AtomicFact, generated_questions: list[str], questions_per_fact: int
) -> list[str]:
    """Complete a Qwen-generated question set with grounded phrasing variants.

    Args:
        fact: Verified fact used as the only answer source.
        generated_questions: Questions returned by the local model.
        questions_per_fact: Desired number of distinct questions.

    Returns:
        At most the requested number of non-contact questions about the fact.
    """

    topic = re.sub(r"\s+", " ", fact.text).strip(" .;:-")
    topic = " ".join(topic.split()[:8]).rstrip(",.;:-")
    templates = (
        "According to the CV, what is noted about {topic}?",
        "Which professional detail does the CV provide about {topic}?",
        "What does the candidate's CV report concerning {topic}?",
        "How is {topic} described in the CV?",
        "What relevant information is stated about {topic}?",
        "Based on the CV, what should be known about {topic}?",
        "What CV detail relates to {topic}?",
    )
    questions = list(generated_questions)
    for template in templates:
        candidate = template.format(topic=topic)
        if candidate not in questions and not contains_contact_detail(candidate):
            questions.append(candidate)
        if len(questions) >= questions_per_fact:
            break
    return questions[:questions_per_fact]


def build_records(
    facts: list[AtomicFact], questions_by_fact: dict[str, list[str]], split_name: str, limit: int
) -> list[dict[str, Any]]:
    """Build conversational SFT records with answers bound to verified facts.

    Args:
        facts: Facts for one split.
        questions_by_fact: Generated questions keyed by fact ID.
        split_name: Dataset split name.
        limit: Maximum examples to emit.

    Returns:
        Valid conversational dataset records.
    """

    records: list[dict[str, Any]] = []
    for fact in facts:
        for question_index, question in enumerate(questions_by_fact.get(fact.fact_id, []), start=1):
            records.append(
                {
                    "id": f"{split_name}-{fact.fact_id}-q{question_index}",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": fact.text},
                    ],
                    "metadata": {
                        "fact_id": fact.fact_id,
                        "source_page": fact.page_number,
                        "split": split_name,
                    },
                }
            )
            if len(records) >= limit:
                return records
    return records


def validate_records(records: list[dict[str, Any]]) -> None:
    """Validate conversational records before they are written.

    Args:
        records: Records to validate.

    Raises:
        ValueError: If a record violates the local SFT format or privacy contract.
    """

    for record in records:
        messages = record.get("messages", [])
        roles = [message.get("role") for message in messages]
        contents = [message.get("content", "") for message in messages]
        if roles != ["system", "user", "assistant"] or not all(str(content).strip() for content in contents):
            raise ValueError(f"Invalid conversational record: {record.get('id')}")
        if any(contains_contact_detail(str(content)) for content in contents):
            raise ValueError(f"Contact detail found in generated record: {record.get('id')}")


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write records to a UTF-8 JSONL file.

    Args:
        records: Records to write.
        output_path: Destination JSONL path.
    """

    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for local CV dataset generation.

    Returns:
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-path", type=Path, required=True, help="Path to a text-based CV PDF.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/datasets/cv-qa"),
        help="Private output directory for train.jsonl and eval.jsonl.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("artifacts/models/Qwen3-0.6B"),
        help="Local Qwen3-0.6B model directory.",
    )
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-fraction", type=float, default=0.25)
    parser.add_argument("--questions-per-fact", type=int, default=7)
    parser.add_argument("--train-limit", type=int, default=150)
    parser.add_argument("--eval-limit", type=int, default=50)
    parser.add_argument("--include-contact-details", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run local CV fact extraction and train/evaluation dataset generation."""

    arguments = parse_arguments()
    if arguments.questions_per_fact < 1 or arguments.train_limit < 1 or arguments.eval_limit < 1:
        raise ValueError("Question and dataset limits must be positive integers.")
    if not arguments.model_dir.is_dir():
        raise FileNotFoundError(
            f"Local model directory not found: {arguments.model_dir}. Download Qwen3-0.6B first."
        )

    pages = extract_pdf_pages(arguments.pdf_path)
    device = select_device(arguments.device)
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"Selected device: {device}")
    print(f"Loading local model from: {arguments.model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(arguments.model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        arguments.model_dir, dtype=torch.float32, local_files_only=True
    ).to(device)
    model.eval()

    extracted_facts: list[AtomicFact] = []
    for page_number, page_text in enumerate(pages, start=1):
        print(f"Extracting facts from page {page_number}/{len(pages)}...")
        extracted_facts.extend(
            extract_verified_facts(
                page_text,
                page_number,
                model,
                tokenizer,
                device,
                arguments.include_contact_details,
            )
        )
    facts = deduplicate_facts(extracted_facts)
    print(f"Verified atomic facts: {len(facts)}")

    train_facts, eval_facts = split_facts(facts, arguments.eval_fraction, arguments.seed)
    train_questions = generate_questions(
        train_facts, model, tokenizer, device, arguments.questions_per_fact
    )
    eval_questions = generate_questions(
        eval_facts, model, tokenizer, device, arguments.questions_per_fact
    )
    train_records = build_records(train_facts, train_questions, "train", arguments.train_limit)
    eval_records = build_records(eval_facts, eval_questions, "eval", arguments.eval_limit)
    validate_records(train_records)
    validate_records(eval_records)

    train_fact_ids = {record["metadata"]["fact_id"] for record in train_records}
    eval_fact_ids = {record["metadata"]["fact_id"] for record in eval_records}
    overlap = train_fact_ids & eval_fact_ids
    if overlap:
        raise RuntimeError(f"Fact leakage detected between splits: {sorted(overlap)}")
    if not train_records or not eval_records:
        raise RuntimeError("No complete train/evaluation datasets were generated from verified facts.")

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train_records, arguments.output_dir / "train.jsonl")
    write_jsonl(eval_records, arguments.output_dir / "eval.jsonl")
    manifest = {
        "source_pdf_filename": arguments.pdf_path.name,
        "model_directory": str(arguments.model_dir),
        "pytorch_version": torch.__version__,
        "selected_device": str(device),
        "seed": arguments.seed,
        "questions_per_fact": arguments.questions_per_fact,
        "verified_fact_count": len(facts),
        "train_fact_ids": sorted(train_fact_ids),
        "eval_fact_ids": sorted(eval_fact_ids),
        "split_overlap_fact_ids": sorted(overlap),
        "train_examples": len(train_records),
        "eval_examples": len(eval_records),
    }
    (arguments.output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(train_records)} training examples to {arguments.output_dir / 'train.jsonl'}")
    print(f"Wrote {len(eval_records)} evaluation examples to {arguments.output_dir / 'eval.jsonl'}")
    if len(train_records) < 100 or len(eval_records) < 30:
        print("Warning: fewer examples than the target range were generated; review the CV and model output.")


if __name__ == "__main__":
    main()
