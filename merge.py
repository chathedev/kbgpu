import logging
from typing import Optional

logger = logging.getLogger(__name__)

MIN_UTTERANCE_DURATION = 0.5
MAX_MERGE_GAP = 0.3


def merge(words: list[dict], diar_segments: list[dict]) -> list[dict]:
    """
    Assign speakers to words using overlap method, then group into utterances.

    words: [{"word": str, "start": float, "end": float, "probability": float}]
    diar_segments: [{"speaker": str, "start": float, "end": float}]

    Returns list of utterances:
    {"speaker": str, "start": float, "end": float, "text": str, "words": list}
    Speakers labeled as "Talare 1", "Talare 2" etc. in order of first appearance.
    """
    if not words:
        logger.warning("No words to merge")
        return []

    if not diar_segments:
        logger.warning("No diarization segments - assigning all words to single speaker")
        return _single_speaker_utterances(words)

    # Assign speaker to each word
    labeled_words = _assign_speakers(words, diar_segments)

    # Build speaker label mapping (ordered by first appearance)
    speaker_order = []
    for w in labeled_words:
        raw = w["speaker"]
        if raw not in speaker_order:
            speaker_order.append(raw)
    speaker_map = {raw: f"Talare {i+1}" for i, raw in enumerate(speaker_order)}

    # Apply Swedish labels
    for w in labeled_words:
        w["speaker"] = speaker_map[w["speaker"]]

    # Group consecutive same-speaker words into utterances
    utterances = _group_into_utterances(labeled_words)

    # Merge short utterances into adjacent same-speaker utterances
    utterances = _merge_short_utterances(utterances)

    logger.info(f"Merged {len(words)} words into {len(utterances)} utterances with {len(speaker_map)} speakers")
    return utterances


def _assign_speakers(words: list[dict], segments: list[dict]) -> list[dict]:
    """Assign each word to a speaker using maximum overlap, fallback to nearest midpoint."""
    labeled = []
    for word in words:
        w_start = word["start"]
        w_end = word["end"]
        w_mid = (w_start + w_end) / 2

        best_speaker = None
        best_overlap = -1.0
        best_dist = float("inf")

        for seg in segments:
            # Compute overlap
            overlap = min(w_end, seg["end"]) - max(w_start, seg["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = seg["speaker"]

            # Track nearest by midpoint for fallback
            seg_mid = (seg["start"] + seg["end"]) / 2
            dist = abs(w_mid - seg_mid)
            if dist < best_dist:
                best_dist = dist
                nearest_speaker = seg["speaker"]

        # Use nearest if no overlap found
        if best_overlap <= 0:
            best_speaker = nearest_speaker

        labeled.append({**word, "speaker": best_speaker})

    return labeled


def _group_into_utterances(words: list[dict]) -> list[dict]:
    """Group consecutive words with same speaker into utterances."""
    if not words:
        return []

    utterances = []
    current_words = [words[0]]
    current_speaker = words[0]["speaker"]

    for word in words[1:]:
        if word["speaker"] == current_speaker:
            current_words.append(word)
        else:
            utterances.append(_make_utterance(current_speaker, current_words))
            current_words = [word]
            current_speaker = word["speaker"]

    utterances.append(_make_utterance(current_speaker, current_words))
    return utterances


def _make_utterance(speaker: str, words: list[dict]) -> dict:
    text = " ".join(w["word"] for w in words).strip()
    return {
        "speaker": speaker,
        "start": round(words[0]["start"], 3),
        "end": round(words[-1]["end"], 3),
        "text": text,
        "words": [{"word": w["word"], "start": w["start"], "end": w["end"]} for w in words],
    }


def _merge_short_utterances(utterances: list[dict]) -> list[dict]:
    """
    Merge utterances shorter than MIN_UTTERANCE_DURATION into adjacent
    same-speaker utterance if gap < MAX_MERGE_GAP.
    """
    if len(utterances) <= 1:
        return utterances

    changed = True
    while changed:
        changed = False
        merged = []
        i = 0
        while i < len(utterances):
            utt = utterances[i]
            duration = utt["end"] - utt["start"]

            if duration < MIN_UTTERANCE_DURATION:
                # Try merging with previous same-speaker utterance
                if merged and merged[-1]["speaker"] == utt["speaker"]:
                    gap = utt["start"] - merged[-1]["end"]
                    if gap < MAX_MERGE_GAP:
                        merged[-1] = _combine_utterances(merged[-1], utt)
                        changed = True
                        i += 1
                        continue

                # Try merging with next same-speaker utterance
                if i + 1 < len(utterances) and utterances[i + 1]["speaker"] == utt["speaker"]:
                    gap = utterances[i + 1]["start"] - utt["end"]
                    if gap < MAX_MERGE_GAP:
                        utterances[i + 1] = _combine_utterances(utt, utterances[i + 1])
                        changed = True
                        i += 1
                        continue

            merged.append(utt)
            i += 1

        utterances = merged

    return utterances


def _combine_utterances(a: dict, b: dict) -> dict:
    combined_words = a["words"] + b["words"]
    return {
        "speaker": a["speaker"],
        "start": a["start"],
        "end": b["end"],
        "text": (a["text"] + " " + b["text"]).strip(),
        "words": combined_words,
    }


def _single_speaker_utterances(words: list[dict]) -> list[dict]:
    """Fallback: put all words under a single speaker."""
    utt = _make_utterance("Talare 1", words)
    return [utt]
