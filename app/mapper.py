def _token_slices_by_weight(text, weights):
    tokens = text.strip().split()
    if not tokens or not weights:
        return []

    total = sum(max(w, 0.0) for w in weights)
    if total <= 0:
        return []

    counts = []
    remaining = len(tokens)
    for idx, w in enumerate(weights):
        if idx == len(weights) - 1:
            c = remaining
        else:
            c = max(1, round((max(w, 0.0) / total) * len(tokens)))
            c = min(c, remaining)
        counts.append(c)
        remaining -= c

    i = len(counts) - 1
    while remaining > 0 and i >= 0:
        counts[i] += 1
        remaining -= 1
        i -= 1

    slices = []
    start = 0
    for c in counts:
        end = min(len(tokens), start + c)
        slices.append(" ".join(tokens[start:end]).strip())
        start = end
    return slices


def _nearest_speaker(seg_start, seg_end, diar_segments):
    if not diar_segments:
        return "Unknown"
    seg_center = (seg_start + seg_end) / 2.0
    best_speaker = "Unknown"
    min_distance = float("inf")
    for d in diar_segments:
        diar_center = (d["start"] + d["end"]) / 2.0
        distance = abs(seg_center - diar_center)
        if distance < min_distance:
            min_distance = distance
            best_speaker = d["speaker"]
    return best_speaker


def _smooth_short_flips(items, short_duration=0.8):
    if len(items) < 3:
        return items

    smoothed = [dict(x) for x in items]
    for i in range(1, len(smoothed) - 1):
        prev_s = smoothed[i - 1]["speaker"]
        cur_s = smoothed[i]["speaker"]
        next_s = smoothed[i + 1]["speaker"]
        cur_dur = smoothed[i]["end"] - smoothed[i]["start"]
        if cur_dur <= short_duration and prev_s == next_s and cur_s != prev_s:
            smoothed[i]["speaker"] = prev_s
    return smoothed


def _merge_same_speaker(items, max_gap=0.25):
    if not items:
        return items
    merged = [dict(items[0])]
    for item in items[1:]:
        last = merged[-1]
        same_speaker = item["speaker"] == last["speaker"]
        close_gap = (item["start"] - last["end"]) <= max_gap
        if same_speaker and close_gap:
            last["end"] = max(last["end"], item["end"])
            text = item["text"].strip()
            if text:
                last["text"] = f"{last['text']} {text}".strip()
        else:
            merged.append(dict(item))
    return merged


def map_speakers(transcription_segments, diarization_output, min_overlap_ratio=0.25):
    """
    Robust speaker mapping:
    - Assigns speaker by aggregated overlap confidence.
    - Splits one ASR segment across multiple speakers only when overlap confidence supports it.
    - Smooths short speaker flips and merges contiguous same-speaker segments.
    """
    final_output = []

    diarization = diarization_output.speaker_diarization
    diar_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diar_segments.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": speaker,
            }
        )
    diar_segments.sort(key=lambda x: x["start"])

    for segment in transcription_segments:
        seg_start = float(segment["start"])
        seg_end = float(segment["end"])
        seg_text = (segment.get("text") or "").strip()
        seg_duration = max(0.0, seg_end - seg_start)
        if seg_duration <= 0 or not seg_text:
            continue

        overlaps = []
        speaker_totals = {}

        for d in diar_segments:
            overlap_start = max(seg_start, d["start"])
            overlap_end = min(seg_end, d["end"])
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap <= 0:
                continue
            overlaps.append(
                {
                    "speaker": d["speaker"],
                    "start": overlap_start,
                    "end": overlap_end,
                    "overlap": overlap,
                }
            )
            speaker_totals[d["speaker"]] = speaker_totals.get(d["speaker"], 0.0) + overlap

        if not overlaps:
            final_output.append(
                {
                    "speaker": _nearest_speaker(seg_start, seg_end, diar_segments),
                    "start": seg_start,
                    "end": seg_end,
                    "text": seg_text,
                }
            )
            continue

        sorted_speakers = sorted(speaker_totals.items(), key=lambda x: x[1], reverse=True)
        best_speaker, best_overlap = sorted_speakers[0]
        best_ratio = best_overlap / seg_duration

        # Controlled split: only for sufficiently long segments and real 2-speaker overlap.
        if (
            len(sorted_speakers) >= 2
            and seg_duration >= 2.5
            and len(seg_text.split()) >= 10
        ):
            second_speaker, second_overlap = sorted_speakers[1]
            second_ratio = second_overlap / seg_duration
            if best_ratio >= 0.2 and second_ratio >= 0.2:
                two_speakers = [best_speaker, second_speaker]
                two_weights = [best_overlap, second_overlap]
                text_chunks = _token_slices_by_weight(seg_text, two_weights)

                # Allocate timing proportionally by overlap.
                total_two = sum(two_weights)
                cursor = seg_start
                for i, spk in enumerate(two_speakers):
                    dur = seg_duration * (two_weights[i] / total_two) if total_two > 0 else seg_duration / 2.0
                    end_t = seg_end if i == len(two_speakers) - 1 else min(seg_end, cursor + dur)
                    chunk = text_chunks[i] if i < len(text_chunks) else ""
                    if chunk.strip():
                        final_output.append(
                            {
                                "speaker": spk,
                                "start": cursor,
                                "end": end_t,
                                "text": chunk.strip(),
                            }
                        )
                    cursor = end_t
                continue

        # Standard single-speaker assignment when confidence is reasonable.
        if best_ratio >= min_overlap_ratio:
            final_output.append(
                {
                    "speaker": best_speaker,
                    "start": seg_start,
                    "end": seg_end,
                    "text": seg_text,
                }
            )
        else:
            final_output.append(
                {
                    "speaker": _nearest_speaker(seg_start, seg_end, diar_segments),
                    "start": seg_start,
                    "end": seg_end,
                    "text": seg_text,
                }
            )

    final_output = _smooth_short_flips(final_output)
    final_output = _merge_same_speaker(final_output)
    return final_output
