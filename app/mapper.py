def _nearest_speaker(span_start, span_end, diar_segments):
    if not diar_segments:
        return "Unknown"
    span_center = (span_start + span_end) / 2.0
    best = "Unknown"
    best_dist = float("inf")
    for d in diar_segments:
        diar_center = (d["start"] + d["end"]) / 2.0
        dist = abs(span_center - diar_center)
        if dist < best_dist:
            best_dist = dist
            best = d["speaker"]
    return best


def _overlap_by_speaker(span_start, span_end, diar_segments, collar=0.08):
    totals = {}
    a = span_start - collar
    b = span_end + collar
    for d in diar_segments:
        overlap = max(0.0, min(b, d["end"]) - max(a, d["start"]))
        if overlap <= 0:
            continue
        totals[d["speaker"]] = totals.get(d["speaker"], 0.0) + overlap
    return totals


def _speaker_for_span(span_start, span_end, diar_segments):
    totals = _overlap_by_speaker(span_start, span_end, diar_segments)
    if totals:
        return max(totals.items(), key=lambda x: x[1])[0]
    return _nearest_speaker(span_start, span_end, diar_segments)


def _words_from_segment(segment):
    raw_words = segment.get("words") or []
    if not raw_words:
        return []

    seg_start = float(segment["start"])
    seg_end = float(segment["end"])
    seg_duration = max(0.01, seg_end - seg_start)

    words = []
    for w in raw_words:
        text = (w.get("word") or "").strip()
        if not text:
            continue
        ws = w.get("start")
        we = w.get("end")
        ws = float(ws) if ws is not None else None
        we = float(we) if we is not None else None
        words.append({"text": text, "start": ws, "end": we})

    if not words:
        return []

    missing = any(w["start"] is None or w["end"] is None for w in words)
    if missing:
        avg = seg_duration / max(1, len(words))
        cursor = seg_start
        for i, w in enumerate(words):
            if w["start"] is None:
                w["start"] = cursor
            if w["end"] is None or w["end"] <= w["start"]:
                w["end"] = w["start"] + avg
            if i == len(words) - 1:
                w["end"] = min(seg_end, max(w["end"], w["start"] + 0.03))
            cursor = w["end"]

    words[0]["start"] = max(seg_start, words[0]["start"])
    for i in range(len(words)):
        words[i]["start"] = max(seg_start, min(seg_end, words[i]["start"]))
        words[i]["end"] = max(words[i]["start"] + 0.02, min(seg_end, words[i]["end"]))
        if i > 0:
            words[i]["start"] = max(words[i - 1]["end"] - 0.01, words[i]["start"])
            words[i]["end"] = max(words[i]["start"] + 0.02, words[i]["end"])
    words[-1]["end"] = min(seg_end, words[-1]["end"])

    return words


def _smooth_short_flips(items, short_duration=0.65):
    if len(items) < 3:
        return items
    out = [dict(x) for x in items]
    for i in range(1, len(out) - 1):
        prev_s = out[i - 1]["speaker"]
        cur_s = out[i]["speaker"]
        next_s = out[i + 1]["speaker"]
        cur_dur = max(0.0, out[i]["end"] - out[i]["start"])
        if cur_dur <= short_duration and cur_s != prev_s and prev_s == next_s:
            out[i]["speaker"] = prev_s
    return out


def _window_majority_smooth(items, radius=2):
    if len(items) < 5:
        return items
    out = [dict(x) for x in items]
    for i in range(len(out)):
        lo = max(0, i - radius)
        hi = min(len(out), i + radius + 1)
        counts = {}
        for j in range(lo, hi):
            s = out[j]["speaker"]
            d = max(0.01, out[j]["end"] - out[j]["start"])
            counts[s] = counts.get(s, 0.0) + d
        best = max(counts.items(), key=lambda x: x[1])[0]
        out[i]["speaker"] = best
    return out


def _merge_same_speaker(items, max_gap=0.22):
    if not items:
        return []
    merged = [dict(items[0])]
    for item in items[1:]:
        last = merged[-1]
        same = item["speaker"] == last["speaker"]
        close = (item["start"] - last["end"]) <= max_gap
        if same and close:
            last["end"] = max(last["end"], item["end"])
            text = (item.get("text") or "").strip()
            if text:
                if last["text"]:
                    last["text"] = f"{last['text']} {text}".strip()
                else:
                    last["text"] = text
        else:
            merged.append(dict(item))
    return merged


def _map_segment_with_words(segment, diar_segments):
    seg_start = float(segment["start"])
    seg_end = float(segment["end"])
    words = _words_from_segment(segment)
    if not words:
        return []

    word_items = []
    for w in words:
        ws = float(w["start"])
        we = float(w["end"])
        if we <= ws:
            continue
        speaker = _speaker_for_span(ws, we, diar_segments)
        word_items.append(
            {
                "speaker": speaker,
                "start": ws,
                "end": we,
                "text": w["text"],
            }
        )

    if not word_items:
        return []

    word_items = _smooth_short_flips(word_items, short_duration=0.3)
    word_items = _window_majority_smooth(word_items, radius=2)
    merged = _merge_same_speaker(word_items, max_gap=0.12)

    for m in merged:
        m["start"] = max(seg_start, m["start"])
        m["end"] = min(seg_end, m["end"])
    return [m for m in merged if m["end"] > m["start"] and (m["text"] or "").strip()]


def _map_segment_fallback(segment, diar_segments, min_overlap_ratio):
    seg_start = float(segment["start"])
    seg_end = float(segment["end"])
    seg_text = (segment.get("text") or "").strip()
    seg_duration = max(0.0, seg_end - seg_start)
    if seg_duration <= 0 or not seg_text:
        return []

    totals = _overlap_by_speaker(seg_start, seg_end, diar_segments, collar=0.1)
    if totals:
        best_speaker, best_overlap = max(totals.items(), key=lambda x: x[1])
        if (best_overlap / seg_duration) >= min_overlap_ratio:
            speaker = best_speaker
        else:
            speaker = _nearest_speaker(seg_start, seg_end, diar_segments)
    else:
        speaker = _nearest_speaker(seg_start, seg_end, diar_segments)

    return [
        {
            "speaker": speaker,
            "start": seg_start,
            "end": seg_end,
            "text": seg_text,
        }
    ]


def map_speakers(transcription_segments, diarization_output, min_overlap_ratio=0.25):
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
        text = (segment.get("text") or "").strip()
        if not text:
            continue

        mapped = _map_segment_with_words(segment, diar_segments)
        if not mapped:
            mapped = _map_segment_fallback(segment, diar_segments, min_overlap_ratio)
        final_output.extend(mapped)

    asr_seconds = 0.0
    for s in transcription_segments:
        start = float(s.get("start", 0.0))
        end = float(s.get("end", 0.0))
        if end > start:
            asr_seconds += end - start

    mapped_seconds = 0.0
    for s in final_output:
        start = float(s.get("start", 0.0))
        end = float(s.get("end", 0.0))
        if end > start:
            mapped_seconds += end - start

    if asr_seconds > 0 and (mapped_seconds / asr_seconds) < 0.8:
        recovered = []
        for segment in transcription_segments:
            recovered.extend(_map_segment_fallback(segment, diar_segments, min_overlap_ratio))
        final_output = recovered

    final_output = _smooth_short_flips(final_output, short_duration=0.8)
    final_output = _merge_same_speaker(final_output, max_gap=0.25)
    return final_output
