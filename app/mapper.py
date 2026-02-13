def map_speakers(transcription, diarization_output, min_overlap_ratio=0.3):
    """
    Improved speaker mapping:
    - Uses overlap ratio (not just raw overlap)
    - Handles no-overlap cases
    - Uses nearest speaker fallback
    - More stable mapping
    """

    final_output = []

    # Get diarization tracks
    diarization = diarization_output.speaker_diarization

    # Convert diarization to simple list (faster + easier)
    diar_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diar_segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })

    # Sort diarization segments by time
    diar_segments.sort(key=lambda x: x["start"])

    for segment in transcription["segments"]:
        seg_start = segment["start"]
        seg_end = segment["end"]
        seg_text = segment["text"]
        seg_duration = seg_end - seg_start

        best_speaker = "Unknown"
        best_ratio = 0

        # ---------- Find best overlap ----------
        for d in diar_segments:
            overlap_start = max(seg_start, d["start"])
            overlap_end = min(seg_end, d["end"])
            overlap = max(0, overlap_end - overlap_start)

            if overlap > 0:
                ratio = overlap / seg_duration
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_speaker = d["speaker"]

        # ---------- If no good overlap → use nearest speaker ----------
        if best_ratio < min_overlap_ratio:
            min_distance = float("inf")

            for d in diar_segments:
                # distance from segment center
                seg_center = (seg_start + seg_end) / 2
                diar_center = (d["start"] + d["end"]) / 2
                distance = abs(seg_center - diar_center)

                if distance < min_distance:
                    min_distance = distance
                    best_speaker = d["speaker"]

        final_output.append({
            "speaker": best_speaker,
            "start": seg_start,
            "end": seg_end,
            "text": seg_text.strip()
        })

    return final_output
