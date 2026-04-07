import os
import logging
import tempfile
import shutil
from typing import Optional

logger = logging.getLogger(__name__)

MSDD_MODEL_PATH = "/models/nemo/diar_msdd_telephony.nemo"
TITANET_MODEL_PATH = "/models/nemo/titanet-large.nemo"


def diarize(audio_path: str, num_speakers: Optional[int] = None, job_id: Optional[str] = None) -> list[dict]:
    """
    Run NeMo ClusteringDiarizer with MSDD for speaker diarization.

    Returns list of segments:
    {"speaker": str, "start": float, "end": float}
    Sorted by start time.
    """
    from omegaconf import OmegaConf
    from nemo.collections.asr.models import ClusteringDiarizer

    job_tag = job_id or "default"
    tmp_dir = f"/tmp/nemo_{job_tag}"
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        logger.info(f"Starting diarization: {audio_path}, num_speakers={num_speakers}")

        # Build manifest file pointing to our audio
        manifest_path = os.path.join(tmp_dir, "manifest.json")
        import json
        with open(manifest_path, "w") as f:
            json.dump({
                "audio_filepath": audio_path,
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": "-",
                "num_speakers": num_speakers,
                "rttm_filepath": None,
                "uem_filepath": None,
            }, f)
            f.write("\n")

        # Build NeMo diarization config
        cfg = _build_diarizer_config(tmp_dir, manifest_path, num_speakers)

        # Run diarization
        diarizer = ClusteringDiarizer(cfg=cfg)
        diarizer.diarize()

        # Parse RTTM output
        rttm_path = os.path.join(tmp_dir, "pred_rttms", os.path.splitext(os.path.basename(audio_path))[0] + ".rttm")
        segments = _parse_rttm(rttm_path)

        logger.info(f"Diarization complete: {len(segments)} segments, {len({s['speaker'] for s in segments})} speakers")
        return segments

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _build_diarizer_config(tmp_dir: str, manifest_path: str, num_speakers: Optional[int]):
    from omegaconf import OmegaConf

    cfg_dict = {
        "num_workers": 1,
        "sample_rate": 16000,
        "batch_size": 64,
        "device": "cuda",
        "verbose": False,
        "diarizer": {
            "manifest_filepath": manifest_path,
            "out_dir": tmp_dir,
            "oracle_vad": False,
            "collar": 0.25,
            "ignore_overlap": False,
            "vad": {
                "model_path": "vad_multilingual_marblenet",
                "external_vad_manifest": None,
                "parameters": {
                    "onset": 0.8,
                    "offset": 0.6,
                    "pad_onset": 0.1,
                    "pad_offset": 0.1,
                    "min_duration_on": 0.2,
                    "min_duration_off": 0.25,
                    "filter_speech_first": True,
                },
            },
            "speaker_embeddings": {
                "model_path": TITANET_MODEL_PATH,
                "parameters": {
                    "window_length_in_sec": [1.5, 1.25, 1.0, 0.75, 0.5],
                    "shift_length_in_sec": [0.75, 0.625, 0.5, 0.375, 0.25],
                    "multiscale_weights": [1, 1, 1, 1, 1],
                    "save_embeddings": False,
                },
            },
            "clustering": {
                "parameters": {
                    "oracle_num_speakers": num_speakers is not None,
                    "max_num_speakers": 12,
                    "enhanced_count_thres": 80,
                    "max_rp_threshold": 0.25,
                    "sparse_search_volume": 30,
                    "maj_vote_spk_count": False,
                },
            },
            "msdd_model": {
                "model_path": MSDD_MODEL_PATH,
                "parameters": {
                    "use_speaker_model_from_ckpt": True,
                    "infer_batch_size": 25,
                    "sigmoid_threshold": [0.5, 0.5],
                    "seq_eval_mode": False,
                    "split_infer": True,
                    "diar_eval_settings": [
                        [0.25, True],
                        [0.25, False],
                        [0.0, False],
                    ],
                },
            },
        },
    }

    # If num_speakers is provided, override oracle count
    if num_speakers is not None:
        cfg_dict["diarizer"]["clustering"]["parameters"]["oracle_num_speakers"] = True
        cfg_dict["diarizer"]["clustering"]["parameters"]["max_num_speakers"] = num_speakers

    return OmegaConf.create(cfg_dict)


def _parse_rttm(rttm_path: str) -> list[dict]:
    """Parse RTTM file into list of segment dicts sorted by start time."""
    if not os.path.exists(rttm_path):
        logger.error(f"RTTM file not found: {rttm_path}")
        return []

    segments = []
    with open(rttm_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("SPEAKER"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            # RTTM format: SPEAKER file 1 start duration <NA> <NA> speaker <NA>
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            segments.append({
                "speaker": speaker,
                "start": round(start, 3),
                "end": round(start + duration, 3),
            })

    segments.sort(key=lambda s: s["start"])
    return segments
