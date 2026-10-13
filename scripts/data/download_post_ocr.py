# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "pyarrow>=21.0.0",
# ]
# ///
"""Download PleIAs Post-OCR-Correction: flat (text, corrected_text) OCR-denoise parquet, passthrough."""

from acquire import acquire

if __name__ == "__main__":
    raise SystemExit(
        acquire(
            "post_ocr",
            repo_id="PleIAs/Post-OCR-Correction",
            shard_suffix=(".parquet",),
            source_kind="hf_snapshot",
            out_name="post-ocr",
        )
    )
