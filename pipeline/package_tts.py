import os
import json
import yaml
import torch
from parler_tts import ParlerTTSForConditionalGeneration

def package_tts(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model = ParlerTTSForConditionalGeneration.from_pretrained(cfg["output_dir"])
    onnx_dir = "outputs/onnx_export"
    fallback_dir = "outputs/packaged_hf"

    onnx_success = False
    onnx_error = None
    try:
        from optimum.exporters.onnx import main_export
        main_export(
            model_name_or_path=cfg["output_dir"],
            output=onnx_dir,
            task="text-to-audio",
        )
        onnx_success = True
        print(f"ONNX export succeeded: {onnx_dir}")
    except Exception as e:
        onnx_error = str(e)
        print(f"ONNX export failed: {onnx_error}")

    if not onnx_success:
        os.makedirs(fallback_dir, exist_ok=True)
        model.save_pretrained(fallback_dir)
        print(f"Fallback: saved standard HF format to {fallback_dir}")

    result = {"onnx_export_succeeded": onnx_success, "fallback_used": not onnx_success, "onnx_error": onnx_error}
    with open("outputs/packaging_result.json", "w") as f:
        json.dump(result, f, indent=2)

    return result

if __name__ == "__main__":
    result = package_tts("configs/hindi_tts.yaml")
    print(result)
