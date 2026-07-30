import os
import json
import yaml
from transformers import WhisperForConditionalGeneration

def package_stt(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["output_dir"]
    onnx_dir = "outputs/onnx_export_stt"
    fallback_dir = "outputs/packaged_hf_stt"

    model = WhisperForConditionalGeneration.from_pretrained(output_dir)

    onnx_success = False
    onnx_error = None
    try:
        from optimum.exporters.onnx import main_export
        main_export(
            model_name_or_path=output_dir,
            output=onnx_dir,
            task="automatic-speech-recognition",
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

    result = {"onnx_export_succeeded": onnx_success, "fallback_used": not onnx_success, "onnx_error": onnx_error, "onnx_dir": onnx_dir}
    with open("outputs/packaging_result_stt.json", "w") as f:
        json.dump(result, f, indent=2)

    return result

if __name__ == "__main__":
    result = package_stt("configs/hindi_stt.yaml")
    print(result)
