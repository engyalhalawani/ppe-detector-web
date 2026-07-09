import requests
import json
from pathlib import Path

# Paths
BASE_URL = "http://127.0.0.1:8000"
PROJECT_DIR = Path(__file__).parent
test_image_path = PROJECT_DIR / "output" / "result_001.jpg"

if not test_image_path.exists():
    print(f"[FAIL] Test image not found at {test_image_path}. Trying another image...")
    # Find any image in output/
    images = list((PROJECT_DIR / "output").glob("*.jpg"))
    if images:
        test_image_path = images[0]
        print(f"Found image: {test_image_path}")
    else:
        print("[FAIL] No images found for testing.")
        exit(1)

print(f"Uploading {test_image_path.name} to {BASE_URL}/detect...")

# 1. Test /detect (JSON response)
with open(test_image_path, "rb") as f:
    files = {"file": (test_image_path.name, f, "image/jpeg")}
    response = requests.post(f"{BASE_URL}/detect", files=files, params={"conf_threshold": 0.25})

if response.status_code == 200:
    print("\n[OK] /detect JSON Response:")
    print(json.dumps(response.json(), indent=2))
else:
    print(f"[FAIL] /detect failed with status code {response.status_code}: {response.text}")

# 2. Test /detect-image (image response)
print(f"\nUploading {test_image_path.name} to {BASE_URL}/detect-image...")
with open(test_image_path, "rb") as f:
    files = {"file": (test_image_path.name, f, "image/jpeg")}
    response = requests.post(f"{BASE_URL}/detect-image", files=files, params={"conf_threshold": 0.25})

if response.status_code == 200:
    out_path = PROJECT_DIR / "output" / "test_api_annotated.jpg"
    out_path.write_bytes(response.content)
    print(f"[OK] /detect-image succeeded! Saved to {out_path}")
else:
    print(f"[FAIL] /detect-image failed with status code {response.status_code}: {response.text}")
