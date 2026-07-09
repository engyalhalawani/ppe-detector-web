"""
PPE Detector - يكشف الخوذة، السترة العاكسة، والأشخاص في الصور
Detects: Helmet, No Helmet, Safety Vest, No Vest, Person

يستخدم YOLOv8 مع نموذج مدرب على PPE
Uses YOLOv8 with a PPE-trained model
"""

import os
import sys
import time
import json
import shutil
import argparse
import urllib.request
import urllib.parse
import ssl
from pathlib import Path
from datetime import datetime

# تجاوز أخطاء SSL عند تحميل النماذج أو الصور
ssl._create_default_https_context = ssl._create_unverified_context

# ─────────────────────────────────────────────
#  تحقق من وجود المكتبات المطلوبة
# ─────────────────────────────────────────────
MISSING = []
try:
    import cv2
except ImportError:
    MISSING.append("opencv-python")

try:
    from ultralytics import YOLO
except ImportError:
    MISSING.append("ultralytics")

try:
    import requests
except ImportError:
    MISSING.append("requests")

try:
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
except ImportError:
    MISSING.append("pillow")
    MISSING.append("numpy")

if MISSING:
    print("❌ المكتبات التالية غير مثبتة:")
    for pkg in MISSING:
        print(f"   pip install {pkg}")
    print("\nقم بتشغيل:")
    print("   pip install ultralytics opencv-python requests pillow numpy")
    sys.exit(1)

# ─────────────────────────────────────────────
#  الإعدادات
# ─────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
OUTPUT_DIR  = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ألوان الكلاسات
CLASS_COLORS = {
    "Helmet":        (0,   200,  80),   # أخضر
    "No Helmet":     (255,  50,  50),   # أحمر
    "Safety Vest":   (0,   180, 255),   # أزرق
    "No Vest":       (255, 160,   0),   # برتقالي
    "Person":        (200, 200, 200),   # رمادي
}

# أسماء الكلاسات في النموذج (قد تختلف حسب النموذج المستخدم)
# سيتم تحديثها تلقائياً بعد تحميل النموذج
CLASS_MAP = {}


# ─────────────────────────────────────────────
#  تحميل نموذج PPE
# ─────────────────────────────────────────────
def load_ppe_model():
    """تحميل نموذج YOLOv8 المدرب على PPE"""

    # نموذج PPE جاهز من HuggingFace
    model_url = "https://huggingface.co/Hansung-Cho/yolov8-ppe-detection/resolve/main/best.pt"
    model_path = PROJECT_DIR / "ppe_model.pt"

    if not model_path.exists():
        print("📥 جاري تحميل نموذج PPE ...")
        print("   (قد يستغرق دقيقة أو أكثر)")
        try:
            urllib.request.urlretrieve(model_url, model_path,
                reporthook=lambda b, bsize, total: print(
                    f"\r   {min(100, int(b*bsize*100/total))}%", end="", flush=True)
                    if total > 0 else None)
            print("\n✅ تم تحميل النموذج بنجاح")
        except Exception as e:
            print(f"\n⚠️  لم يتمكن من تحميل النموذج المخصص: {e}")
            print("   سيتم استخدام YOLOv8n (الكشف العام)")
            model_path = "yolov8n.pt"
    
    model = YOLO(str(model_path))
    
    # بناء خريطة الكلاسات
    global CLASS_MAP
    CLASS_MAP = model.names  # {0: 'Helmet', 1: 'No Helmet', ...}
    print(f"📋 الكلاسات المتاحة: {list(CLASS_MAP.values())}")
    
    return model


# ─────────────────────────────────────────────
#  البحث عن صور من الإنترنت
# ─────────────────────────────────────────────
def search_images_online(query: str, num_images: int = 10) -> list[str]:
    """
    البحث عن صور من DuckDuckGo (لا يحتاج API key)
    Returns list of image URLs
    """
    print(f"\n🔍 البحث عن: '{query}' ...")
    
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    # DuckDuckGo Image Search
    encoded_query = urllib.parse.quote(query)
    search_url = f"https://duckduckgo.com/?q={encoded_query}&iax=images&ia=images"
    
    image_urls = []
    
    try:
        import requests
        # الحصول على vqd token من DuckDuckGo
        resp = requests.get(search_url, headers=headers, timeout=10)
        
        import re
        vqd_match = re.search(r'vqd=([\d-]+)', resp.text)
        if not vqd_match:
            vqd_match = re.search(r'"vqd":"(.*?)"', resp.text)
        
        if vqd_match:
            vqd = vqd_match.group(1)
            api_url = (
                f"https://duckduckgo.com/i.js"
                f"?q={encoded_query}&o=json&vqd={vqd}"
                f"&f=,,,,,&p=1"
            )
            api_resp = requests.get(api_url, headers=headers, timeout=10)
            data = api_resp.json()
            
            for item in data.get("results", [])[:num_images]:
                image_urls.append(item.get("image", ""))
        
        if not image_urls:
            raise Exception("لم يتم العثور على نتائج من DuckDuckGo")
            
    except Exception as e:
        print(f"   ⚠️  DuckDuckGo: {e}")
        print("   📦 محاولة Unsplash ...")
        
        # Unsplash بديل
        unsplash_url = (
            f"https://source.unsplash.com/featured/?{encoded_query}"
        )
        # fallback to known static images if DDG fails
        image_urls = [
            "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg",
            "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg",
            "https://images.unsplash.com/photo-1504307651254-35680f356f12?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
            "https://images.unsplash.com/photo-1541888086225-eb713524b898?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80"
        ] * (num_images // 4 + 1)
        image_urls = image_urls[:num_images]

    print(f"   ✅ تم العثور على {len(image_urls)} رابط صورة")
    return [url for url in image_urls if url]


# ─────────────────────────────────────────────
#  تحميل صورة من URL
# ─────────────────────────────────────────────
def download_image(url: str, idx: int) -> np.ndarray | None:
    """تحميل صورة من URL وتحويلها إلى numpy array"""
    try:
        import requests
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=15, stream=True)
        resp.raise_for_status()
        
        import io
        img_array = np.frombuffer(resp.content, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return None
            
        # تغيير الحجم إذا كانت كبيرة جداً
        h, w = img.shape[:2]
        if max(h, w) > 1280:
            scale = 1280 / max(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)))
        
        return img
        
    except Exception as e:
        print(f"   ⚠️  فشل تحميل الصورة {idx}: {e}")
        return None


# ─────────────────────────────────────────────
#  رسم النتائج على الصورة
# ─────────────────────────────────────────────
def draw_detections(img: np.ndarray, results, model_names: dict, conf_threshold: float = 0.30) -> tuple[np.ndarray, dict]:
    """رسم مربعات الكشف والتسميات على الصورة"""
    
    stats = {
        "Helmet": 0,
        "No Helmet": 0,
        "Safety Vest": 0,
        "No Vest": 0,
        "Person": 0,
    }
    
    annotated = img.copy()
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id   = int(box.cls[0])
            conf     = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            raw_name = model_names.get(cls_id, "Unknown")
            
            # تطبيع أسماء الكلاسات
            label = normalize_class_name(raw_name)
            
            if conf < conf_threshold:   # تجاهل الاكتشافات منخفضة الثقة
                continue
            
            # الحصول على اللون
            color = get_class_color(label)
            
            # رسم المربع
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # نص التسمية
            display_text = f"{label} {conf:.0%}"
            
            # خلفية النص
            (tw, th), baseline = cv2.getTextSize(
                display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                annotated,
                (x1, y1 - th - 8),
                (x1 + tw + 4, y1),
                color, -1
            )
            cv2.putText(
                annotated, display_text,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2
            )
            
            # تحديث الإحصائيات
            for key in stats:
                if key.lower() in label.lower():
                    stats[key] += 1
                    break
    
    return annotated, stats


def normalize_class_name(name: str) -> str:
    """تطبيع أسماء الكلاسات للعرض"""
    name_map = {
        "helmet":       "Helmet",
        "hard hat":     "Helmet",
        "hardhat":      "Helmet",
        "no helmet":    "No Helmet",
        "no-helmet":    "No Helmet",
        "no hard hat":  "No Helmet",
        "no-hardhat":   "No Helmet",
        "vest":         "Safety Vest",
        "safety vest":  "Safety Vest",
        "no vest":      "No Vest",
        "no-vest":      "No Vest",
        "no-safety vest": "No Vest",
        "person":       "Person",
        "worker":       "Person",
        "people":       "Person",
    }
    return name_map.get(name.lower(), name)


def get_class_color(label: str) -> tuple:
    """إرجاع لون BGR للكلاس"""
    for key, color in CLASS_COLORS.items():
        if key.lower() in label.lower():
            return tuple(reversed(color))  # RGB → BGR
    return (150, 150, 150)


# ─────────────────────────────────────────────
#  لوحة الإحصائيات
# ─────────────────────────────────────────────
def add_stats_panel(img: np.ndarray, stats: dict, img_name: str) -> np.ndarray:
    """إضافة لوحة إحصائيات في أسفل الصورة"""
    
    panel_h = 80
    h, w = img.shape[:2]
    
    panel = np.zeros((panel_h, w, 3), dtype=np.uint8)
    panel[:] = (25, 25, 35)  # خلفية داكنة
    
    # خط فاصل
    cv2.line(panel, (0, 0), (w, 0), (80, 80, 100), 2)
    
    # عنوان
    cv2.putText(panel, img_name[:40],
        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (180, 180, 200), 1)
    
    # الإحصائيات
    x_pos = 10
    for label, count in stats.items():
        color = get_class_color(label)
        text  = f"{label}: {count}"
        
        cv2.circle(panel, (x_pos + 6, 50), 6, color, -1)
        cv2.putText(panel, text,
            (x_pos + 16, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (220, 220, 220), 1)
        
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        x_pos += tw + 30
        
        if x_pos > w - 150:
            break
    
    return np.vstack([img, panel])


# ─────────────────────────────────────────────
#  معالجة صورة واحدة
# ─────────────────────────────────────────────
def process_image(model, img: np.ndarray, name: str, save_path: Path, conf_threshold: float = 0.30) -> dict:
    """تشغيل الكشف على صورة وحفظ النتيجة"""
    
    results = model(img, verbose=False)
    annotated, stats = draw_detections(img, results, CLASS_MAP, conf_threshold)
    
    # إضافة لوحة الإحصائيات
    final_img = add_stats_panel(annotated, stats, name)
    
    cv2.imwrite(str(save_path), final_img)
    
    return stats


# ─────────────────────────────────────────────
#  البحث والكشف الرئيسي
# ─────────────────────────────────────────────
def search_and_detect(
    query: str = "construction worker safety helmet vest site",
    num_images: int = 10,
    model_path: str = None
):
    """البحث عن صور وتطبيق كشف PPE عليها"""
    
    print("=" * 60)
    print("🦺  PPE Detector - كاشف معدات السلامة")
    print("=" * 60)
    
    # تحميل النموذج
    if model_path and Path(model_path).exists():
        print(f"📂 تحميل النموذج من: {model_path}")
        model = YOLO(model_path)
        global CLASS_MAP
        CLASS_MAP = model.names
    else:
        model = load_ppe_model()
    
    print(f"\n🗂️  مجلد النتائج: {OUTPUT_DIR}")
    
    # البحث عن صور
    image_urls = search_images_online(query, num_images)
    
    all_stats = []
    processed = 0
    
    for idx, url in enumerate(image_urls, 1):
        print(f"\n[{idx}/{len(image_urls)}] معالجة الصورة ...")
        
        img = download_image(url, idx)
        if img is None:
            continue
        
        save_path = OUTPUT_DIR / f"result_{processed+1:03d}.jpg"
        
        try:
            stats = process_image(model, img, f"Image {idx}", save_path)
            all_stats.append(stats)
            processed += 1
            
            print(f"   ✅ تم الحفظ: {save_path.name}")
            print(f"   📊 {stats}")
            
        except Exception as e:
            print(f"   ❌ خطأ في المعالجة: {e}")
    
    # ملخص نهائي
    print("\n" + "=" * 60)
    print(f"✅ تمت معالجة {processed} صورة من أصل {len(image_urls)}")
    print(f"📁 النتائج محفوظة في: {OUTPUT_DIR}")
    
    if all_stats:
        total = {k: sum(s.get(k, 0) for s in all_stats) for k in CLASS_COLORS}
        print("\n📊 الإجمالي الكلي:")
        for label, count in total.items():
            color_indicator = "🟢" if label in ["Helmet", "Safety Vest"] else (
                              "🔴" if label in ["No Helmet", "No Vest"] else "⚪")
            print(f"   {color_indicator} {label}: {count}")
    
    # حفظ ملف JSON للإحصائيات
    report = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "total_images": processed,
        "per_image_stats": all_stats,
        "totals": total if all_stats else {},
    }
    report_path = OUTPUT_DIR / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n📄 تقرير مفصل: {report_path}")
    
    return all_stats


# ─────────────────────────────────────────────
#  معالجة صورة محلية
# ─────────────────────────────────────────────
def process_local_image(image_path: str, model_path: str = None):
    """معالجة صورة محلية موجودة على الجهاز"""
    
    print("=" * 60)
    print("🦺  PPE Detector - تحليل صورة محلية")
    print("=" * 60)
    
    path = Path(image_path)
    if not path.exists():
        print(f"❌ الملف غير موجود: {image_path}")
        return
    
    if model_path and Path(model_path).exists():
        model = YOLO(model_path)
        global CLASS_MAP
        CLASS_MAP = model.names
    else:
        model = load_ppe_model()
    
    img = cv2.imread(str(path))
    if img is None:
        print(f"❌ لم يتمكن من قراءة الصورة: {image_path}")
        return
    
    save_path = OUTPUT_DIR / f"result_{path.stem}_detected.jpg"
    stats = process_image(model, img, path.name, save_path)
    
    print(f"\n✅ النتيجة محفوظة: {save_path}")
    print(f"📊 الاكتشافات: {stats}")
    
    # عرض الصورة
    result_img = cv2.imread(str(save_path))
    cv2.imshow(f"PPE Detection - {path.name}", result_img)
    print("\nاضغط أي مفتاح لإغلاق النافذة ...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
#  نقطة الدخول
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="🦺 PPE Detector - كاشف معدات السلامة (خوذة، سترة، شخص)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
أمثلة على الاستخدام:
  # البحث عن صور من الإنترنت:
  python ppe_detector.py --search "construction worker helmet"

  # معالجة صورة محلية:
  python ppe_detector.py --image "C:/photos/worker.jpg"

  # تحديد عدد الصور:
  python ppe_detector.py --search "safety vest worker" --num 20

  # استخدام نموذج مخصص:
  python ppe_detector.py --search "worker" --model my_model.pt
        """
    )
    
    parser.add_argument(
        "--search", type=str,
        default="construction worker safety helmet vest PPE site",
        help="استعلام البحث عن صور من الإنترنت"
    )
    parser.add_argument(
        "--image", type=str,
        help="مسار صورة محلية للتحليل"
    )
    parser.add_argument(
        "--num", type=int, default=10,
        help="عدد الصور للبحث (افتراضي: 10)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="مسار نموذج YOLOv8 مخصص (.pt)"
    )
    
    args = parser.parse_args()
    
    if args.image:
        process_local_image(args.image, args.model)
    else:
        search_and_detect(args.search, args.num, args.model)


if __name__ == "__main__":
    main()
