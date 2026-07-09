import gradio as gr
from ultralytics import YOLO
import ssl
import numpy as np
import os
from PIL import Image

ssl._create_default_https_context = ssl._create_unverified_context

# تحميل نموذج YOLO لرسم المربعات
if os.path.exists("ppe_model.pt"):
    model = YOLO("ppe_model.pt")
else:
    model = YOLO("yolov8n.pt")

def generate_safety_report(detections):
    """توليد تقرير سلامة احترافي من نتائج YOLO"""
    if not detections:
        return "⚠️ لم يتم اكتشاف أي شخص في الصورة."

    # تصنيف المكتشفات
    persons = 0
    no_hardhat = 0
    no_vest = 0
    no_mask = 0
    hardhat = 0
    vest = 0
    mask = 0
    gloves = 0
    no_gloves = 0
    boots = 0
    no_boots = 0

    for cls_name in detections:
        name = cls_name.lower()
        if "person" in name:
            persons += 1
        if "no-hardhat" in name or "no hardhat" in name:
            no_hardhat += 1
        elif "hardhat" in name or "helmet" in name:
            hardhat += 1
        if "no-safety vest" in name or "no safety vest" in name or "no-vest" in name:
            no_vest += 1
        elif "safety vest" in name or "vest" in name:
            vest += 1
        if "no-mask" in name or "no mask" in name:
            no_mask += 1
        elif "mask" in name:
            mask += 1
        if "no-gloves" in name or "no gloves" in name:
            no_gloves += 1
        elif "gloves" in name or "glove" in name:
            gloves += 1
        if "no-boots" in name or "no boots" in name:
            no_boots += 1
        elif "boots" in name or "boot" in name:
            boots += 1

    # بناء التقرير
    report_lines = []
    report_lines.append("# 🦺 تقرير السلامة المهنية (HSE)")
    report_lines.append("=" * 40)
    report_lines.append(f"\n📊 **ملخص الفحص:**")
    report_lines.append(f"- عدد الأشخاص المكتشفين: **{persons} شخص**")
    report_lines.append(f"- إجمالي المخالفات المكتشفة: **{no_hardhat + no_vest + no_mask + no_gloves + no_boots} مخالفة**")

    report_lines.append(f"\n🔍 **تفاصيل المعدات (كل معدة على حدة):**")

    # الخوذة - دائماً تظهر
    if hardhat > 0 and no_hardhat == 0:
        report_lines.append(f"✅ الخوذة (Hardhat): موجودة ({hardhat} شخص يرتديها)")
    elif hardhat > 0 and no_hardhat > 0:
        report_lines.append(f"⚠️ الخوذة (Hardhat): {hardhat} يرتديها | {no_hardhat} لا يرتديها ❌")
    elif no_hardhat > 0:
        report_lines.append(f"❌ الخوذة (Hardhat): غائبة - {no_hardhat} شخص لا يرتديها!")
    else:
        report_lines.append(f"➖ الخوذة (Hardhat): لم يتم اكتشافها في الصورة")

    # السترة العاكسة - دائماً تظهر
    if vest > 0 and no_vest == 0:
        report_lines.append(f"✅ السترة العاكسة (Safety Vest): موجودة ({vest} شخص يرتديها)")
    elif vest > 0 and no_vest > 0:
        report_lines.append(f"⚠️ السترة العاكسة (Safety Vest): {vest} يرتديها | {no_vest} لا يرتديها ❌")
    elif no_vest > 0:
        report_lines.append(f"❌ السترة العاكسة (Safety Vest): غائبة - {no_vest} شخص لا يرتديها!")
    else:
        report_lines.append(f"➖ السترة العاكسة (Safety Vest): لم يتم اكتشافها في الصورة")

    # الكمامة - دائماً تظهر
    if mask > 0 and no_mask == 0:
        report_lines.append(f"✅ الكمامة (Mask): موجودة ({mask} شخص يرتديها)")
    elif mask > 0 and no_mask > 0:
        report_lines.append(f"⚠️ الكمامة (Mask): {mask} يرتديها | {no_mask} لا يرتديها ❌")
    elif no_mask > 0:
        report_lines.append(f"❌ الكمامة (Mask): غائبة - {no_mask} شخص لا يرتديها!")
    else:
        report_lines.append(f"➖ الكمامة (Mask): لم يتم اكتشافها في الصورة")

    # القفازات - دائماً تظهر
    if gloves > 0 and no_gloves == 0:
        report_lines.append(f"✅ القفازات (Gloves): موجودة ({gloves} شخص يرتديها)")
    elif gloves > 0 and no_gloves > 0:
        report_lines.append(f"⚠️ القفازات (Gloves): {gloves} يرتديها | {no_gloves} لا يرتديها ❌")
    elif no_gloves > 0:
        report_lines.append(f"❌ القفازات (Gloves): غائبة - {no_gloves} شخص لا يرتديها!")
    else:
        report_lines.append(f"➖ القفازات (Gloves): لم يتم اكتشافها في الصورة")

    # الأحذية - دائماً تظهر
    if boots > 0 and no_boots == 0:
        report_lines.append(f"✅ الأحذية الواقية (Boots): موجودة ({boots} شخص يرتديها)")
    elif boots > 0 and no_boots > 0:
        report_lines.append(f"⚠️ الأحذية الواقية (Boots): {boots} يرتديها | {no_boots} لا يرتديها ❌")
    elif no_boots > 0:
        report_lines.append(f"❌ الأحذية الواقية (Boots): غائبة - {no_boots} شخص لا يرتديها!")
    else:
        report_lines.append(f"➖ الأحذية الواقية (Boots): لم يتم اكتشافها في الصورة")

    # الحكم النهائي
    total_violations = no_hardhat + no_vest + no_mask + no_gloves + no_boots
    report_lines.append(f"\n{'='*40}")
    if total_violations == 0:
        report_lines.append("✅ **الحكم: الموقع آمن - جميع العمال ملتزمون بمعدات السلامة!**")
    elif total_violations <= 2:
        report_lines.append(f"⚠️ **الحكم: تنبيه - يوجد {total_violations} مخالفة تحتاج تصحيحاً فورياً!**")
    else:
        report_lines.append(f"🚨 **الحكم: خطر - يوجد {total_violations} مخالفة! يجب إيقاف العمل وتصحيح الأوضاع فوراً!**")

    return "\n".join(report_lines)


def detect_ppe(image):
    if image is None:
        return None, "⚠️ يرجى رفع صورة أولاً."

    # رسم المربعات باستخدام YOLO
    results = model(image)
    annotated_img = results[0].plot()

    # استخراج أسماء الكلاسات المكتشفة
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = results[0].names[cls_id]
        detections.append(cls_name)

    # توليد التقرير من النتائج
    report = generate_safety_report(detections)

    return annotated_img, report


with gr.Blocks(
    title="PPE Detector - كاشف معدات السلامة",
    theme=gr.themes.Base(
        primary_hue="blue",
        secondary_hue="cyan",
    ),
    css="""
    .gradio-container { direction: rtl; }
    h1, h2, h3, p, label { text-align: right !important; }
    .header-text { text-align: center !important; }
    """
) as app:
    gr.HTML("""
    <div style="text-align:center; padding: 20px; background: linear-gradient(135deg, #1e3a5f, #0d7377); border-radius: 12px; margin-bottom: 20px;">
        <h1 style="color: white; font-size: 2em; margin: 0;">🦺 كاشف معدات السلامة بالذكاء الاصطناعي</h1>
        <p style="color: #a8d8ea; margin: 8px 0 0 0; font-size: 1.1em;">يستخدم نموذج YOLO للكشف عن معدات الحماية الشخصية وإنتاج تقرير سلامة احترافي</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="numpy", label="📸 ارفع صورة العمال هنا")
            submit_btn = gr.Button("🔍 تحليل الصورة وإنتاج تقرير السلامة", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_image = gr.Image(type="numpy", label="🖼️ الصورة المعالجة (YOLO)")
            output_text = gr.Textbox(
                label="📋 تقرير السلامة المفصل",
                lines=18,
                rtl=True
            )

    submit_btn.click(fn=detect_ppe, inputs=[input_image], outputs=[output_image, output_text])

    gr.HTML("""
    <div style="text-align:center; padding: 10px; color: #888; font-size: 0.9em; margin-top: 10px;">
        <p>تم تطويره باستخدام YOLOv8 | AI-Powered PPE Detection System</p>
    </div>
    """)

if __name__ == "__main__":
    app.launch(share=True)
