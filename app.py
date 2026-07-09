import gradio as gr
from ultralytics import YOLO
import ssl
import numpy as np
import os
import google.generativeai as genai
from PIL import Image

ssl._create_default_https_context = ssl._create_unverified_context

# تحميل نموذج YOLO لرسم المربعات
if os.path.exists("ppe_model.pt"):
    model = YOLO("ppe_model.pt")
else:
    model = YOLO("yolov8n.pt")

def detect_ppe_with_gemini(image, api_key):
    if image is None:
        return None, "يرجى رفع صورة أولاً."
    
    # رسم المربعات باستخدام YOLO
    if model is not None:
        results = model(image)
        annotated_img = results[0].plot()
    else:
        annotated_img = image

    # إذا لم يتم توفير مفتاح Gemini، نكتفي بالتحليل البسيط
    if not api_key or api_key.strip() == "":
        return annotated_img, "يرجى إدخال مفتاح Gemini API للحصول على تقرير مفصل."

    # تجهيز Gemini
    try:
        genai.configure(api_key=api_key.strip())
        # يفضل استخدام gemini-1.5-flash كونه سريع وممتاز في تحليل الصور
        try:
            pil_img = Image.fromarray(image)
            prompt = """
            أنت خبير في السلامة المهنية (HSE). قم بتحليل هذه الصورة التي تحتوي على عمال بدقة.
            اخبرني بالتفصيل:
            1. هل يرتدي الأشخاص في الصورة معدات الأمان المطلوبة (خوذة أمان، سترة عاكسة)؟
            2. اذكر بالتحديد ما الذي ينقص كل شخص (مثلاً: شخص يرتدي خوذة ولكن تنقصه السترة، أو شخص لا يرتدي شيئاً على الإطلاق).
            3. إذا كان الجميع ملتزمين، اذكر ذلك بوضوح.
            
            اكتب التقرير باللغة العربية بأسلوب واضح ومباشر مع استخدام نقاط (Bullet points).
            """
            
            gemini_model = genai.GenerativeModel('gemini-flash-latest')
            response = gemini_model.generate_content([prompt, pil_img])
        except Exception:
            gemini_model = genai.GenerativeModel('gemini-2.0-flash')
            response = gemini_model.generate_content([prompt, pil_img])
        report = response.text
        
    except Exception as e:
        report = f"❌ حدث خطأ أثناء الاتصال بـ Gemini API:\n{str(e)}"

    return annotated_img, report

with gr.Blocks(title="PPE Detector & Gemini") as app:
    gr.Markdown("# 🦺 واجهة كاشف معدات السلامة بالذكاء الاصطناعي (Gemini + YOLO)")
    gr.Markdown("تقوم هذه الواجهة برسم مربعات التحديد على الأشخاص، ثم ترسل الصورة إلى **Gemini Vision** لكتابة تقرير أمان دقيق.")
    
    with gr.Row():
        has_key = bool(os.environ.get("GEMINI_API_KEY"))
        api_key_input = gr.Textbox(label="مفتاح Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""), visible=not has_key)

        
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="الصورة المدخلة")
            submit_btn = gr.Button("تحليل الصورة باستخدام Gemini", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(type="numpy", label="الصورة المعالجة (YOLO)")
            output_text = gr.Textbox(label="تقرير السلامة المفصل (Gemini)", lines=12)
            
    submit_btn.click(fn=detect_ppe_with_gemini, inputs=[input_image, api_key_input], outputs=[output_image, output_text])

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7864, share=True)
