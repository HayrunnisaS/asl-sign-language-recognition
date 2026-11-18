import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import gradio as gr

# 1. Cihaz ve Eğitilmiş Modeli Yükleme
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("best_asl_model.pth", map_location=DEVICE)
class_names = checkpoint['class_names']

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(DEVICE)
model.eval()

# 2. Görüntü Ön İşleme (Eğitimdeki ile birebir aynı)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Tahmin ve Kelime Birleştirme Fonksiyonları
def predict_sign(image, current_word):
    if image is None:
        return {}, current_word
    
    img_tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
    
    # En yüksek 3 ihtimali al
    top3_prob, top3_idx = torch.topk(probs, 3)
    results = {class_names[idx]: float(prob) for prob, idx in zip(top3_prob, top3_idx)}
    
    predicted_letter = class_names[top3_idx[0]]
    updated_word = current_word + predicted_letter
    return results, updated_word

def add_space(current_word):
    return current_word + " "

def clear_word():
    return ""

# 4. Gradio Web Arayüzü
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤟 İşaret Dili (ASL) Harf Tanıma ve Kelime Oluşturucu")
    gr.Markdown("Tek elle yapılan işaret dili harfi fotoğrafını yükleyin; model harfi tahmin edip aşağıdaki metne eklesin.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="El Fotoğrafı Yükle")
            submit_btn = gr.Button("Harfi Tahmin Et ve Ekle", variant="primary")
            with gr.Row():
                space_btn = gr.Button("Boşluk Ekle")
                clear_btn = gr.Button("Metni Temizle")
        
        with gr.Column():
            label_output = gr.Label(num_top_classes=3, label="Model Tahminleri (Güven Oranı)")
            word_output = gr.Textbox(label="Oluşturulan Kelime / Metin", value="", interactive=False)
            
    submit_btn.click(fn=predict_sign, inputs=[image_input, word_output], outputs=[label_output, word_output])
    space_btn.click(fn=add_space, inputs=[word_output], outputs=[word_output])
    clear_btn.click(fn=clear_word, inputs=[], outputs=[word_output])

if __name__ == "__main__":
    demo.launch()