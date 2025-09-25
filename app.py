# TrOCRProcessor - Importamos el procesador de TrOCR
# VisionEncoderDecoderModel - Importamos el modelo con su arquitectura, pesos, sesgos,... para predecir
# Gradio - Biblioteca para interfaz simple
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import gradio as gr

# Generamos el procesador pasando uno de los modelos disponibles en Hugging-face
# https://huggingface.co/microsoft/trocr-base-handwritten
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

# Generamos el modelo entrenado con su arquitectura y procesado por defecto
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Usamos el Procesador y Modelo en la función
type = "cuda" if torch.cuda.is_available() else "cpu"
model.to(type)
print(f"using {type}") # (Opcional) Comprobar el procesador que usa nuestro equipo

# Función process 
def process (image):
    # Primero le pedimos a pytorch que no calcule las gradientes
    with torch.no_grad():
        # Usamos las funciones para procesar la imagen
        pixels = processor(image, return_tensors="pt").pixel_values.to(type)
        # Hacer predicts
        ids = model.generate(pixels)
        # Convertirlo a texto
        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    return text

# Crear una interfaz simple
# Le indicamos que la entrada será una imagen, la salida será un campo de texto.
# Cuando la imagen se suba llamará a la función "process"
iface = gr.Interface(
    fn=process,
    inputs=gr.Image(type="pil", label="Imagen"),
    outputs= gr.Textbox(label="Texto encontrado")
)
iface.launch(debug=True)