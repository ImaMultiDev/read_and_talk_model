# TrOCRProcessor - Importamos el procesador de TrOCR
# VisionEncoderDecoderModel - Importamos el modelo con su arquitectura, pesos, sesgos,... para predecir
# Gradio - Biblioteca para interfaz simple
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import gradio as gr

# Generamos el procesador pasando uno de los modelos disponibles en Hugging-face
# https://huggingface.co/microsoft/trocr-base-handwritten
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

# Generamos el modelo entrenado con su arquitectura y procesado por defecto
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")


# Función process 
def process (image):
    return "Nada"

# Crear una interfaz simple
# Le indicamos que la entrada será una imagen, la salida será un campo de texto.
# Cuando la imagen se suba llamará a la función "process"
iface = gr.Interface(
    fn=process,
    inputs=gr.Image(type="pill", label="Imagen"),
    outputs= gr.TextBox(label="Texto encontrado")
)
iface.launch(debug=True)