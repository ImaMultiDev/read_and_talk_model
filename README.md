### Modelos para lectura de texto escrito

#### Fuentes de Modelos

1. [EasyOCR](https://github.com/JaidedAI/EasyOCR)

   - Soporte para más de 80 lenguajes
   - Todavía no posee soporte para texto escrito a mano

2. [TrOCR](https://huggingface.co/docs/transformers/model_doc/trocr)
   - Uso de transformers
   - Opción de cuantización
   - **Modelos con soporte para texto escrito a mano**

---

#### Iniciar entorno virtual de python

```bash
$ python -m venv env
$ source env/Scripts/activate
```

#### Instalar la biblioteca "transformers"

**`Esta ya trae dentro TrOCR`**
https://huggingface.co/microsoft/trocr-base-handwritten

```bash
$ pip install -q transformers
```

#### Instalar "gradio" para que nos permita crear una interfaz web rapida para probar el prototipo

```bash
pip install gradio
```

#### Instalar "Torch"

###### Con "Cuda12.6"

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

###### Con "CPU"

```bash
pip3 install torch torchvision
```

`Una vez instaladas las bibliotecas necesarias podemos generar el prototipo`

#### app.py

```python
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
    inputs=gr.Image(type="pil", label="Imagen"),
    outputs= gr.Textbox(label="Texto encontrado")
)
iface.launch(debug=True)
```

#### Probamos a ejecutar la aplicación

```bash
python ./app.py
```

**Abrirá la interfaz pero por el momento la salida va a ser la cadena de texto "Nada" como indicamos en la función process.
Vamos a implementar la lógica con el procesador y el modelo ahora**

#### app.py

```python
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
```

#### Prueba de modelo "HandWritter"

Accedemos a la dirección de puerto local http://127.0.0.1:7860 y comprobamos una imagen de un texto escrito a mano
<image src="public\images\handwrittermodel_test_01.jpg" alt="prueba 01 del modelo HandWritter" width=100%>
Bien, pero ahora probemos con un texto escrito en varias líneas
<image src="public\images\handwrittermodel_test_02.jpg" alt="prueba 02 del modelo HandWritter" width=100%>
Como podemos observar aquí tendríamos un problema, el cual podemos reparar de forma simple con una biblioteca de separación de líneas en las imágenes.

#### Crear requirements.txt

```bash
$ pip freeze > requirements.txt
```
