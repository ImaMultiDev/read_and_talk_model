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
# IMPORTAMOS TrOCRProcessor de la biblioteca transformers de Python
from transformers import TrOCRProcessor, V

# GENERAMOS EL PROCESADOR
# Este sirve para procesar la imagen y después procesar la respuesta del modelo a texto
# Para ello le pasamos uno de los modelos disponibles en Hugginface "microsoft/trocr-base-handwritten"


processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
```

#### Crear requirements.txt

```bash
$ pip freeze > requirements.txt
```
