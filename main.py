from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import io
from PIL import Image
from core.cors import configure_cors

app = FastAPI()

#Configuración 
configure_cors(app)

# Cargar el modelo
modelo_ruta = 'model/modelo_clasificacion_basura.h5'
modelo = load_model(modelo_ruta)

# Preprocesar la imagen
def preprocesar_imagen(img, target_size):
    img = cv2.resize(img, target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    img = np.array(img)
    img_array = preprocesar_imagen(img, target_size=(100, 100))
    predicciones = modelo.predict(img_array)
    prediccion_clase = np.argmax(predicciones, axis=1)
    print(f"prediccion:: {int(prediccion_clase[0])} ")
    return {"prediction": int(prediccion_clase[0])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)