from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import requests

app = FastAPI(title="Asistente de Co-Construcción en IBD")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CamposIBD(BaseModel):
    contextoProblema: str
    brechaEducativa: str
    propuestaDisDiseno: str

class MensajeChat(BaseModel):
    role: str
    content: str

class PeticionChat(BaseModel):
    campos: CamposIBD
    historial: List[MensajeChat]

@app.post("/api/chat")
async def procesar_dialogo_ibd(peticion: PeticionChat):
    campos = peticion.campos
    historial_estudiante = peticion.historial

    system_prompt = f"""
    Eres un facilitador académico y par crítico experto en Investigación Basada en Diseño (IBD) en educación. 
    Tu enfoque es andragógico, basado en el acompañamiento, el diálogo horizontal y la co-construcción.
    
    ESTADO ACTUAL DE LA INVESTIGACIÓN:
    - Contexto: "{campos.contextoProblema or 'No ingresado aún'}"
    - Brecha: "{campos.brechaEducativa or 'No ingresada aún'}"
    - Propuesta (Diseño 1.0): "{campos.propuestaDisDiseno or 'No ingresada aún'}"
    
    DIRECTRICES:
    1. Mediador pedagógico, valora saberes previos.
    2. Ante dudas o "no sé", da un andamiaje conceptual claro y corto.
    3. Una sola pregunta generativa al final. No redactes la tesis por él.
    """

    mensajes_para_groq = [{"role": "system", "content": system_prompt}]
    for msg in historial_estudiante:
        mensajes_para_groq.append({"role": msg.role, "content": msg.content})

    try:
        # Llamada directa y segura de servidor a servidor a Groq
        # Usamos requests para no necesitar librerías extra
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama3-70b-8192",
                "messages": mensajes_para_groq,
                "temperature": 0.7
            },
            timeout=10
        )
        
        datos = response.json()
        respuesta_ia = datos["choices"][0]["message"]["content"]
        return {"success": True, "reply": respuesta_ia}

    except Exception as e:
        return {"success": False, "reply": f"Error en backend: {str(e)}"}
