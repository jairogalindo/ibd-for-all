from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import openai

app = FastAPI(title="Asistente de Co-Construcción en IBD")

# Permitir conexiones directas desde tu entorno web de CodeSandbox
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos de datos para validar la información entrante de React
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

    # Prompt del Sistema: Define el comportamiento del Par Crítico
    system_prompt = f"""
    Eres un facilitador académico y par crítico experto en Investigación Basada en Diseño (IBD) en educación. 
    Tu enfoque es andragógico, basado en el acompañamiento, el diálogo horizontal y la co-construcción de conocimiento entre pares.
    
    ESTADO ACTUAL DE LA INVESTIGACIÓN DEL ESTUDIANTE:
    - Contexto y Problema: "{campos.contextoProblema or 'No ingresado aún'}"
    - Brecha Pedagógica: "{campos.brechaEducativa or 'No ingresada aún'}"
    - Propuesta Inicial (Diseño 1.0): "{campos.propuestaDisDiseno or 'No ingresada aún'}"
    
    DIRECTRICES DE INTERACCIÓN:
    1. Actúa como un mediador pedagógico, no como un evaluador punitivo. Valora los saberes previos.
    2. Si el estudiante expresa dudas conceptuales (ej: qué es un concepto) o dice "no sé", detén la presión metodológica, aclara la duda usando un andamiaje conceptual claro y cercano, y luego conéctalo con su proyecto.
    3. Nunca redactes la tesis ni des las respuestas definitivas. Devuelve el protagonismo cognitivo al estudiante a través de UNA sola pregunta generativa a la vez.
    4. Usa la información del estado de su investigación para que tus preguntas sean profundamente contextualizadas.
    """

    # Formatear el historial de conversación para la API de OpenAI
    mensajes_para_ia = [{"role": "system", "content": system_prompt}]
    for msg in historial_estudiante:
        role_mapeado = "user" if msg.role == "user" else "assistant"
        mensajes_para_ia.append({"role": role_mapeado, "content": msg.content})

    try:
        # Llamada al modelo utilizando la API Key inyectada en Render
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o",  # Puedes cambiar a "gpt-4o-mini" para optimizar consumos de saldo
            messages=mensajes_para_ia,
            temperature=0.7
        )
        respuesta_ia = response.choices[0].message.content
        return {"success": True, "reply": respuesta_ia}

    except Exception as e:
        return {
            "success": False, 
            "reply": f"Lo siento, Jairo Alberto. Ocurrió un inconveniente técnico al conectar con mi motor cognitivo backend: {str(e)}"
        }
