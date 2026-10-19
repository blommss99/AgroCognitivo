# AgroCognitivo — Invernadero inteligente de tomates

Prototipo académico para la actividad de Computación Cognitiva en Big Data.

## Funcionalidades
- Generación de datos simulados para demostración.
- Carga opcional de CSV propio.
- Análisis exploratorio con tendencias, histogramas y tabla.
- Reglas configurables de alerta para temperatura, humedad relativa, humedad del suelo y pH.
- Detección no supervisada de observaciones atípicas mediante Isolation Forest.
- Interfaz en español e inglés y reporte bilingüe descargable.
- Descarga de datos analizados en CSV.

## Requisitos
Python 3.10 o superior recomendado.

## Instalación local
En una terminal, dentro de esta carpeta:

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Instala las dependencias y ejecuta:
```bash
pip install -r requirements.txt
streamlit run app.py
```

Se abrirá la aplicación en el navegador local.

## Formato de CSV
El archivo debe contener estas columnas exactas:
- `fecha` (fecha/hora reconocible por pandas)
- `temperatura_c`
- `humedad_relativa_pct`
- `humedad_suelo_pct`
- `ph`

Ejemplo:
```csv
fecha,temperatura_c,humedad_relativa_pct,humedad_suelo_pct,ph
2026-10-18 08:00:00,24.5,68,55,6.1
2026-10-18 09:00:00,25.2,70,53,6.0
```

## Nota académica y limitaciones
Los datos por defecto son sintéticos, no vienen de sensores ni de una empresa real. El sistema combina reglas explícitas con Isolation Forest. La detección de anomalías identifica valores estadísticamente atípicos, pero no confirma por sí sola un problema de cultivo. Los rangos son orientativos para un prototipo y deben validarse para el cultivo, sustrato y método de medición concretos. El programa no controla equipos físicos.

## Publicación
Puede publicarse como aplicación Streamlit mediante un repositorio GitHub y Streamlit Community Cloud. Antes de publicar, sube `app.py` y `requirements.txt` al repositorio. La carpeta `.venv` no se debe subir.
