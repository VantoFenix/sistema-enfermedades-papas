import os
import sys
import time
import glob
import requests
import numpy as np

# Configuración del endpoint y del benchmark
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
HEALTH_URL = os.getenv("HEALTH_URL", "http://localhost:8000/health")
N_WARMUP = 10
N_CALLS = 150
LATENCY_THRESHOLD_MS = 150.0

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
test_dir = os.path.join(project_root, 'data', 'Potato_Particionado', 'test')

def find_sample_image(search_dir: str) -> str:
    """Busca automáticamente la primera imagen .jpg, .jpeg o .png disponible."""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    for ext in extensions:
        matches = glob.glob(os.path.join(search_dir, "**", ext), recursive=True)
        if matches:
            return matches[0]
            
    # Fallback en caso de que la ruta de test difiera
    fallback_dir = os.path.join(project_root, "data")
    for ext in extensions:
        matches = glob.glob(os.path.join(fallback_dir, "**", ext), recursive=True)
        if matches:
            return matches[0]
            
    raise FileNotFoundError(f"No se encontró ninguna imagen de prueba en {search_dir} ni en data/.")

def main():
    print("=" * 65)
    print("🔬 BENCHMARK DE LATENCIA REAL DE INFERENCIA - HOTPOTATO API")
    print("=" * 65)
    
    # 1. Localizar imagen real de prueba
    try:
        sample_image_path = find_sample_image(test_dir)
        print(f"[OK] Imagen de prueba localizada:")
        print(f"     -> {sample_image_path}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # 2. Verificar disponibilidad del servicio
    print(f"\n[INFO] Verificando conexión con el servicio en {HEALTH_URL}...")
    try:
        health_resp = requests.get(HEALTH_URL, timeout=5)
        if health_resp.status_code == 200:
            print(f"[OK] Servicio activo y saludable: {health_resp.json()}")
        else:
            print(f"[WARN] Healthcheck respondió con código {health_resp.status_code}: {health_resp.text}")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] No se pudo conectar con el servicio: {e}")
        print("Asegúrate de que la API esté corriendo: uvicorn servicio.main:app --port 8000")
        sys.exit(1)

    # 3. Fase de Warm-up
    print(f"\n[INFO] Ejecutando {N_WARMUP} llamadas de calentamiento (warm-up)...")
    with open(sample_image_path, "rb") as img_file:
        img_bytes = img_file.read()

    for i in range(1, N_WARMUP + 1):
        files = {"file": ("sample.jpg", img_bytes, "image/jpeg")}
        resp = requests.post(API_URL, files=files)
        if resp.status_code != 200:
            print(f"  [WARN] Falló llamada warm-up {i}: {resp.status_code} - {resp.text}")
    print("[OK] Warm-up finalizado exitosamente.")

    # 4. Fase de Medición Consecutiva (150 llamadas)
    print(f"\n[INFO] Ejecutando {N_CALLS} llamadas consecutivas a {API_URL}...")
    latencias_ms = []
    exitosas = 0
    ultima_respuesta = None

    for i in range(1, N_CALLS + 1):
        files = {"file": ("sample.jpg", img_bytes, "image/jpeg")}
        t_start = time.perf_counter()
        resp = requests.post(API_URL, files=files)
        t_end = time.perf_counter()
        
        if resp.status_code == 200:
            latency = (t_end - t_start) * 1000.0
            latencias_ms.append(latency)
            exitosas += 1
            ultima_respuesta = resp.json()
        else:
            print(f"  [WARN] Error en llamada {i}: {resp.status_code} - {resp.text}")
            
        if i % 30 == 0 or i == N_CALLS:
            print(f"  Progreso: {i}/{N_CALLS} solicitudes completadas...")

    if exitosas == 0:
        print("\n[ERROR] Ninguna solicitud fue procesada exitosamente.")
        sys.exit(1)

    # 5. Cálculo Estadístico
    latencias = np.array(latencias_ms)
    lat_min = np.min(latencias)
    lat_max = np.max(latencias)
    lat_mean = np.mean(latencias)
    lat_median = np.median(latencias)
    lat_p95 = np.percentile(latencias, 95)
    lat_p99 = np.percentile(latencias, 99)
    cumple_umbral = lat_p95 < LATENCY_THRESHOLD_MS

    # 6. Reporte Final
    print("\n" + "=" * 65)
    print("📊 RESULTADOS DEL BENCHMARK DE LATENCIA")
    print("=" * 65)
    print(f" Total de llamadas exitosas:  {exitosas}/{N_CALLS} ({exitosas/N_CALLS*100:.1f}%)")
    print(f" Latencia Mínima:             {lat_min:.2f} ms")
    print(f" Latencia Máxima:             {lat_max:.2f} ms")
    print(f" Latencia Promedio:           {lat_mean:.2f} ms")
    print(f" Latencia Mediana (P50):      {lat_median:.2f} ms")
    print(f" Latencia Percentil 95 (P95): {lat_p95:.2f} ms")
    print(f" Latencia Percentil 99 (P99): {lat_p99:.2f} ms")
    print("-" * 65)
    print(f" Umbral de SLA Requerido:     < {LATENCY_THRESHOLD_MS:.1f} ms")
    if cumple_umbral:
        print(f" ✅ ESTADO: CUMPLE EL SLA (P95 de {lat_p95:.2f} ms < {LATENCY_THRESHOLD_MS:.1f} ms)")
    else:
        print(f" ❌ ESTADO: NO CUMPLE EL SLA (P95 de {lat_p95:.2f} ms >= {LATENCY_THRESHOLD_MS:.1f} ms)")
    print("=" * 65)
    
    if ultima_respuesta:
        print("\nEjemplo de respuesta obtenida del endpoint /predict:")
        print(ultima_respuesta)

if __name__ == "__main__":
    main()
