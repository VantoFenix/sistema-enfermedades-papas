import React, { useState, useEffect, useRef } from 'react';

const API_BASE_URL = 'http://localhost:8000';
const CLASES = ['Blackspot Bruising', 'Dry Rot', 'Healthy Potatoes', 'Soft Rot'];

const CLASS_DESCRIPTIONS = {
  'Healthy Potatoes': {
    name: 'Papa Sana (Healthy)',
    badge: 'Conforme',
    isHealthy: true,
    color: '#10b981'
  },
  'Blackspot Bruising': {
    name: 'Mancha Negra (Blackspot Bruising)',
    badge: 'Defecto Físico',
    isHealthy: false,
    color: '#ef4444'
  },
  'Dry Rot': {
    name: 'Pudrición Seca (Dry Rot - Fusarium)',
    badge: 'Fúngico',
    isHealthy: false,
    color: '#f59e0b'
  },
  'Soft Rot': {
    name: 'Pudrición Blanda (Soft Rot - Pectobacterium)',
    badge: 'Bacteriano Crítico',
    isHealthy: false,
    color: '#dc2626'
  }
};

export default function App() {
  // 1. Estado de conectividad API
  const [apiStatus, setApiStatus] = useState({ online: false, loading: true, data: null });
  const [activeTab, setActiveTab] = useState('image'); // 'image' | 'camera'

  // 2. Estado de Modo Imagen
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [analyzingImage, setAnalyzingImage] = useState(false);

  // 3. Estado de Modo Cámara
  const [cameraActive, setCameraActive] = useState(false);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const roiRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);
  const isProcessingRef = useRef(false);

  // Buffer para Suavizado Temporal (Promedio Móvil de 4 frames)
  const historyRef = useRef([]);

  // 4. Estado de Resultados y Telemetría Compartidos
  const [result, setResult] = useState(null);
  const [latencyMs, setLatencyMs] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  const fileInputRef = useRef(null);

  // Verificación periódica de la API (cada 5 segundos)
  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/health`);
      if (res.ok) {
        const data = await res.json();
        setApiStatus({ online: true, loading: false, data });
      } else {
        setApiStatus({ online: false, loading: false, data: null });
      }
    } catch {
      setApiStatus({ online: false, loading: false, data: null });
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 5000);
    return () => clearInterval(interval);
  }, []);

  // Control de Cámara Web
  const startCamera = async () => {
    try {
      setErrorMsg(null);
      historyRef.current = []; // Reiniciar buffer de suavizado
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'environment' }
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      streamRef.current = stream;
      setCameraActive(true);
    } catch (err) {
      setErrorMsg(`Error al acceder a la cámara web: ${err.message}`);
      setCameraActive(false);
    }
  };

  const stopCamera = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
    isProcessingRef.current = false;
    historyRef.current = []; // Limpiar historial al detener
  };

  // Al cambiar de pestaña, apagar la cámara y limpiar buffers
  const handleTabChange = (tab) => {
    if (tab !== 'camera' && cameraActive) {
      stopCamera();
    }
    setActiveTab(tab);
    setErrorMsg(null);
    historyRef.current = [];
  };

  // Cálculo de coordenadas del recorte ROI visual hacia el espacio nativo del sensor de video
  const calculateRoiCoordinates = (video, roiElement) => {
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) return { sx: 0, sy: 0, sWidth: 224, sHeight: 224 };

    if (roiElement && video) {
      const videoRect = video.getBoundingClientRect();
      const roiRect = roiElement.getBoundingClientRect();

      if (videoRect.width > 0 && videoRect.height > 0) {
        // Escala y compensación de centrado por object-fit: cover
        const scale = Math.max(videoRect.width / vw, videoRect.height / vh);
        const offsetX = (videoRect.width - vw * scale) / 2;
        const offsetY = (videoRect.height - vh * scale) / 2;

        const sx = Math.max(0, Math.round((roiRect.left - videoRect.left - offsetX) / scale));
        const sy = Math.max(0, Math.round((roiRect.top - videoRect.top - offsetY) / scale));
        const sWidth = Math.min(vw - sx, Math.round(roiRect.width / scale));
        const sHeight = Math.min(vh - sy, Math.round(roiRect.height / scale));

        return { sx, sy, sWidth, sHeight };
      }
    }

    // Fallback simétrico centrado
    const size = Math.round(Math.min(vw, vh) * 0.65);
    return {
      sx: Math.round((vw - size) / 2),
      sy: Math.round((vh - size) / 2),
      sWidth: size,
      sHeight: size
    };
  };

  // Ciclo de Inferencia de Cámara (5 FPS = 200 ms) con Recorte ROI Estricto y Suavizado Temporal
  useEffect(() => {
    if (!cameraActive) return;

    const captureAndPredict = () => {
      if (isProcessingRef.current || !videoRef.current || !canvasRef.current) return;
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const roiElement = roiRef.current;

      // Asegurar que el video tenga fotogramas listos
      if (video.readyState < 2) return;

      const { sx, sy, sWidth, sHeight } = calculateRoiCoordinates(video, roiElement);

      // Redimensionar el canvas a exactamente 224x224 (entrada estándar ResNet18)
      canvas.width = 224;
      canvas.height = 224;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Recorte estricto: extrae únicamente el área visual del ROI
      ctx.drawImage(video, sx, sy, sWidth, sHeight, 0, 0, 224, 224);

      isProcessingRef.current = true;

      canvas.toBlob(
        async (blob) => {
          if (!blob) {
            isProcessingRef.current = false;
            return;
          }

          const tStart = performance.now();
          try {
            const formData = new FormData();
            formData.append('file', blob, 'roi_frame.jpg');

            const res = await fetch(`${API_BASE_URL}/predict`, {
              method: 'POST',
              body: formData,
            });

            const tEnd = performance.now();
            const elapsed = (tEnd - tStart).toFixed(1);

            if (res.ok) {
              const data = await res.json();

              // Suavizado temporal: buffer de los últimos 4 vectores de probabilidades
              historyRef.current.push(data.probabilidades);
              if (historyRef.current.length > 4) {
                historyRef.current.shift();
              }

              // Calcular promedio móvil por clase sobre los últimos 4 frames
              const n = historyRef.current.length;
              const smoothedProbs = {};
              for (const cls of CLASES) {
                const sum = historyRef.current.reduce((acc, curr) => acc + (curr[cls] || 0), 0);
                smoothedProbs[cls] = sum / n;
              }

              // Determinar la clase ganadora suavizada
              let bestClass = CLASES[0];
              let maxProb = 0;
              for (const [cls, prob] of Object.entries(smoothedProbs)) {
                if (prob > maxProb) {
                  maxProb = prob;
                  bestClass = cls;
                }
              }

              setResult({
                clase_predicha: bestClass,
                confianza: maxProb,
                probabilidades: smoothedProbs,
                modelo_version: data.modelo_version
              });
              setLatencyMs(elapsed);
            }
          } catch (err) {
            console.warn('Fallo en frame de streaming:', err);
          } finally {
            isProcessingRef.current = false;
          }
        },
        'image/jpeg',
        0.85
      );
    };

    intervalRef.current = setInterval(captureAndPredict, 200);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [cameraActive]);

  // Manejo de Archivo de Imagen
  const handleFileSelect = (file) => {
    if (!file || !file.type.startsWith('image/')) {
      setErrorMsg('Por favor selecciona un archivo de imagen válido (JPG, PNG).');
      return;
    }
    setErrorMsg(null);
    setSelectedFile(file);
    setResult(null);
    setLatencyMs(null);
    historyRef.current = [];

    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
  };

  const onDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const onDragLeave = () => {
    setIsDragOver(false);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleAnalyzeImage = async () => {
    if (!selectedFile) return;

    setAnalyzingImage(true);
    setErrorMsg(null);
    historyRef.current = [];
    const tStart = performance.now();

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const res = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      const tEnd = performance.now();
      const elapsed = (tEnd - tStart).toFixed(1);

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Error HTTP ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
      setLatencyMs(elapsed);
    } catch (err) {
      setErrorMsg(`Fallo en inferencia: ${err.message}`);
    } finally {
      setAnalyzingImage(false);
    }
  };

  // Lógica de decisión del Actuador PLC
  // Regla estricta: Confianza >= 0.75 para Healthy o Defect. Si < 0.75, Derivar a revisión humana (Sec. 9.2.c).
  const confianza = result?.confianza ?? 0;
  const tieneBajaConfianza = result && confianza < 0.75;
  const esPapaSana = result && !tieneBajaConfianza && result.clase_predicha === 'Healthy Potatoes';
  const esDefecto = result && !tieneBajaConfianza && result.clase_predicha !== 'Healthy Potatoes';

  return (
    <div className="dashboard-container">
      {/* 1. Encabezado y Barra de Estado en Vivo */}
      <header className="header-card">
        <div className="brand-wrapper">
          <div className="brand-icon">🥔</div>
          <div>
            <h1 className="brand-title">
              HOTPOTATO AI
              <span className="badge">INSPECCIÓN VISUAL EN LÍNEA</span>
            </h1>
            <p className="brand-subtitle">
              Control de Calidad Fitosanitario y Accionamiento de Eyector PLC · ResNet18
            </p>
          </div>
        </div>

        <div>
          {apiStatus.loading ? (
            <div className="status-pill offline">
              <span className="spinner" />
              <span>Verificando API...</span>
            </div>
          ) : apiStatus.online ? (
            <div className="status-pill online" title="Backend FastAPI en Docker">
              <span className="pulse-dot online" />
              <span>API Activa: {apiStatus.data?.model || 'HotPotato-ResNet18'} (v{apiStatus.data?.version || '1'})</span>
            </div>
          ) : (
            <div className="status-pill offline" title="Verifique que el contenedor Docker esté en ejecución">
              <span className="pulse-dot offline" />
              <span>API Desconectada (Inicia Docker)</span>
            </div>
          )}
        </div>
      </header>

      {/* Selector de Modo de Operación (Tabs) */}
      <nav className="tabs-nav">
        <button
          className={`tab-btn ${activeTab === 'image' ? 'active' : ''}`}
          onClick={() => handleTabChange('image')}
        >
          <span>📁</span>
          <span>Modo 1: Inspección por Imagen (Subir Archivo)</span>
        </button>

        <button
          className={`tab-btn ${activeTab === 'camera' ? 'active' : ''}`}
          onClick={() => handleTabChange('camera')}
        >
          <span>📹</span>
          <span>Modo 2: Cámara en Vivo — Banda Transportadora (5 FPS)</span>
        </button>
      </nav>

      {/* Canvas oculto para muestreo de frames de video */}
      <canvas ref={canvasRef} style={{ display: 'none' }} />

      {/* Rejilla Principal */}
      <div className="main-grid">
        {/* Columna Izquierda: Entrada Óptica (Imagen o Cámara según Tab) */}
        <section className="panel-card">
          {activeTab === 'image' ? (
            <>
              <div className="panel-header">
                <h2 className="panel-title">
                  📷 Carga de Fotograma Estático
                </h2>
                {selectedFile && (
                  <button
                    className="btn-secondary"
                    onClick={() => {
                      setSelectedFile(null);
                      setPreviewUrl(null);
                      setResult(null);
                      historyRef.current = [];
                    }}
                  >
                    Limpiar
                  </button>
                )}
              </div>

              <div
                className={`dropzone ${isDragOver ? 'dragover' : ''}`}
                onDragOver={onDragOver}
                onDragLeave={onDragLeave}
                onDrop={onDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
                  accept="image/jpeg,image/png,image/jpg"
                  style={{ display: 'none' }}
                />

                {previewUrl ? (
                  <div className="preview-container">
                    <img src={previewUrl} alt="Vista previa del tubérculo" className="preview-image" />
                    <span className="file-info-badge font-mono">
                      {selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)
                    </span>
                  </div>
                ) : (
                  <>
                    <div className="dropzone-icon">📥</div>
                    <h3 className="dropzone-title">Arrastra una imagen de tubérculo aquí</h3>
                    <p className="dropzone-subtitle">o haz clic para explorar tus archivos (JPG, PNG)</p>
                  </>
                )}
              </div>

              {errorMsg && (
                <div style={{ marginTop: '14px', color: '#f87171', fontSize: '13px', background: 'rgba(239,68,68,0.1)', padding: '10px 14px', borderRadius: '8px', border: '1px solid rgba(239,68,68,0.3)' }}>
                  ⚠️ {errorMsg}
                </div>
              )}

              <button
                className="btn-primary"
                onClick={handleAnalyzeImage}
                disabled={!selectedFile || analyzingImage || !apiStatus.online}
              >
                {analyzingImage ? (
                  <>
                    <span className="spinner" />
                    <span>Ejecutando Inferencia Neuronal...</span>
                  </>
                ) : (
                  <>
                    <span>⚡</span>
                    <span>Analizar Tubérculo</span>
                  </>
                )}
              </button>
            </>
          ) : (
            <>
              <div className="panel-header">
                <h2 className="panel-title">
                  🎥 Transmisión en Tiempo Real (Banda Transportadora)
                </h2>
                <span className="fps-badge font-mono">
                  {cameraActive ? 'MUESTREO: 5 FPS (200ms)' : 'STANDBY'}
                </span>
              </div>

              {/* Vista de Cámara con Guía ROI Superpuesta */}
              <div className="camera-container">
                <video
                  ref={videoRef}
                  playsInline
                  muted
                  autoPlay
                  className="camera-video"
                  style={{ display: cameraActive ? 'block' : 'none' }}
                />

                {!cameraActive && (
                  <div className="camera-placeholder">
                    <div style={{ fontSize: '48px', marginBottom: '10px' }}>📹</div>
                    <h3 style={{ fontSize: '15px', color: '#fff', marginBottom: '6px' }}>
                      Cámara en Línea Desactivada
                    </h3>
                    <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                      Inicia la cámara para simular la captura en movimiento sobre la banda clasificadora.
                    </p>
                  </div>
                )}

                {/* Recuadro Guía Central (ROI / Zona de Inspección) */}
                {cameraActive && (
                  <>
                    <div className="roi-overlay" ref={roiRef}>
                      <div className="roi-corner roi-tl" />
                      <div className="roi-corner roi-tr" />
                      <div className="roi-corner roi-bl" />
                      <div className="roi-corner roi-br" />
                      <div className="roi-crosshair" />
                      <span className="roi-label">ZONA DE INSPECCIÓN (ROI)</span>
                    </div>

                    {/* Etiqueta Dinámica Superpuesta */}
                    <div className="live-telemetry-tag">
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: esPapaSana ? '#10b981' : esDefecto ? '#ef4444' : '#f59e0b' }} />
                        <strong style={{ color: '#fff' }}>
                          {result ? result.clase_predicha : 'Encuadrando...'}
                        </strong>
                        {result && (
                          <span style={{ color: 'var(--text-secondary)' }} className="font-mono">
                            ({(result.confianza * 100).toFixed(1)}%)
                          </span>
                        )}
                      </div>

                      <div className="font-mono" style={{ color: '#60a5fa' }}>
                        {latencyMs ? `${latencyMs} ms (Suavizado)` : '-- ms'}
                      </div>
                    </div>
                  </>
                )}
              </div>

              {/* Texto Guía Fitosanitario */}
              <p className="roi-instruction-text">
                💡 Coloque el tubérculo sobre fondo claro llenando el recuadro ROI
              </p>

              {errorMsg && (
                <div style={{ marginTop: '14px', color: '#f87171', fontSize: '13px', background: 'rgba(239,68,68,0.1)', padding: '10px 14px', borderRadius: '8px', border: '1px solid rgba(239,68,68,0.3)' }}>
                  ⚠️ {errorMsg}
                </div>
              )}

              {/* Botón de Control de Cámara */}
              {!cameraActive ? (
                <button
                  className="btn-primary"
                  onClick={startCamera}
                  disabled={!apiStatus.online}
                >
                  <span>▶</span>
                  <span>Encender Cámara (5 FPS)</span>
                </button>
              ) : (
                <button
                  className="btn-primary btn-danger"
                  onClick={stopCamera}
                >
                  <span>⏹</span>
                  <span>Detener Cámara</span>
                </button>
              )}
            </>
          )}
        </section>

        {/* Columna Derecha: Panel de Diagnóstico & Actuador PLC (Compartido) */}
        <section className="panel-card">
          <div className="panel-header">
            <h2 className="panel-title">
              ⚙️ Actuador PLC & Diagnóstico Fitosanitario
            </h2>
            <span className="brand-subtitle font-mono">
              Umbral Conforme: ≥ 75%
            </span>
          </div>

          {/* Tarjeta de Acción Mecánica según Rúbrica y Sec. 9.2.c */}
          {result ? (
            tieneBajaConfianza ? (
              <div className="decision-card warning">
                <div className="decision-icon">⚠️</div>
                <div>
                  <h3 className="decision-title">
                    ⚠️ BAJA CONFIANZA (&lt;75%) — DERIVAR A REVISIÓN HUMANA
                  </h3>
                  <p className="decision-subtitle">
                    Mitigación Sec. 9.2.c: Incertidumbre visual detectada ({(result.confianza * 100).toFixed(1)}%). Desvío automático a inspección manual secundaria.
                  </p>
                </div>
              </div>
            ) : esPapaSana ? (
              <div className="decision-card healthy">
                <div className="decision-icon">✅</div>
                <div>
                  <h3 className="decision-title">
                    ✅ TUBÉRCULO SANO — CONTINÚA A CORTE
                  </h3>
                  <p className="decision-subtitle">
                    Tubérculo conforme. Procede al lote de procesamiento industrial sin riesgo fitosanitario.
                  </p>
                </div>
              </div>
            ) : (
              <div className="decision-card defect">
                <div className="decision-icon">🚨</div>
                <div>
                  <h3 className="decision-title">
                    🚨 DEFECTO DETECTADO — ACCIONAR EYECTOR MECÁNICO
                  </h3>
                  <p className="decision-subtitle">
                    Patología identificada: {result.clase_predicha}. Pulso PLC emitido hacia el eyector neumático.
                  </p>
                </div>
              </div>
            )
          ) : (
            <div className="decision-card idle">
              <div className="decision-icon">⏸️</div>
              <div>
                <h3 className="decision-title">EN ESPERA DE CAPTURA</h3>
                <p className="decision-subtitle">
                  Coloque una muestra en el área óptica o active la cámara en vivo para recibir el veredicto en tiempo real.
                </p>
              </div>
            </div>
          )}

          {/* Telemetría */}
          <div className="telemetry-grid">
            <div className="telemetry-card">
              <div className="telemetry-label">Confianza</div>
              <div
                className="telemetry-value font-mono"
                style={{
                  color: result
                    ? tieneBajaConfianza
                      ? '#fbbf24'
                      : esPapaSana
                        ? '#34d399'
                        : '#f87171'
                    : '#fff',
                }}
              >
                {result ? `${(result.confianza * 100).toFixed(2)}%` : '--'}
              </div>
            </div>

            <div className="telemetry-card">
              <div className="telemetry-label">Latencia E2E</div>
              <div className="telemetry-value font-mono" style={{ color: '#60a5fa' }}>
                {latencyMs ? `${latencyMs} ms` : '--'}
              </div>
            </div>

            <div className="telemetry-card">
              <div className="telemetry-label">Modelo Activo</div>
              <div className="telemetry-value font-mono" style={{ fontSize: '13px', paddingTop: '4px' }}>
                {result?.modelo_version || 'ResNet18:v1'}
              </div>
            </div>
          </div>

          {/* Desglose de Probabilidades por Clase */}
          <div>
            <h3 className="prob-section-title">
              Distribución de Probabilidades por Clase {activeTab === 'camera' && cameraActive && '(Promedio 4 Frames)'}
            </h3>

            {result ? (
              Object.entries(result.probabilidades || {}).map(([clase, prob]) => {
                const isSelected = result.clase_predicha === clase;
                const isClassHealthy = clase === 'Healthy Potatoes';
                const percentage = (prob * 100).toFixed(2);

                return (
                  <div key={clase} className="prob-row">
                    <div className="prob-meta">
                      <span className="prob-label">
                        <span
                          style={{
                            width: '8px',
                            height: '8px',
                            borderRadius: '50%',
                            backgroundColor: isClassHealthy ? '#10b981' : '#ef4444',
                          }}
                        />
                        {CLASS_DESCRIPTIONS[clase]?.name || clase}
                        {isSelected && (
                          <span
                            style={{
                              fontSize: '10px',
                              background: 'rgba(255,255,255,0.1)',
                              padding: '1px 6px',
                              borderRadius: '4px',
                              color: tieneBajaConfianza ? '#fbbf24' : '#60a5fa',
                            }}
                          >
                            Predicho
                          </span>
                        )}
                      </span>
                      <span className="prob-percentage">{percentage}%</span>
                    </div>

                    <div className="prob-bar-track">
                      <div
                        className={`prob-bar-fill ${isSelected
                            ? tieneBajaConfianza
                              ? 'warning'
                              : isClassHealthy
                                ? 'healthy'
                                : 'defect'
                            : 'other'
                          }`}
                        style={{ width: `${percentage}%` }}
                      />
                    </div>
                  </div>
                );
              })
            ) : (
              <p style={{ color: 'var(--text-muted)', fontSize: '13px', fontStyle: 'italic', padding: '10px 0' }}>
                Las probabilidades aparecerán aquí tras la inferencia...
              </p>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
