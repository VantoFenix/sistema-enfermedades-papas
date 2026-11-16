import React, { useState, useEffect, useRef } from 'react';

const API_BASE_URL = 'http://localhost:8000';

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
  const [apiStatus, setApiStatus] = useState({ online: false, loading: true, data: null });
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [latencyMs, setLatencyMs] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  const fileInputRef = useRef(null);

  // 1. Verificación de salud de la API
  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/health`);
      if (res.ok) {
        const data = await res.json();
        setApiStatus({ online: true, loading: false, data });
      } else {
        setApiStatus({ online: false, loading: false, data: null });
      }
    } catch (err) {
      setApiStatus({ online: false, loading: false, data: null });
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  // 2. Manejo de selección de archivo y Drag & Drop
  const handleFileSelect = (file) => {
    if (!file || !file.type.startsWith('image/')) {
      setErrorMsg('Por favor selecciona un archivo de imagen válido (JPG, PNG).');
      return;
    }
    setErrorMsg(null);
    setSelectedFile(file);
    setResult(null);
    setLatencyMs(null);

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

  // 3. Envío de imagen e inferencia con medición de latencia
  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setAnalyzing(true);
    setErrorMsg(null);
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
      setAnalyzing(false);
    }
  };

  const isHealthy = result?.clase_predicha === 'Healthy Potatoes';

  return (
    <div className="dashboard-container">
      {/* Encabezado Industrial */}
      <header className="header-card">
        <div className="brand-wrapper">
          <div className="brand-icon">🥔</div>
          <div>
            <h1 className="brand-title">
              HOTPOTATO AI
              <span className="badge">INSPECCIÓN EN LÍNEA</span>
            </h1>
            <p className="brand-subtitle">
              Sistema de Clasificación Fitosanitaria y Control Mecánico de Eyección · ResNet18
            </p>
          </div>
        </div>

        {/* Indicador de Estado en Vivo */}
        <div>
          {apiStatus.loading ? (
            <div className="status-pill offline">
              <span className="spinner" />
              <span>Conectando API...</span>
            </div>
          ) : apiStatus.online ? (
            <div className="status-pill online" title="Backend FastAPI operativo en Docker">
              <span className="pulse-dot online" />
              <span>API Conectada ({apiStatus.data?.model || 'HotPotato-ResNet18'} v{apiStatus.data?.version || '1'})</span>
            </div>
          ) : (
            <div className="status-pill offline" title="Verifique que Docker esté ejecutando el contenedor">
              <span className="pulse-dot offline" />
              <span>API Desconectada (http://localhost:8000)</span>
            </div>
          )}
        </div>
      </header>

      {/* Rejilla Principal */}
      <div className="main-grid">
        {/* Columna Izquierda: Adquisición y Carga de Tubérculo */}
        <section className="panel-card">
          <div className="panel-header">
            <h2 className="panel-title">
              📷 Cámara de Inspección (Estación Óptica)
            </h2>
            {selectedFile && (
              <button 
                className="btn-secondary"
                onClick={() => {
                  setSelectedFile(null);
                  setPreviewUrl(null);
                  setResult(null);
                }}
              >
                Limpiar Muestra
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
                <img
                  src={previewUrl}
                  alt="Tubérculo cargado"
                  className="preview-image"
                />
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
            onClick={handleAnalyze}
            disabled={!selectedFile || analyzing || !apiStatus.online}
          >
            {analyzing ? (
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
        </section>

        {/* Columna Derecha: Panel de Diagnóstico & Acción Mecánica */}
        <section className="panel-card">
          <div className="panel-header">
            <h2 className="panel-title">
              ⚙️ Actuador y Diagnóstico en Tiempo Real
            </h2>
            <span className="brand-subtitle font-mono">
              SLA: &lt; 150 ms
            </span>
          </div>

          {/* Tarjeta de Acción Mecánica en Banda */}
          {result ? (
            <div className={`decision-card ${isHealthy ? 'healthy' : 'defect'}`}>
              <div className="decision-icon">
                {isHealthy ? '✓' : '⚠️'}
              </div>
              <div>
                <h3 className="decision-title">
                  {isHealthy
                    ? 'APROBADA — PAPA SANA (Continúa a corte)'
                    : 'DEFECTO DETECTADO — ACCIONAR EYECTOR MECÁNICO'}
                </h3>
                <p className="decision-subtitle">
                  {isHealthy
                    ? 'Tubérculo conforme. Procede al lote de procesamiento industrial sin riesgo fitosanitario.'
                    : `Patología identificada: ${result.clase_predicha}. Orden de desvío automático emitida a compuerta.`}
                </p>
              </div>
            </div>
          ) : (
            <div className="decision-card idle">
              <div className="decision-icon">⏸</div>
              <div>
                <h3 className="decision-title">EN ESPERA DE CAPTURA</h3>
                <p className="decision-subtitle">
                  Cargue una muestra de tubérculo para activar el clasificador neuronal en tiempo real.
                </p>
              </div>
            </div>
          )}

          {/* Métricas de Telemetría */}
          <div className="telemetry-grid">
            <div className="telemetry-card">
              <div className="telemetry-label">Confianza</div>
              <div className="telemetry-value font-mono" style={{ color: result ? (isHealthy ? '#34d399' : '#f87171') : '#fff' }}>
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

          {/* Distribución de Probabilidades */}
          <div>
            <h3 className="prob-section-title">
              Distribución de Probabilidades por Clase
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
                            backgroundColor: isClassHealthy ? '#10b981' : '#ef4444' 
                          }} 
                        />
                        {CLASS_DESCRIPTIONS[clase]?.name || clase}
                        {isSelected && (
                          <span style={{ fontSize: '10px', background: 'rgba(255,255,255,0.1)', padding: '1px 6px', borderRadius: '4px', color: '#60a5fa' }}>
                            Predicho
                          </span>
                        )}
                      </span>
                      <span className="prob-percentage">
                        {percentage}%
                      </span>
                    </div>

                    <div className="prob-bar-track">
                      <div
                        className={`prob-bar-fill ${isSelected ? (isClassHealthy ? 'healthy' : 'defect') : 'other'}`}
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
