import React, { useState, useEffect, useRef } from 'react';
import {
  ShieldCheck,
  ShieldAlert,
  AlertTriangle,
  Upload,
  Camera,
  Layers,
  Flame,
  FileText,
  Clock,
  Sparkles,
  Calendar,
  CheckCircle2,
  XCircle,
  RefreshCw,
  Eye,
  Sliders,
  ChevronRight,
  Database,
  History,
  Download,
  Crosshair,
  Filter,
  Maximize2
} from 'lucide-react';

const API_BASE = '/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('upload'); // 'upload' | 'camera' | 'samples'
  const [viewMode, setViewMode] = useState('annotated'); // 'annotated' | 'heatmap' | 'raw'
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.85);

  // Filter toggles for visual overlays
  const [showDates, setShowDates] = useState(true);
  const [showTamper, setShowTamper] = useState(true);
  const [showTokens, setShowTokens] = useState(false);

  // Custom reference date picker for real-time testing
  const [customDate, setCustomDate] = useState('2026-08-14');

  const [samples, setSamples] = useState([]);
  const [selectedSample, setSelectedSample] = useState(null);
  const [loading, setLoading] = useState(false);
  const [verdictData, setVerdictData] = useState(null);
  const [history, setHistory] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [hoveredBox, setHoveredBox] = useState(null);

  // Camera state
  const videoRef = useRef(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState(null);

  // Drag and drop state
  const [dragActive, setDragActive] = useState(false);

  useEffect(() => {
    fetchSamples();
    fetchHistory();
    fetchMetrics();
  }, []);

  const fetchSamples = async () => {
    try {
      const res = await fetch(`${API_BASE}/samples?limit=30`);
      if (res.ok) {
        const data = await res.json();
        setSamples(data);
        if (data.length > 0 && !verdictData) {
          handleSelectSample(data[0]);
        }
      }
    } catch (e) {
      console.error('Failed fetching sample catalog:', e);
    }
  };

  const fetchHistory = async () => {
    try {
      const res = await fetch(`${API_BASE}/history`);
      if (res.ok) {
        const data = await res.json();
        setHistory(data);
      }
    } catch (e) {
      console.error('Failed fetching history:', e);
    }
  };

  const fetchMetrics = async () => {
    try {
      const res = await fetch(`${API_BASE}/metrics`);
      if (res.ok) {
        const data = await res.json();
        setMetrics(data);
      }
    } catch (e) {
      console.error('Failed fetching metrics:', e);
    }
  };

  const handleFileUpload = async (file) => {
    if (!file) return;
    setLoading(true);
    setSelectedSample(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('product_name', file.name);

    try {
      const res = await fetch(`${API_BASE}/verify`, {
        method: 'POST',
        body: formData
      });
      if (res.ok) {
        const data = await res.json();
        setVerdictData(data);
        fetchHistory();
        fetchMetrics();
      } else {
        alert('Inspection failed. Please try another packaging image.');
      }
    } catch (e) {
      console.error('Inspection error:', e);
      alert('Error connecting to backend verification service.');
    } finally {
      setLoading(false);
    }
  };

  const handleSelectSample = async (sample) => {
    setSelectedSample(sample);
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/samples/verify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sample_id: sample.id })
      });
      if (res.ok) {
        const data = await res.json();
        setVerdictData(data);
        fetchHistory();
        fetchMetrics();
      }
    } catch (e) {
      console.error('Sample verification error:', e);
    } finally {
      setLoading(false);
    }
  };

  // Camera Management
  const startCamera = async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        setCameraActive(true);
      }
    } catch (err) {
      setCameraError('Camera access denied or unavailable on this device.');
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
  };

  const captureFrame = () => {
    if (!videoRef.current) return;
    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth || 640;
    canvas.height = videoRef.current.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.9);

    stopCamera();
    verifyBase64(dataUrl, 'webcam_scan.jpg');
  };

  const verifyBase64 = async (b64String, filename) => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/verify-base64`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_base64: b64String, filename })
      });
      if (res.ok) {
        const data = await res.json();
        setVerdictData(data);
        fetchHistory();
        fetchMetrics();
      }
    } catch (e) {
      console.error('Base64 verification error:', e);
    } finally {
      setLoading(false);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') setDragActive(true);
    else if (e.type === 'dragleave') setDragActive(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const exportReport = () => {
    if (!verdictData) return;
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(verdictData, null, 2));
    const dlAnchorElem = document.createElement('a');
    dlAnchorElem.setAttribute('href', dataStr);
    dlAnchorElem.setAttribute('download', `VeriSight_Report_${verdictData.id}.json`);
    dlAnchorElem.click();
  };

  return (
    <div className="app-container">
      {/* Navbar */}
      <header className="navbar">
        <div className="brand-wrapper">
          <div className="brand-logo">
            <ShieldCheck size={22} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span className="brand-title">VeriSight</span>
              <span className="brand-badge">Forensic AI</span>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Real-time Packaging Integrity & Expiry Verification Engine
            </div>
          </div>
        </div>

        {/* Global Operational Stats */}
        <div className="nav-stats">
          <div className="stat-chip">
            <span>Pass Rate:</span>
            <strong style={{ color: 'var(--color-pass)' }}>
              {metrics ? `${metrics.pass_rate_percentage}%` : '100%'}
            </strong>
          </div>
          <div className="stat-chip">
            <span>Inspected:</span>
            <strong>{metrics ? metrics.total_inspected : 0}</strong>
          </div>
          <div className="stat-chip">
            <span>Avg Speed:</span>
            <strong>{metrics ? `${metrics.avg_processing_time_sec}s` : '0.35s'}</strong>
          </div>
        </div>
      </header>

      {/* Main Grid */}
      <main className="dashboard-grid">
        {/* Left Column: Input Source & Catalog */}
        <section className="panel-card">
          <div className="panel-header">
            <div className="panel-title">
              <Layers size={16} color="var(--color-info)" />
              Packaging Source
            </div>
            {loading && <div className="loading-spinner" />}
          </div>

          <div className="panel-body">
            {/* Input Selection Tabs */}
            <div className="tab-group">
              <button
                className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
                onClick={() => { setActiveTab('upload'); stopCamera(); }}
              >
                <Upload size={14} /> Upload
              </button>
              <button
                className={`tab-btn ${activeTab === 'camera' ? 'active' : ''}`}
                onClick={() => { setActiveTab('camera'); startCamera(); }}
              >
                <Camera size={14} /> Camera
              </button>
              <button
                className={`tab-btn ${activeTab === 'samples' ? 'active' : ''}`}
                onClick={() => { setActiveTab('samples'); stopCamera(); }}
              >
                <Database size={14} /> Benchmark
              </button>
            </div>

            {/* Mode 1: Upload */}
            {activeTab === 'upload' && (
              <div
                className={`dropzone ${dragActive ? 'drag-active' : ''}`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={() => document.getElementById('file-input').click()}
              >
                <input
                  id="file-input"
                  type="file"
                  accept="image/*"
                  style={{ display: 'none' }}
                  onChange={(e) => e.target.files[0] && handleFileUpload(e.target.files[0])}
                />
                <div className="dropzone-icon">
                  <Upload size={22} />
                </div>
                <div className="dropzone-text">Drop product image here</div>
                <div className="dropzone-subtext">JPG, PNG, WEBP high-resolution packaging photo</div>
                <button className="btn btn-secondary" style={{ marginTop: '0.4rem', fontSize: '0.75rem', padding: '0.35rem 0.75rem' }}>
                  Browse Files
                </button>
              </div>
            )}

            {/* Mode 2: Camera */}
            {activeTab === 'camera' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                <div style={{
                  position: 'relative',
                  width: '100%',
                  height: '210px',
                  backgroundColor: '#000',
                  borderRadius: 'var(--radius-md)',
                  overflow: 'hidden',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                  />
                  {cameraError && (
                    <div style={{ color: 'var(--color-danger)', fontSize: '0.8rem', padding: '1rem', textAlign: 'center' }}>
                      {cameraError}
                    </div>
                  )}
                </div>
                <button className="btn btn-primary btn-full" onClick={captureFrame}>
                  <Camera size={15} /> Capture & Inspect Packaging
                </button>
              </div>
            )}

            {/* Mode 3: Dataset Catalog */}
            {activeTab === 'samples' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  Select benchmark image from project datasets:
                </div>
                <div className="sample-list">
                  {samples.map((sample) => (
                    <div
                      key={sample.id}
                      className={`sample-item ${selectedSample?.id === sample.id ? 'selected' : ''}`}
                      onClick={() => handleSelectSample(sample)}
                    >
                      <div className="sample-info">
                        <div className="sample-name">{sample.name}</div>
                        <div className="sample-meta">
                          <span>{sample.dataset}</span>
                          <span>•</span>
                          <span>{sample.category}</span>
                        </div>
                      </div>
                      <div>
                        {sample.ground_truth.is_tampered ? (
                          <span className="tag-badge tampered">Tampered</span>
                        ) : (
                          <span className="tag-badge authentic">Authentic</span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Active Target Card */}
            {verdictData && (
              <div className="telemetry-block">
                <div className="telemetry-header">
                  <span>Target Target</span>
                  <span className="mono" style={{ fontSize: '0.7rem', color: 'var(--color-info)' }}>{verdictData.id}</span>
                </div>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-primary)', fontWeight: 600 }}>
                  {verdictData.product_name}
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                  <span>Latency: <strong className="mono" style={{ color: 'var(--text-primary)' }}>{verdictData.processing_time_seconds}s</strong></span>
                  <span>Tokens: <strong className="mono" style={{ color: 'var(--text-primary)' }}>{verdictData.ocr_analysis.total_tokens}</strong></span>
                </div>
                <button className="btn btn-secondary btn-full" onClick={exportReport} style={{ fontSize: '0.75rem', padding: '0.4rem' }}>
                  <Download size={13} /> Export JSON Audit
                </button>
              </div>
            )}
          </div>
        </section>

        {/* Center Column: Interactive Multi-Layer Viewport */}
        <section className="viewport-card">
          <div className="viewport-toolbar">
            <div className="viewport-tabs">
              <button
                className={`viewport-tab ${viewMode === 'annotated' ? 'active' : ''}`}
                onClick={() => setViewMode('annotated')}
              >
                <Eye size={14} /> Visual Inspection
              </button>
              <button
                className={`viewport-tab ${viewMode === 'heatmap' ? 'active' : ''}`}
                onClick={() => setViewMode('heatmap')}
              >
                <Flame size={14} /> Forensic Heatmap
              </button>
              <button
                className={`viewport-tab ${viewMode === 'raw' ? 'active' : ''}`}
                onClick={() => setViewMode('raw')}
              >
                <Layers size={14} /> Original Frame
              </button>
            </div>

            {viewMode === 'heatmap' && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.75rem' }}>
                <Sliders size={13} color="var(--text-muted)" />
                <span style={{ color: 'var(--text-secondary)' }}>Intensity:</span>
                <input
                  type="range"
                  min="0.2"
                  max="1.0"
                  step="0.05"
                  value={heatmapOpacity}
                  onChange={(e) => setHeatmapOpacity(parseFloat(e.target.value))}
                  style={{ width: '80px' }}
                />
              </div>
            )}
          </div>

          <div className="viewport-stage">
            {loading ? (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
                <div className="loading-spinner" style={{ width: '36px', height: '36px' }} />
                <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  Processing Region Proposals, OCR & Forensic Tamper Engine...
                </div>
              </div>
            ) : verdictData ? (
              <div style={{ position: 'relative', display: 'inline-block' }}>
                {viewMode === 'annotated' && (
                  <img
                    src={verdictData.images.annotated || verdictData.images.raw}
                    alt="Annotated Inspection"
                    className="viewport-image"
                  />
                )}

                {viewMode === 'heatmap' && (
                  <div style={{ position: 'relative' }}>
                    <img
                      src={verdictData.images.raw}
                      alt="Raw"
                      className="viewport-image"
                    />
                    <img
                      src={verdictData.images.heatmap}
                      alt="Forensic Heatmap"
                      className="viewport-image"
                      style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        opacity: heatmapOpacity,
                        mixBlendMode: 'screen',
                        pointerEvents: 'none'
                      }}
                    />
                  </div>
                )}

                {viewMode === 'raw' && (
                  <img
                    src={verdictData.images.raw}
                    alt="Raw Packaging"
                    className="viewport-image"
                  />
                )}
              </div>
            ) : (
              <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                Select a sample or upload a photo to start verification.
              </div>
            )}
          </div>
        </section>

        {/* Right Column: Telemetry & Expiry Analysis */}
        <section className="panel-card">
          <div className="panel-header">
            <div className="panel-title">
              <Sparkles size={16} color="var(--color-brand)" />
              Inspection Verdict & Telemetry
            </div>
            {verdictData && (
              <span className="mono" style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                {verdictData.timestamp}
              </span>
            )}
          </div>

          <div className="panel-body">
            {verdictData ? (
              <>
                {/* Master Verdict Banner */}
                <div className={`verdict-banner ${verdictData.verdict.color}`}>
                  <div className="verdict-header">
                    <div className="verdict-badge">
                      {verdictData.verdict.status === 'PASS' && <CheckCircle2 size={22} />}
                      {verdictData.verdict.status === 'WARNING' && <AlertTriangle size={22} />}
                      {verdictData.verdict.status === 'REJECT' && <XCircle size={22} />}
                      VERDICT: {verdictData.verdict.status}
                    </div>
                    <span className="mono" style={{ fontWeight: 800, fontSize: '1rem' }}>
                      {verdictData.verdict.overall_score}%
                    </span>
                  </div>

                  <div className="verdict-action">
                    {verdictData.verdict.action}
                  </div>

                  <ul className="verdict-reasons">
                    {verdictData.verdict.reasons.map((r, i) => (
                      <li key={i}>
                        <ChevronRight size={12} style={{ flexShrink: 0, marginTop: '2px' }} />
                        <span>{r}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Expiry & Shelf Life Card */}
                <div className="telemetry-block">
                  <div className="telemetry-header">
                    <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                      <Calendar size={14} color="var(--color-pass)" /> Expiry & Shelf Life
                    </span>
                    <span
                      className="mono"
                      style={{
                        fontSize: '0.7rem',
                        fontWeight: 700,
                        color: verdictData.expiry_analysis.expiry_status === 'VALID' ? 'var(--color-pass)' :
                               verdictData.expiry_analysis.expiry_status === 'EXPIRING_SOON' ? 'var(--color-warn)' :
                               verdictData.expiry_analysis.expiry_status === 'EXPIRED' ? 'var(--color-danger)' : 'var(--text-muted)'
                      }}
                    >
                      {verdictData.expiry_analysis.expiry_status}
                    </span>
                  </div>

                  <div className="metric-grid">
                    <div className="metric-item">
                      <span className="metric-label">Expiration Date</span>
                      <span className="metric-value" style={{ color: 'var(--color-info)' }}>
                        {verdictData.expiry_analysis.expiry_date || 'Not Found'}
                      </span>
                    </div>

                    <div className="metric-item">
                      <span className="metric-label">Days Remaining</span>
                      <span className="metric-value">
                        {verdictData.expiry_analysis.days_remaining !== null
                          ? `${verdictData.expiry_analysis.days_remaining} d`
                          : 'N/A'}
                      </span>
                    </div>

                    <div className="metric-item">
                      <span className="metric-label">Manufacturing Date</span>
                      <span className="metric-value">
                        {verdictData.expiry_analysis.mfg_date || 'N/A'}
                      </span>
                    </div>

                    <div className="metric-item">
                      <span className="metric-label">Batch / Lot Code</span>
                      <span className="metric-value">
                        {verdictData.expiry_analysis.batch_number || 'N/A'}
                      </span>
                    </div>
                  </div>

                  {verdictData.expiry_analysis.shelf_life_percentage !== null && (
                    <div className="progress-container">
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                        <span>Shelf Life Remaining</span>
                        <span className="mono">{verdictData.expiry_analysis.shelf_life_percentage}%</span>
                      </div>
                      <div className="progress-track">
                        <div
                          className={`progress-fill ${
                            verdictData.expiry_analysis.shelf_life_percentage > 50 ? 'pass' :
                            verdictData.expiry_analysis.shelf_life_percentage > 20 ? 'warn' : 'danger'
                          }`}
                          style={{ width: `${verdictData.expiry_analysis.shelf_life_percentage}%` }}
                        />
                      </div>
                    </div>
                  )}
                </div>

                {/* Packaging Tampering Forensic Card */}
                <div className="telemetry-block">
                  <div className="telemetry-header">
                    <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                      <ShieldAlert size={14} color="var(--color-danger)" /> Forensic Tamper Scan
                    </span>
                    <span
                      className="mono"
                      style={{
                        fontSize: '0.7rem',
                        fontWeight: 700,
                        color: verdictData.tamper_analysis.status === 'AUTHENTIC' ? 'var(--color-pass)' :
                               verdictData.tamper_analysis.status === 'SUSPICIOUS' ? 'var(--color-warn)' : 'var(--color-danger)'
                      }}
                    >
                      {verdictData.tamper_analysis.status}
                    </span>
                  </div>

                  <div className="metric-grid">
                    <div className="metric-item">
                      <span className="metric-label">Authenticity Score</span>
                      <span className="metric-value" style={{ color: 'var(--color-pass)' }}>
                        {verdictData.tamper_analysis.authenticity_score}%
                      </span>
                    </div>

                    <div className="metric-item">
                      <span className="metric-label">Tamper Risk Score</span>
                      <span className="metric-value" style={{ color: 'var(--color-danger)' }}>
                        {int(verdictData.tamper_analysis.tamper_score * 100)}%
                      </span>
                    </div>

                    <div className="metric-item">
                      <span className="metric-label">ELA Compression Error</span>
                      <span className="metric-value mono">
                        {verdictData.tamper_analysis.forensic_breakdown.ela_compression_error}
                      </span>
                    </div>

                    <div className="metric-item">
                      <span className="metric-label">Sensor Noise Variance</span>
                      <span className="metric-value mono">
                        {verdictData.tamper_analysis.forensic_breakdown.noise_discrepancy}
                      </span>
                    </div>
                  </div>

                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                    {verdictData.tamper_analysis.verdict_description}
                  </div>
                </div>

                {/* OCR Text Telemetry */}
                <div className="telemetry-block">
                  <div className="telemetry-header">
                    <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                      <FileText size={14} color="var(--color-info)" /> Packaging Text Tokens
                    </span>
                    <span className="mono" style={{ fontSize: '0.7rem' }}>
                      {verdictData.ocr_analysis.total_tokens} tokens
                    </span>
                  </div>

                  <div style={{
                    maxHeight: '120px',
                    overflowY: 'auto',
                    backgroundColor: 'var(--bg-app)',
                    padding: '0.6rem',
                    borderRadius: 'var(--radius-sm)',
                    fontSize: '0.725rem',
                    color: 'var(--text-secondary)',
                    fontFamily: 'var(--font-mono)',
                    whiteSpace: 'pre-wrap'
                  }}>
                    {verdictData.ocr_analysis.text_summary || 'No text extracted.'}
                  </div>
                </div>
              </>
            ) : (
              <div style={{ textAlign: 'center', padding: '3rem 1rem', color: 'var(--text-muted)' }}>
                No active scan telemetry.
              </div>
            )}
          </div>
        </section>
      </main>

      {/* Bottom Panel: Audit Trail & Inspection History */}
      <section className="audit-history-panel">
        <div className="panel-header">
          <div className="panel-title">
            <History size={16} color="var(--color-info)" />
            Verification Audit Log & History
          </div>
          <button className="btn btn-secondary" onClick={fetchHistory} style={{ fontSize: '0.75rem', padding: '0.3rem 0.6rem' }}>
            <RefreshCw size={12} /> Refresh Log
          </button>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table className="history-table">
            <thead>
              <tr>
                <th>Inspection ID</th>
                <th>Product Target</th>
                <th>Timestamp</th>
                <th>Expiry Status</th>
                <th>Expiration Date</th>
                <th>Packaging Integrity</th>
                <th>Authenticity</th>
                <th>Overall Verdict</th>
              </tr>
            </thead>
            <tbody>
              {history.length > 0 ? (
                history.map((h) => (
                  <tr key={h.id}>
                    <td className="mono" style={{ color: 'var(--color-info)', fontWeight: 600 }}>{h.id}</td>
                    <td style={{ fontWeight: 600 }}>{h.product_name}</td>
                    <td className="mono" style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>{h.timestamp}</td>
                    <td>
                      <span
                        className="tag-badge"
                        style={{
                          background: h.expiry_status === 'VALID' ? 'var(--color-pass-bg)' :
                                      h.expiry_status === 'EXPIRING_SOON' ? 'var(--color-warn-bg)' : 'var(--color-danger-bg)',
                          color: h.expiry_status === 'VALID' ? 'var(--color-pass)' :
                                 h.expiry_status === 'EXPIRING_SOON' ? 'var(--color-warn)' : 'var(--color-danger)'
                        }}
                      >
                        {h.expiry_status}
                      </span>
                    </td>
                    <td className="mono">{h.expiry_date || 'N/A'}</td>
                    <td>
                      <span
                        className="tag-badge"
                        style={{
                          background: h.tamper_status === 'AUTHENTIC' ? 'var(--color-pass-bg)' :
                                      h.tamper_status === 'SUSPICIOUS' ? 'var(--color-warn-bg)' : 'var(--color-danger-bg)',
                          color: h.tamper_status === 'AUTHENTIC' ? 'var(--color-pass)' :
                                 h.tamper_status === 'SUSPICIOUS' ? 'var(--color-warn)' : 'var(--color-danger)'
                        }}
                      >
                        {h.tamper_status}
                      </span>
                    </td>
                    <td className="mono" style={{ fontWeight: 700 }}>{h.authenticity_score}%</td>
                    <td>
                      <strong style={{
                        color: h.verdict_status === 'PASS' ? 'var(--color-pass)' :
                               h.verdict_status === 'WARNING' ? 'var(--color-warn)' : 'var(--color-danger)'
                      }}>
                        {h.verdict_status}
                      </strong>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={8} style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '1.5rem' }}>
                    No audit records logged yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function int(val) {
  return Math.round(val);
}
