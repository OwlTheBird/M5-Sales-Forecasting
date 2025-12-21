import { useState, useEffect } from 'react'
import './App.css'

// API Base URL
const API_URL = 'http://localhost:8000'

// Default form values
const defaultFormData = {
  store_id: 'CA_1',
  dept_id: 'FOODS_1',
  cat_id: 'FOODS',
  state_id: 'CA',
  wday: 1,
  month: 3,
  year: 2016,
  sell_price: 2.99,
  event_name_1: '',
  event_type_1: '',
  snap_CA: 0,
  snap_TX: 0,
  snap_WI: 0,
  lag_28: 5.0,
  lag_35: 4.0,
  lag_42: 6.0,
  lag_49: 3.0,
  roll_mean_28: 4.5,
  roll_std_28: 1.2,
  price_max: 3.99,
  price_momentum: 0.95,
  price_roll_std_7: 0.1
}

// Dropdown options
const OPTIONS = {
  store_id: ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3'],
  dept_id: ['FOODS_1', 'FOODS_2', 'FOODS_3', 'HOBBIES_1', 'HOBBIES_2', 'HOUSEHOLD_1', 'HOUSEHOLD_2'],
  cat_id: ['FOODS', 'HOBBIES', 'HOUSEHOLD'],
  state_id: ['CA', 'TX', 'WI'],
  event_type_1: ['', 'Sporting', 'Cultural', 'National', 'Religious']
}

function App() {
  const [modelType, setModelType] = useState('lightgbm')
  const [formData, setFormData] = useState(defaultFormData)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [apiHealth, setApiHealth] = useState(null)

  // Check API health on mount
  useEffect(() => {
    checkHealth()
  }, [])

  const checkHealth = async () => {
    try {
      const response = await fetch(`${API_URL}/health`)
      const data = await response.json()
      setApiHealth(data)
    } catch (err) {
      setApiHealth({ status: 'unhealthy' })
    }
  }

  const handleInputChange = (e) => {
    const { name, value, type } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: type === 'number' ? parseFloat(value) : value
    }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_type: modelType,
          data: {
            ...formData,
            event_name_1: formData.event_name_1 || null,
            event_type_1: formData.event_type_1 || null
          }
        })
      })

      if (!response.ok) {
        const err = await response.json()
        throw new Error(err.detail || 'Prediction failed')
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <div className="container">
        {/* Header */}
        <header className="header">
          <h1>
            <span className="icon">📊</span>
            <span className="gradient-text">M5 Sales Forecasting</span>
          </h1>
          <p className="subtitle">
            Predict daily sales for Walmart products using machine learning models
          </p>

          {apiHealth && (
            <div className={`status-badge ${apiHealth.status === 'healthy' ? 'healthy' : 'unhealthy'}`}>
              <span className="status-dot"></span>
              API {apiHealth.status === 'healthy' ? 'Connected' : 'Disconnected'}
            </div>
          )}
        </header>

        {/* Main Grid */}
        <div className="main-grid">
          {/* Prediction Form */}
          <div className="card prediction-form">
            <h2>
              <span>🎯</span>
              Make Prediction
            </h2>

            {/* Model Selector */}
            <div className="model-selector">
              <button
                type="button"
                className={`model-option ${modelType === 'lightgbm' ? 'active' : ''}`}
                onClick={() => setModelType('lightgbm')}
              >
                ⚡ LightGBM
              </button>
              <button
                type="button"
                className={`model-option ${modelType === 'gbtregressor' ? 'active' : ''}`}
                onClick={() => setModelType('gbtregressor')}
              >
                🌲 GBT Regressor
              </button>
              <button
                type="button"
                className={`model-option ${modelType === 'hybrid' ? 'active' : ''}`}
                onClick={() => setModelType('hybrid')}
              >
                🔀 Hybrid Model
              </button>
            </div>

            <form onSubmit={handleSubmit}>
              {/* Store Info */}
              <div className="form-section">
                <h3>Store Information</h3>
                <div className="form-grid">
                  <div className="form-group">
                    <label>Store ID</label>
                    <select name="store_id" value={formData.store_id} onChange={handleInputChange}>
                      {OPTIONS.store_id.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                    </select>
                  </div>
                  <div className="form-group">
                    <label>Department</label>
                    <select name="dept_id" value={formData.dept_id} onChange={handleInputChange}>
                      {OPTIONS.dept_id.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                    </select>
                  </div>
                  <div className="form-group">
                    <label>Category</label>
                    <select name="cat_id" value={formData.cat_id} onChange={handleInputChange}>
                      {OPTIONS.cat_id.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                    </select>
                  </div>
                  <div className="form-group">
                    <label>State</label>
                    <select name="state_id" value={formData.state_id} onChange={handleInputChange}>
                      {OPTIONS.state_id.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                    </select>
                  </div>
                </div>
              </div>

              {/* Time Features */}
              <div className="form-section">
                <h3>Time Features</h3>
                <div className="form-grid">
                  <div className="form-group">
                    <label>Day of Week</label>
                    <input type="number" name="wday" min="1" max="7" value={formData.wday} onChange={handleInputChange} />
                  </div>
                  <div className="form-group">
                    <label>Month</label>
                    <input type="number" name="month" min="1" max="12" value={formData.month} onChange={handleInputChange} />
                  </div>
                  <div className="form-group">
                    <label>Year</label>
                    <input type="number" name="year" min="2011" max="2030" value={formData.year} onChange={handleInputChange} />
                  </div>
                </div>
              </div>

              {/* Price Features */}
              <div className="form-section">
                <h3>Price Features</h3>
                <div className="form-grid">
                  <div className="form-group">
                    <label>Sell Price ($)</label>
                    <input type="number" name="sell_price" step="0.01" min="0" value={formData.sell_price} onChange={handleInputChange} />
                  </div>
                  <div className="form-group">
                    <label>Max Price ($)</label>
                    <input type="number" name="price_max" step="0.01" min="0" value={formData.price_max} onChange={handleInputChange} />
                  </div>
                  <div className="form-group">
                    <label>Price Momentum</label>
                    <input type="number" name="price_momentum" step="0.01" min="0" max="1" value={formData.price_momentum} onChange={handleInputChange} />
                  </div>
                  <div className="form-group">
                    <label>Price Volatility</label>
                    <input type="number" name="price_roll_std_7" step="0.01" min="0" value={formData.price_roll_std_7} onChange={handleInputChange} />
                  </div>
                </div>
              </div>

              {/* Lag Features */}
              <div className="form-section">
                <h3>Historical Sales (Lag Features)</h3>
                <div className="form-grid">
                  <div className="form-group">
                    <label>28 Days Ago</label>
                    <input type="number" name="lag_28" step="0.1" value={formData.lag_28} onChange={handleInputChange} />
                  </div>
                  <div className="form-group">
                    <label>35 Days Ago</label>
                    <input type="number" name="lag_35" step="0.1" value={formData.lag_35} onChange={handleInputChange} />
                  </div>
                  <div className="form-group">
                    <label>42 Days Ago</label>
                    <input type="number" name="lag_42" step="0.1" value={formData.lag_42} onChange={handleInputChange} />
                  </div>
                  <div className="form-group">
                    <label>49 Days Ago</label>
                    <input type="number" name="lag_49" step="0.1" value={formData.lag_49} onChange={handleInputChange} />
                  </div>
                </div>
              </div>

              {/* Rolling Statistics */}
              <div className="form-section">
                <h3>Rolling Statistics (28-day)</h3>
                <div className="form-grid">
                  <div className="form-group">
                    <label>Rolling Mean</label>
                    <input type="number" name="roll_mean_28" step="0.1" value={formData.roll_mean_28} onChange={handleInputChange} />
                  </div>
                  <div className="form-group">
                    <label>Rolling Std</label>
                    <input type="number" name="roll_std_28" step="0.1" value={formData.roll_std_28} onChange={handleInputChange} />
                  </div>
                </div>
              </div>

              {/* Event & SNAP */}
              <div className="form-section">
                <h3>Events & SNAP</h3>
                <div className="form-grid">
                  <div className="form-group">
                    <label>Event Type</label>
                    <select name="event_type_1" value={formData.event_type_1} onChange={handleInputChange}>
                      {OPTIONS.event_type_1.map(opt => <option key={opt} value={opt}>{opt || 'None'}</option>)}
                    </select>
                  </div>
                  <div className="form-group">
                    <label>SNAP CA</label>
                    <select name="snap_CA" value={formData.snap_CA} onChange={handleInputChange}>
                      <option value={0}>No</option>
                      <option value={1}>Yes</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label>SNAP TX</label>
                    <select name="snap_TX" value={formData.snap_TX} onChange={handleInputChange}>
                      <option value={0}>No</option>
                      <option value={1}>Yes</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label>SNAP WI</label>
                    <select name="snap_WI" value={formData.snap_WI} onChange={handleInputChange}>
                      <option value={0}>No</option>
                      <option value={1}>Yes</option>
                    </select>
                  </div>
                </div>
              </div>

              <button type="submit" className="btn-primary submit-btn" disabled={loading}>
                {loading ? (
                  <>
                    <span className="spinner"></span>
                    Predicting...
                  </>
                ) : (
                  <>
                    🚀 Predict Sales
                  </>
                )}
              </button>
            </form>
          </div>

          {/* Results Panel */}
          <div className="card results-card">
            <h2>
              <span>📈</span>
              Prediction Result
            </h2>

            {error && (
              <div className="error-message">
                <span className="icon">⚠️</span>
                <span>{error}</span>
              </div>
            )}

            {!result && !error && (
              <div className="empty-state">
                <span className="icon">🔮</span>
                <p>Enter features and click "Predict Sales" to see the forecast</p>
              </div>
            )}

            {result && (
              <div className="prediction-result">
                <div className="result-value">
                  <span className="result-label">Predicted Daily Sales</span>
                  <span className="result-number gradient-text">{result.predicted_sales}</span>
                  <span className="result-unit">units</span>
                </div>

                <div className="result-details">
                  <div className="detail-item">
                    <span className="label">Model Used</span>
                    <span className="value">
                      {result.model_used === 'lightgbm' && '⚡ LightGBM'}
                      {result.model_used === 'gbtregressor' && '🌲 GBT Regressor'}
                      {result.model_used === 'hybrid' && '🔀 Hybrid Model'}
                    </span>
                  </div>
                  <div className="detail-item">
                    <span className="label">Status</span>
                    <span className="value" style={{ color: 'var(--success)' }}>✓ Success</span>
                  </div>
                  <div className="detail-item">
                    <span className="label">Store</span>
                    <span className="value">{formData.store_id}</span>
                  </div>
                  <div className="detail-item">
                    <span className="label">Category</span>
                    <span className="value">{formData.cat_id}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <footer className="footer">
          <p>
            M5 Sales Forecasting Dashboard • Built with FastAPI + React •
            <a href="http://localhost:8000/docs" target="_blank" rel="noopener noreferrer"> API Docs</a>
          </p>
        </footer>
      </div>
    </div>
  )
}

export default App
