/**
 * ARIMA Sales Forecast - Frontend Application
 * Connects to Flask backend proxy for Azure ML Batch Endpoint
 */

// Configuration
const CONFIG = {
    API_BASE_URL: 'http://localhost:5000/api',
    POLL_INTERVAL: 5000, // 5 seconds
    MAX_POLL_ATTEMPTS: 120 // 10 minutes max
};

// State
let currentJobId = null;
let pollInterval = null;
let pollAttempts = 0;

// DOM Elements
const forecastForm = document.getElementById('forecastForm');
const submitBtn = document.getElementById('submitBtn');
const statusContainer = document.getElementById('statusContainer');
const statusTimeline = document.getElementById('statusTimeline');
const jobInfo = document.getElementById('jobInfo');
const resultsContainer = document.getElementById('resultsContainer');
const resultsTable = document.getElementById('resultsTable');
const resultsBody = document.getElementById('resultsBody');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    forecastForm.addEventListener('submit', handleFormSubmit);
});

// Toggle Advanced Options
function toggleAdvanced() {
    const toggle = document.querySelector('.collapsible-toggle');
    const content = document.getElementById('advancedOptions');
    toggle.classList.toggle('active');
    content.classList.toggle('active');
}

// Form Submit Handler
async function handleFormSubmit(e) {
    e.preventDefault();
    
    const storeId = document.getElementById('storeId').value;
    const featuresStr = document.getElementById('features').value;
    const blobUri = document.getElementById('blobUri').value;
    
    // Validate features
    const features = parseFeaturesInput(featuresStr);
    if (!features) {
        showToast('Please enter exactly 13 comma-separated numeric values', 'error');
        return;
    }
    
    // Prepare payload
    const payload = {
        store_id: storeId,
        features: features,
        blob_uri: blobUri || null
    };
    
    // Submit job
    await submitJob(payload);
}

// Parse Features Input
function parseFeaturesInput(input) {
    try {
        const values = input.split(',').map(v => parseFloat(v.trim()));
        if (values.length !== 13 || values.some(v => isNaN(v))) {
            return null;
        }
        return values;
    } catch {
        return null;
    }
}

// Submit Forecast Job
async function submitJob(payload) {
    setLoading(true);
    resetStatus();
    
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/submit-job`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Failed to submit job');
        }
        
        currentJobId = data.job_id;
        showToast('Job submitted successfully!', 'success');
        
        // Update UI
        showJobStatus(currentJobId, 'Pending');
        updateTimeline('submitted');
        document.getElementById('submittedTime').textContent = formatTime(new Date());
        
        // Start polling
        startPolling();
        
    } catch (error) {
        console.error('Submit error:', error);
        showToast(error.message, 'error');
    } finally {
        setLoading(false);
    }
}

// Start Job Status Polling
function startPolling() {
    pollAttempts = 0;
    
    pollInterval = setInterval(async () => {
        pollAttempts++;
        
        if (pollAttempts >= CONFIG.MAX_POLL_ATTEMPTS) {
            stopPolling();
            showToast('Job polling timed out. Please check Azure portal.', 'warning');
            return;
        }
        
        await checkJobStatus();
    }, CONFIG.POLL_INTERVAL);
}

// Stop Polling
function stopPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
}

// Check Job Status
async function checkJobStatus() {
    if (!currentJobId) return;
    
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/job-status/${currentJobId}`);
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Failed to get job status');
        }
        
        const status = data.status.toLowerCase();
        updateJobStatus(status);
        
        if (status === 'running' || status === 'starting') {
            updateTimeline('processing');
            document.getElementById('processingTime').textContent = formatTime(new Date());
        }
        
        if (status === 'completed') {
            stopPolling();
            updateTimeline('completed');
            document.getElementById('completedTime').textContent = formatTime(new Date());
            showToast('Forecast completed!', 'success');
            await fetchResults();
        }
        
        if (status === 'failed' || status === 'canceled') {
            stopPolling();
            showToast(`Job ${status}. ${data.error || ''}`, 'error');
        }
        
    } catch (error) {
        console.error('Status poll error:', error);
    }
}

// Fetch Results
async function fetchResults() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/results/${currentJobId}`);
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Failed to fetch results');
        }
        
        displayResults(data.predictions);
        
    } catch (error) {
        console.error('Results error:', error);
        showToast('Could not fetch results. Check Azure portal.', 'warning');
    }
}

// Display Results
function displayResults(predictions) {
    resultsContainer.classList.add('hidden');
    resultsTable.classList.remove('hidden');
    
    resultsBody.innerHTML = '';
    
    predictions.forEach(pred => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${escapeHtml(pred.store_id)}</td>
            <td>${parseFloat(pred.prediction).toFixed(4)}</td>
            <td>${pred.confidence ? (pred.confidence * 100).toFixed(1) + '%' : 'N/A'}</td>
        `;
        resultsBody.appendChild(row);
    });
}

// Show Job Status UI
function showJobStatus(jobId, status) {
    statusContainer.classList.add('hidden');
    statusTimeline.classList.remove('hidden');
    jobInfo.classList.remove('hidden');
    
    document.getElementById('jobId').textContent = jobId;
    updateJobStatus(status);
}

// Update Job Status Badge
function updateJobStatus(status) {
    const statusBadge = document.getElementById('jobStatus');
    statusBadge.textContent = status;
    statusBadge.className = 'status-badge ' + status.toLowerCase();
}

// Update Timeline
function updateTimeline(step) {
    const steps = ['submitted', 'processing', 'completed'];
    const currentIndex = steps.indexOf(step);
    
    steps.forEach((s, i) => {
        const item = document.querySelector(`.timeline-item[data-step="${s}"]`);
        item.classList.remove('active', 'completed');
        
        if (i < currentIndex) {
            item.classList.add('completed');
        } else if (i === currentIndex) {
            item.classList.add('active');
        }
    });
}

// Reset Status UI
function resetStatus() {
    statusContainer.classList.remove('hidden');
    statusTimeline.classList.add('hidden');
    jobInfo.classList.add('hidden');
    resultsContainer.classList.remove('hidden');
    resultsTable.classList.add('hidden');
    
    document.querySelectorAll('.timeline-item').forEach(item => {
        item.classList.remove('active', 'completed');
    });
    
    document.getElementById('submittedTime').textContent = '--';
    document.getElementById('processingTime').textContent = '--';
    document.getElementById('completedTime').textContent = '--';
}

// Set Loading State
function setLoading(loading) {
    submitBtn.disabled = loading;
    submitBtn.classList.toggle('loading', loading);
}

// Format Time
function formatTime(date) {
    return date.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
}

// Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Toast Notification
function showToast(message, type = 'success') {
    const container = document.getElementById('toastContainer');
    
    const icons = {
        success: '<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
        error: '<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
        warning: '<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
    };
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        ${icons[type]}
        <span class="toast-message">${escapeHtml(message)}</span>
    `;
    
    container.appendChild(toast);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}
