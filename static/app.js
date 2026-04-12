// AgriEnv Premium Dashboard JS
let socket = null;
let trendsChart = null;
let rewardChart = null;
let stepHistory = [];
let moistureHistory = [];
let nutrientHistory = [];
let rewardHistory = [];
let isAutoPilot = false;
let currentTask = 'medium';

// Constants for Heuristic (simplified for JS)
const TARGETS = {
    easy: { moisture: 0.70, n: 0.6, p: 0.6, k: 0.6 },
    medium: { moisture: 0.70, n: 0.65, p: 0.65, k: 0.65 },
    hard: { moisture: 0.68, n: 0.7, p: 0.65, k: 0.7 }
};

document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    connectWebSocket();
    setupEventListeners();
});

function initCharts() {
    const trendsCtx = document.getElementById('trendsChart').getContext('2d');
    trendsChart = new Chart(trendsCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Moisture',
                    borderColor: '#00f5d4',
                    borderWidth: 2,
                    data: [],
                    tension: 0.3,
                    pointRadius: 0
                },
                {
                    label: 'Nutrients (Avg)',
                    borderColor: '#4361ee',
                    borderWidth: 2,
                    data: [],
                    tension: 0.3,
                    pointRadius: 0
                },
                {
                    label: 'Pests',
                    borderColor: '#fee440',
                    borderWidth: 1,
                    data: [],
                    tension: 0.3,
                    pointRadius: 0,
                    borderDash: [5, 5]
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { min: 0, max: 1, grid: { color: 'rgba(255,255,255,0.05)' } },
                x: { grid: { display: false } }
            },
            plugins: { legend: { display: false } }
        }
    });

    const rewardCtx = document.getElementById('rewardChart').getContext('2d');
    rewardChart = new Chart(rewardCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Reward',
                borderColor: '#f15bb5',
                backgroundColor: 'rgba(241, 91, 181, 0.1)',
                fill: true,
                data: [],
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { grid: { color: 'rgba(255,255,255,0.05)' } },
                x: { grid: { display: false } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    // Get the base path without the filename (e.g., /index.html -> /)
    let basePath = window.location.pathname.split('/').slice(0, -1).join('/') + '/';
    const wsUrl = `${protocol}//${window.location.host}${basePath}ws`;
    
    socket = new WebSocket(wsUrl);
    const statusDot = document.querySelector('.status-indicator');
    const statusText = document.querySelector('.status-text');

    socket.onopen = () => {
        statusDot.classList.add('connected');
        statusText.innerText = 'Connected';
        resetEnv();
    };

    socket.onclose = () => {
        statusDot.classList.remove('connected');
        statusText.innerText = 'Disconnected. Retrying...';
        setTimeout(connectWebSocket, 3000);
    };

    socket.onmessage = (event) => {
        try {
            const response = JSON.parse(event.data);
            console.log("WebSocket message received:", response);

            if (response.type === 'observation') {
                const payload = response.data;
                // OpenEnv observations include observation (dict), reward, and done in the payload
                updateUI(payload.observation, payload.reward, payload.done);
                
                if (isAutoPilot && !payload.done) {
                    setTimeout(() => runHeuristicStep(payload.observation), 100);
                }
            } else if (response.type === 'error') {
                console.error("Simulation error:", response.data);
            }
        } catch (e) {
            console.error("Failed to parse message:", e, event.data);
        }
    };
}

function setupEventListeners() {
    document.getElementById('step-btn').addEventListener('click', manualStep);
    document.getElementById('reset-btn').addEventListener('click', resetEnv);
    document.getElementById('task-select').addEventListener('change', (e) => {
        currentTask = e.target.value;
        resetEnv();
    });

    document.getElementById('auto-btn').addEventListener('click', () => {
        isAutoPilot = !isAutoPilot;
        const btn = document.getElementById('auto-btn');
        btn.innerText = isAutoPilot ? "Stop Autopilot" : "Resume RL Autopilot";
        btn.classList.toggle('active', isAutoPilot);
        if (isAutoPilot) manualStep();
    });

    // Slider range labels
    const ranges = ['irrigation', 'npk', 'co2', 'pest'];
    ranges.forEach(id => {
        const input = document.getElementById(`input-${id}`);
        const label = document.getElementById(`label-${id}`);
        input.addEventListener('input', (e) => {
            label.innerText = e.target.value;
        });
    });
}

function resetEnv() {
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: 'reset', data: { task: currentTask } }));
        stepHistory = [];
        moistureHistory = [];
        nutrientHistory = [];
        rewardHistory = [];
        updateCharts();
    }
}

function manualStep() {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;

    const action = {
        irrigation: parseFloat(document.getElementById('input-irrigation').value),
        nitrogen_injection: parseFloat(document.getElementById('input-npk').value),
        phosphorus_injection: parseFloat(document.getElementById('input-npk').value) * 0.9,
        potassium_injection: parseFloat(document.getElementById('input-npk').value) * 0.95,
        co2_ppm: parseFloat(document.getElementById('input-co2').value),
        pesticide: parseFloat(document.getElementById('input-pest').value)
    };

    socket.send(JSON.stringify({ type: 'step', data: action }));
}

function runHeuristicStep(obs) {
    if (!isAutoPilot) return;

    const target = TARGETS[currentTask] || TARGETS.medium;
    
    // Very simple proportional controller for the demo
    const irrigation = Math.max(0, (target.moisture - obs.soil_moisture) * 1500 + 400);
    const npk = Math.max(0, (target.n - obs.nitrogen) * 0.5 + 0.05);
    const co2 = 500 + (1 - obs.energy_price) * 400;
    const pesticide = obs.pest_density > 0.15 ? 0.1 : 0.0;

    const action = {
        irrigation: Math.min(2500, irrigation),
        nitrogen_injection: Math.min(0.4, npk),
        phosphorus_injection: Math.min(0.4, npk * 0.9),
        potassium_injection: Math.min(0.4, npk * 0.9),
        co2_ppm: Math.min(1000, co2),
        pesticide: pesticide
    };

    socket.send(JSON.stringify({ type: 'step', data: action }));
}

function updateUI(obs, reward, done) {
    document.getElementById('val-soil-moisture').innerText = obs.soil_moisture.toFixed(3);
    document.getElementById('val-nitrogen').innerText = obs.nitrogen.toFixed(3);
    document.getElementById('val-pests').innerText = obs.pest_density.toFixed(3);
    document.getElementById('val-yield').innerText = obs.cumulative_yield.toFixed(2);
    
    document.getElementById('step-count').innerText = stepHistory.length + 1;
    document.getElementById('current-stage').innerText = `Growth Stage: ${obs.stage_name || 'Active'}`;
    
    if (reward !== undefined) {
        document.getElementById('reward-value').innerText = reward.toFixed(2);
        rewardHistory.push(reward);
    }

    stepHistory.push(stepHistory.length);
    moistureHistory.push(obs.soil_moisture);
    nutrientHistory.push((obs.nitrogen + obs.phosphorus + obs.potassium) / 3);
    
    updateCharts();

    if (done) {
        isAutoPilot = false;
        document.getElementById('auto-btn').innerText = "Simulation Done (Reset to restart)";
        document.getElementById('auto-btn').classList.remove('active');
    }
}

function updateCharts() {
    trendsChart.data.labels = stepHistory;
    trendsChart.data.datasets[0].data = moistureHistory;
    trendsChart.data.datasets[1].data = nutrientHistory;
    trendsChart.update('none');

    rewardChart.data.labels = stepHistory;
    rewardChart.data.datasets[0].data = rewardHistory;
    rewardChart.update('none');
}
