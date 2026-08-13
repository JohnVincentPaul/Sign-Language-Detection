// Grab DOM elements
const videoElement = document.getElementById('input_video');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const resultText = document.getElementById('predictionResult');

let currentResults = null;

// ===== 1. Setup MediaPipe Hands =====
const hands = new Hands({
  locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
  }
});

hands.setOptions({
  maxNumHands: 2,
  modelComplexity: 1,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});

hands.onResults(onResults);

// ===== 2. Start Camera =====
const camera = new Camera(videoElement, {
  onFrame: async () => {
    await hands.send({ image: videoElement });
  },
  width: 640,
  height: 480
});

camera.start();

// ===== 3. Drawing callback =====
function onResults(results) {
  currentResults = results;

  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    for (const landmarks of results.multiHandLandmarks) {
      drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
        color: '#00FF00',
        lineWidth: 4
      });
      drawLandmarks(canvasCtx, landmarks, {
        color: '#FF0000',
        lineWidth: 2
      });
    }
  }
  canvasCtx.restore();
}

// ===== 4. Temporal smoothing for predictions =====
let lastPreds = [];
const SMOOTHING_WINDOW = 7;  

function updatePredictionDisplay(gesture, confidence) {
  lastPreds.push(gesture);
  if (lastPreds.length > SMOOTHING_WINDOW) {
    lastPreds.shift();
  }

  const counts = {};
  for (const g of lastPreds) {
    counts[g] = (counts[g] || 0) + 1;
  }

  const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const stableGesture = sorted[0][0];

  resultText.innerText = `Prediction: ${stableGesture} (${(confidence * 100).toFixed(1)}%)`;

  if (confidence > 0.7) {
    resultText.style.color = 'lime';
  } else if (confidence > 0.4) {
    resultText.style.color = 'orange';
  } else {
    resultText.style.color = 'gray';
  }
}

// ===== 5. Continuous prediction loop (Anti-Lag Enabled) =====
let isPredicting = false; // Prevents the browser from sending requests too fast

function startPredictionLoop() {
  if (
    !isPredicting && 
    currentResults &&
    currentResults.multiHandLandmarks &&
    currentResults.multiHandLandmarks.length > 0
  ) {
    isPredicting = true; // Lock the gate

    fetch('http://127.0.0.1:5000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        landmarks: currentResults.multiHandLandmarks
      })
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.error) {
          console.error('Server Error:', data.error);
          resultText.innerText = 'Error from server';
          resultText.style.color = 'red';
        } else if (data.gesture && data.gesture !== 'Unsure') {
          updatePredictionDisplay(data.gesture, data.confidence);
        } else {
          resultText.innerText = 'Prediction: Unsure';
          resultText.style.color = 'gray';
        }
        isPredicting = false; // Unlock the gate
      })
      .catch((error) => {
        console.error(error);
        resultText.innerText = 'Error: Python server offline?';
        resultText.style.color = 'red';
        isPredicting = false; // Unlock the gate on error
      });
  } else if (!isPredicting) {
    resultText.innerText = 'Prediction: Waiting for hand...';
    resultText.style.color = 'gray';
  }

  // Request prediction 10 times a second (100ms) to let the CPU breathe
  setTimeout(startPredictionLoop, 100); 
}

// ===== 6. Start prediction loop =====
startPredictionLoop();