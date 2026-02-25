const videoElement = document.getElementById('input_video');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const resultText = document.getElementById('predictionResult');

let currentResults = null;

// 1. Setup MediaPipe
const hands = new Hands({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
}});

hands.setOptions({
  maxNumHands: 2,
  modelComplexity: 1,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});

hands.onResults(onResults);

// 2. Start Camera
const camera = new Camera(videoElement, {
  onFrame: async () => {
    await hands.send({image: videoElement});
  },
  width: 640,
  height: 480
});
camera.start();

// 3. Draw Hands
function onResults(results) {
  currentResults = results;
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  
  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    for (const landmarks of results.multiHandLandmarks) {
      drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {color: '#00FF00', lineWidth: 4});
      drawLandmarks(canvasCtx, landmarks, {color: '#FF0000', lineWidth: 2});
    }
  }
  canvasCtx.restore();
}

// 4. Constant Prediction Loop (Talks to Python)
function startPredictionLoop() {
    if (currentResults && currentResults.multiHandLandmarks && currentResults.multiHandLandmarks.length > 0) {
        
        // Send to Flask
        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                landmarks: currentResults.multiHandLandmarks,
                handedness: currentResults.multiHandedness // Crucial for Left/Right differentiation
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error("Server Error:", data.error);
            } else if (data.gesture) {
                resultText.innerText = `Prediction: ${data.gesture} (${(data.confidence * 100).toFixed(1)}%)`;
                
                // Visual feedback based on confidence
                if (data.confidence > 0.7) resultText.style.color = "lime";
                else resultText.style.color = "orange";
            }
        })
        .catch(error => {
            resultText.innerText = "Error: Python Server Offline?";
            resultText.style.color = "red";
        });
    } else {
        resultText.innerText = "Prediction: Waiting for hand...";
        resultText.style.color = "gray";
    }

    // Run again in 200ms
    setTimeout(startPredictionLoop, 200);
}

// Start the continuous loop immediately
startPredictionLoop();