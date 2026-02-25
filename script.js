const videoElement = document.getElementById('input_video');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const recordBtn = document.getElementById('recordBtn');
const statusText = document.getElementById('status');
const counterText = document.getElementById('counter');
const gestureSelector = document.getElementById('gestureLabel');

let collectedSequences = [];
let isRecording = false;
let recordedFrames = [];
let missedFramesCount = 0;

// 1. Setup MediaPipe
const hands = new Hands({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
}});

hands.setOptions({
  maxNumHands: 2,
  modelComplexity: 1,       
  minDetectionConfidence: 0.3, 
  minTrackingConfidence: 0.5   
});

hands.onResults(onResults);

// 2. Camera Setup
const camera = new Camera(videoElement, {
  onFrame: async () => {
    await hands.send({image: videoElement});
  },
  width: 640,
  height: 480
});
camera.start();

// 3. The Loop
function onResults(results) {
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  
  // Decide skeleton color (Red = Recording, White = Waiting)
  const landmarkColor = isRecording ? '#FF0000' : '#FFFFFF';
  const connectorColor = isRecording ? '#00FF00' : '#AAAAAA';

  // Draw Landmarks
  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    for (const landmarks of results.multiHandLandmarks) {
      drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {color: connectorColor, lineWidth: 4});
      drawLandmarks(canvasCtx, landmarks, {color: landmarkColor, lineWidth: 2});
    }
  }
  canvasCtx.restore();

  // --- RECORDING LOGIC ---
  if (isRecording) {
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        missedFramesCount = 0;

        // Save exactly what we see
        recordedFrames.push({
            landmarks: results.multiHandLandmarks,
        });
        
        statusText.innerText = `🔴 RECORDING: Frame ${recordedFrames.length} / 30`;
        statusText.style.color = "red";

        // EXPLICIT STOP AT 30 FRAMES
        if (recordedFrames.length === 30) {
            finishRecording();
        }

    } else {
        missedFramesCount++;
        if (missedFramesCount > 5) {
            isRecording = false;
            recordedFrames = [];
            statusText.innerText = "❌ Hand Lost! Try again.";
            statusText.style.color = "orange";
            recordBtn.disabled = false;
        } else if (recordedFrames.length > 0) {
            recordedFrames.push(recordedFrames[recordedFrames.length - 1]);
        }
    }
  }
}

// 4. Start Recording Button
recordBtn.addEventListener('click', () => {
    if (!isRecording) {
        isRecording = true;
        recordedFrames = [];
        missedFramesCount = 0;
        recordBtn.disabled = true; // Lock button
        statusText.innerText = "🔴 RECORDING STARTED...";
        statusText.style.color = "red";
    }
});

// 5. Finish & Wait Logic
function finishRecording() {
    isRecording = false; // STOP RECORDING
    const label = gestureSelector.value;
    
    // Save to our array
    collectedSequences.push({
        label: label,
        frames: recordedFrames
    });

    counterText.innerText = collectedSequences.length;
    
    // Clear the buffer so it doesn't accidentally keep filling
    recordedFrames = [];

    // UI Feedback: Tell user it stopped!
    statusText.innerText = `✅ Sequence Complete! Click Record for next.`;
    statusText.style.color = "cyan";
    
    // Flash the screen green to show a successful capture
    canvasElement.style.border = "5px solid lime";
    
    // Wait 1 second before allowing the next click
    setTimeout(() => {
        canvasElement.style.border = "none";
        recordBtn.disabled = false; // Unlock button
    }, 1000);
}

window.downloadData = function() {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(collectedSequences));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", "dynamic_dataset.json");
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
};