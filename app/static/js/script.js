const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const cameraBtn = document.getElementById("cameraBtn");
const uploadBtn = document.getElementById("uploadBtn");
const fileInput = document.getElementById("fileInput");

const resultText = document.getElementById("result");

const videoSection = document.getElementById("videoSection");
const imageSection = document.getElementById("imageSection");
const previewImage = document.getElementById("previewImage");
const backBtn = document.getElementById("backBtn");
const backImageBtn = document.getElementById("backImageBtn");

let stream = null;
let intervalId = null;  // ✅ FIX

// ================= CAMERA =================

cameraBtn.addEventListener("click", () => {

    // Check browser support
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("Camera not supported. Use Chrome + localhost.");
        return;
    }

    // Show video section
    imageSection.style.display = "none";
    videoSection.style.display = "block";

    navigator.mediaDevices.getUserMedia({
        video: true
    })
    .then(s => {

        stream = s;

        video.srcObject = stream;

        video.play();

        // Prevent multiple intervals
        if (intervalId) {
            clearInterval(intervalId);
        }

        // Send frame every 1 second
        intervalId = setInterval(sendFrame, 1000);
    })
    .catch(err => {
        console.error("Camera Error:", err);
        alert("Unable to access camera");
    });
});

// ================= SEND VIDEO FRAME =================

function sendFrame() {

    // Wait until video fully loads
    if (!video.videoWidth || !video.videoHeight) {
        return;
    }

    // Match canvas size to video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw current frame
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert frame to image
    const dataURL = canvas.toDataURL("image/jpeg");

    // Send to backend
    sendToServer(dataURL);
}

// ================= IMAGE UPLOAD =================

uploadBtn.addEventListener("click", () => {
    fileInput.click();
});

fileInput.addEventListener("change", () => {

    const file = fileInput.files[0];

    if (!file) return;

    const reader = new FileReader();

    reader.onload = function(e) {

        // Hide camera section
        videoSection.style.display = "none";

        // Show image section
        imageSection.style.display = "block";

        // Show image preview
        previewImage.src = e.target.result;

        // Send image to backend
        sendToServer(e.target.result);
    };

    reader.readAsDataURL(file);
});

// ================= API CALL =================

function sendToServer(imageData) {

    fetch("/predict", {
        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify({
            image: imageData
        })
    })

    .then(response => response.json())

    .then(data => {

        // Clear old drawings
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Redraw video frame
        if (video.videoWidth && video.videoHeight) {
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        }

        // No face detected
        if (!data.faces || data.faces.length === 0) {

            resultText.innerText = "Status: No Face Detected";

            return;
        }

        let statusText = "Status:\n";

        // Draw all faces
        data.faces.forEach(face => {

            let [x1, y1, x2, y2] = face.box;

            // Color
            const color = face.label.includes("Mask")
                ? "lime"
                : "red";

            // Rectangle
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;

            ctx.strokeRect(
                x1,
                y1,
                x2 - x1,
                y2 - y1
            );

            // Text
            const text =
                `${face.label} (${face.confidence}%)`;

            ctx.fillStyle = color;
            ctx.font = "18px Arial";

            ctx.fillText(
                text,
                x1,
                y1 - 10
            );

            statusText += text + "\n";
        });

        // Show result
        resultText.innerText = statusText;
    })

    .catch(err => {
        console.error("Prediction Error:", err);
    });
}

// ================= BACK BUTTON =================

backBtn.addEventListener("click", () => {

    // Stop camera
    if (stream) {

        stream.getTracks().forEach(track => {
            track.stop();
        });

        stream = null;
    }

    // Stop interval
    if (intervalId) {

        clearInterval(intervalId);

        intervalId = null;
    }

    // Hide sections
    videoSection.style.display = "none";

    imageSection.style.display = "none";

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Reset text
    resultText.innerText = "Status: Waiting...";
});

// ================= IMAGE BACK BUTTON =================

backImageBtn.addEventListener("click", () => {

    imageSection.style.display = "none";

    resultText.innerText = "Status: Waiting...";
});