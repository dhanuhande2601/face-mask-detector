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
    imageSection.style.display = "none";
    videoSection.style.display = "block";

    navigator.mediaDevices.getUserMedia({ video: true })
        .then(s => {
            stream = s;
            video.srcObject = stream;
            video.play();

            // ✅ Prevent multiple intervals
            if (intervalId) clearInterval(intervalId);

            intervalId = setInterval(sendFrame, 1500);
        })
        .catch(err => console.error(err));
});

function sendFrame() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.drawImage(video, 0, 0);

    const dataURL = canvas.toDataURL("image/jpeg");

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
        imageSection.style.display = "block";
        videoSection.style.display = "none";

        previewImage.src = e.target.result;

        sendToServer(e.target.result);
    };

    reader.readAsDataURL(file);
});

// ================= API CALL =================

// ================= API CALL =================

function sendToServer(imageData) {
    fetch("/predict", {
        method: "POST",
        body: JSON.stringify({ image: imageData }),
        headers: {
            "Content-Type": "application/json"
        }
    })
    .then(res => res.json())
    .then(data => {

        // Set canvas size
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Draw frame
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        if (!data.faces || data.faces.length === 0) {
            resultText.innerText = "Status: No Face Detected";
            return;
        }

        let statusText = "Status:\n";

        data.faces.forEach(face => {
            let [x1, y1, x2, y2] = face.box;

            const color = face.label.includes("Mask") ? "lime" : "red";

            // Draw rectangle
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

            // Draw label
            const text = `${face.label} (${face.confidence}%)`;

            ctx.fillStyle = color;
            ctx.font = "16px Arial";
            ctx.fillText(text, x1, y1 - 10);

            statusText += text + "\n";
        });

        resultText.innerText = statusText;
    })
    .catch(err => console.error(err));
}
// ================= BACK BUTTON =================

backBtn.addEventListener("click", () => {

    // Stop camera
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    // ✅ Stop interval
    if (intervalId) {
        clearInterval(intervalId);
        intervalId = null;
    }

    // Hide sections
    videoSection.style.display = "none";
    imageSection.style.display = "none";

    // Reset UI
    resultText.innerText = "Status: Waiting...";
});
// ================= IMAGE BACK BUTTON =================

backImageBtn.addEventListener("click", () => {
    imageSection.style.display = "none";
    resultText.innerText = "Status: Waiting...";
});