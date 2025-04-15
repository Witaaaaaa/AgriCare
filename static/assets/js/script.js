document.addEventListener("DOMContentLoaded", function () {
    const mobileToggle = document.querySelector(".mobile-nav-toggle");
    const body = document.querySelector("body");
  
    if (mobileToggle) {
        mobileToggle.addEventListener("click", function () {
            body.classList.toggle("mobile-nav-active");
            console.log("Navbar Mobile: " + (body.classList.contains("mobile-nav-active") ? "Aktif" : "Nonaktif"));
        });
    } else {
        console.error("Tombol navbar tidak ditemukan!");
    }
  });
  
document.getElementById("capture").addEventListener("click", function() {

        fetch("/capture", { method: "POST" })
            .then(response => response.json())
            .then(result => {
                if (result.image) {
                    document.getElementById("detected-image").src = result.image;
                    document.getElementById("result-container").style.display = "block";
                    alert("Gambar berhasil diproses!"); // Notifikasi sukses
                } else {
                    alert("Error: " + result.error);
                }
            })
            .catch(error => console.error("Error:", error));
    });

document.getElementById("upload-form").onsubmit = async (event) => {
    event.preventDefault();

    let formData = new FormData(event.target);

    let response = await fetch("/check", {
        method: "POST",
        body: formData
    });

    let result = await response.json();

    if (result.image) {
        document.getElementById("upload-detected-image").src = result.image; // Use the correct URL from the server
        document.getElementById("upload-result-container").style.display = "block";

        let accuracyText = "";
        result.detections.forEach(detection => {
            accuracyText += `Label: ${detection['name']} | Akurasi: ${(detection['confidence'] * 100).toFixed(2)}%<br>`;
        });

        document.getElementById("upload-accuracy").innerHTML = accuracyText;

        alert("Upload berhasil! Gambar telah diproses."); // Notifikasi sukses
    } else {
        alert("Error: " + result.error);
    }
};

document.getElementById("start-video").addEventListener("click", function () {
    const videoContainer = document.getElementById("video-detection-container");
    const videoElement = document.getElementById("video-detection");

    // Tampilkan container video
    videoContainer.style.display = "block";

    // Mulai stream video dengan deteksi real-time
    videoElement.src = "/video_feed_with_detection";
});

document.getElementById("close-camera").addEventListener("click", function () {
    const videoContainer = document.getElementById("video-detection-container");
    const videoElement = document.getElementById("video-detection");

    // Sembunyikan container video dan hentikan stream
    videoContainer.style.display = "none";
    videoElement.src = "";
});

function startVideoDetection() {
    const videoContainer = document.getElementById("video-detection-container");
    const videoElement = document.getElementById("video-detection");

    videoContainer.style.display = "block";
    videoElement.src = "/video_feed_with_detection"; // Set the video feed URL
}

function stopVideoDetection() {
    const videoContainer = document.getElementById("video-detection-container");
    const videoElement = document.getElementById("video-detection");

    videoContainer.style.display = "none";
    videoElement.src = ""; // Stop the video stream
}

function startVideoFeature() {
  const videoContainer = document.getElementById("video-feature-container");
  const videoElement = document.getElementById("video-feature");

  videoContainer.style.display = "block";
  videoElement.src = "/video_feed"; // Endpoint for video streaming
}

function stopVideoFeature() {
  const videoContainer = document.getElementById("video-feature-container");
  const videoElement = document.getElementById("video-feature");

  videoContainer.style.display = "none";
  videoElement.src = ""; // Stop the video stream
}

function openVideo() {
  const videoModal = document.getElementById("video-modal");
  const videoElement = document.getElementById("video-detection");

  videoModal.style.display = "flex"; // Show the video modal
  videoElement.src = "/video_feed_with_detection"; // Set the video feed URL
}

function stopVideo() {
  const videoModal = document.getElementById("video-modal");
  const videoElement = document.getElementById("video-detection");

  videoModal.style.display = "none"; // Hide the video modal
  videoElement.src = ""; // Stop the video stream
}

function closeVideo() {
  const videoModal = document.getElementById("video-modal");
  const videoElement = document.getElementById("video-detection");

  videoModal.style.display = "none"; // Hide the video modal
  videoElement.src = ""; // Stop the video stream
}

function saveVideoFrame() {
    const videoElement = document.getElementById("video-detection");
    if (!videoElement.src) {
        alert("Video belum dimulai!");
        return;
    }

    const canvas = document.createElement("canvas");
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const context = canvas.getContext("2d");

    // Pastikan video sudah berjalan sebelum menangkap frame
    if (videoElement.readyState < 2) {
        alert("Video belum siap!");
        return;
    }

    context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

    const imageData = canvas.toDataURL("image/png");

    // Kirim frame ke server untuk disimpan
    fetch("/save_video_frame", {
        method: "POST",
        body: JSON.stringify({ image: imageData }),
        headers: { "Content-Type": "application/json" }
    })
    .then(response => response.json())
    .then(result => {
        if (result.success) {
            alert("Frame berhasil disimpan!");
        } else {
            alert("Error: " + result.error);
        }
    })
    .catch(error => console.error("Error:", error));
}