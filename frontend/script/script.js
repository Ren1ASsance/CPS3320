const uploadBtn = document.getElementById("upload-btn");
const fileInput = document.getElementById("file-input");
const previewImg = document.getElementById("preview-img");
const predictionElement = document.getElementById("prediction");
const dragDropArea = document.getElementById("drag-drop-area");

// Handle button click to open file picker
uploadBtn.addEventListener("click", () => {
  fileInput.click();  // Trigger the file input to open file picker
});

// Display the selected image on the page
fileInput.addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (file && file.type.startsWith("image/")) {
    // Show the selected image on the page
    const reader = new FileReader();
    reader.onload = function(e) {
      previewImg.src = e.target.result;
      previewImg.style.display = "block";
    };
    reader.readAsDataURL(file);

    // Call the backend API to get the prediction
    await handleFileUpload(file);
  } else {
    alert("Please upload a valid image file.");
  }
});

// Handle drag and drop functionality
dragDropArea.addEventListener("dragover", (event) => {
  event.preventDefault();
  dragDropArea.style.border = "2px solid #4e9f3d"; // Change border color on drag over
});

dragDropArea.addEventListener("dragleave", () => {
  dragDropArea.style.border = "2px dashed #4e9f3d"; // Reset border color when drag leaves
});

dragDropArea.addEventListener("drop", async (event) => {
  event.preventDefault();
  const file = event.dataTransfer.files[0];

  if (file && file.type.startsWith("image/")) {
    // Show the selected image on the page
    const reader = new FileReader();
    reader.onload = function(e) {
      previewImg.src = e.target.result;
      previewImg.style.display = "block";
    };
    reader.readAsDataURL(file);

    // Call the backend API to get the prediction
    await handleFileUpload(file);
  } else {
    alert("Please drop a valid image file.");
  }
});

// Upload the image and get prediction
async function handleFileUpload(file) {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("http://127.0.0.1:8000/predict/", {
      method: "POST",
      body: formData
    });

    if (response.ok) {
      const data = await response.json();
      predictionElement.textContent = data.prediction;
    } else {
      alert("Failed to get prediction. Try again!");
    }
  } catch (error) {
    console.error("Error:", error);
    alert("An error occurred. Please try again.");
  }
}
