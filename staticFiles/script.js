const canvas = document.getElementById("canvas");
const openModalButtons = document.querySelectorAll('[data-modal-target]');
const closeModalButtons = document.querySelectorAll('[data-close-button]');
const overlay = document.getElementById('overlay');
const test = document.getElementById("save-img");
const topic2222 = document.getElementById("topic");
const topics = ['airplane', 'monkey', 'skull'];
// , 'brain', 'door', 'mug', 'table', 'stop-sign', 'line', 'butterfly'];
sizeSlider = document.querySelector("#size-slider"),
clearCanvas = document.querySelector(".clear-canvas"),
saveImg = document.querySelector(".save-img"),
directions = document.querySelector(".directions"),

ctx = canvas.getContext("2d")
let isDrawing = false;
selectedTool = "brush",
brushWidth = 5;
var timer = 8;

openModalButtons.forEach(button => {
    button.addEventListener('click', () => {
      const modal = document.querySelector(button.dataset.modalTarget)
      openModal(modal)
    })
  })
  
  overlay.addEventListener('click', () => {
    const modals = document.querySelectorAll('.modal.active')
    modals.forEach(modal => {
      closeModal(modal)
    })
  })
  
  closeModalButtons.forEach(button => {
    button.addEventListener('click', () => {
      const modal = button.closest('.modal')
      closeModal(modal)
    })
  })
  
  function openModal(modal) {
    if (modal == null) return
    modal.classList.add('active')
    overlay.classList.add('active')
  }
  
  function closeModal(modal) {
    if (modal == null) return
    modal.classList.remove('active')
    overlay.classList.remove('active')
  }

function change() {
    test.innerHTML = "Start";
    test.style.background = 'green';
}

const setCanvasBackground = () => {
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#fff";
}

window.addEventListener("load", () => {
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    setCanvasBackground();
})

const startDraw = () => {
    isDrawing = true;
    ctx.beginPath(); // creating new path to draw
    ctx.lineWidth = brushWidth; // passing brush width
}

const drawing = (e) => { // if isDrawing is false, return here
    if (!isDrawing) return;
    ctx.lineTo(e.offsetX, e.offsetY); // Creating line according to mouse pointer
    ctx.stroke(); // drawing/filling line with color
}

function convertCanvasToImage() {
    let canvas = document.getElementById("canvas");
    let image = new Image();
    image.src = canvas.toDataURL();
    return image;
}

clearCanvas.addEventListener("click", () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    setCanvasBackground();
});


saveImg.addEventListener("click", () => {
    // Change buttons as needed
    test.style.visibility = 'hidden';
    topic2222.innerHTML = topics[(Math.floor(Math.random() * (topics.length)))];
    // Start the countdown timer
    var tt = setInterval(function(){
        const upload = (file) => {
            fetch('/submit', {
                method: 'POST',
                body: file
            })
            .then(function (response) {
                return response.json();
              })
              .then(function (data) {
                var prediction = data.prediction;
          
                // Update the HTML with the prediction result
                document.getElementById("predictionText").textContent = "Prediction: " + prediction;
          
                // Display the prediction popup
                var modal = document.getElementById("predictionPopup");
                var span = document.getElementsByClassName("close")[0];
                var predictionText = document.getElementById("predictionText");
          
                predictionText.textContent = prediction;
                modal.style.display = "block";
          
                // Close the prediction popup when the user clicks the close button
                span.onclick = function () {
                  modal.style.display = "none";
                };
          
                // Close the prediction popup when the user clicks outside the modal content
                window.onclick = function (event) {
                  if (event.target == modal) {
                    modal.style.display = "none";
                  }
                };
              })
              .catch(function (error) {
                console.log("Error:", error);
              });
        }
        var temp = document.getElementById('yay')
        timer--;
        if (timer === 0) {
            clearInterval(tt);
            timer = 8;
            temp.innerHTML = 8;
            test.style.visibility = 'visible';
            // Upload the file and get the predicted result
            canvas.toBlob(function(blob) {
                var data = new FormData();
                data.append('file', blob);
                upload(data);
            });
        }
        temp.innerHTML = timer;
    }, 1000);
    // Reset canvas
    setCanvasBackground();
});


sizeSlider.addEventListener("change", () => brushWidth = sizeSlider.value);

canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", drawing);
canvas.addEventListener("mouseup", () => isDrawing = false);
