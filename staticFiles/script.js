const canvas = document.getElementById("canvas"),
sizeSlider = document.querySelector("#size-slider"),
clearCanvas = document.querySelector(".clear-canvas")
saveImg = document.querySelector(".save-img"),
ctx = canvas.getContext("2d")

let isDrawing = false;
selectedTool = "brush",
brushWidth = 5;

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
    const upload = (file) => {
        fetch('/submit', {
            method: 'POST',
            body: file
        })
    }

    canvas.toBlob(function(blob) {
        var data = new FormData();
        data.append('file', blob);
        upload(data);
      });
    setCanvasBackground();
});

sizeSlider.addEventListener("change", () => brushWidth = sizeSlider.value);

canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", drawing);
canvas.addEventListener("mouseup", () => isDrawing = false);