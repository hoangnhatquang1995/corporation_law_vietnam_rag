const frame = document.getElementById("gradio-frame");
const loading = document.getElementById("frame-loading");

if (frame && loading) {
    frame.addEventListener("load", () => {
        loading.classList.add("is-hidden");
    });
}