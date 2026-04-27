const frame = document.getElementById("gradio-frame");
const loading = document.getElementById("frame-loading");

const EMBEDDED_FRAME_STYLE = `
    body {
        background: #ffffff !important;
    }

    h1:first-of-type,
    h1:first-of-type + p,
    main > div > p:first-of-type {
        display: none !important;
    }

    main {
        padding-top: 14px !important;
    }
`;

function injectEmbeddedFrameStyle() {
    if (!frame) {
        return;
    }

    try {
        const doc = frame.contentDocument;
        if (!doc || !doc.head) {
            return;
        }

        let styleTag = doc.getElementById("embedded-shell-style");
        if (!styleTag) {
            styleTag = doc.createElement("style");
            styleTag.id = "embedded-shell-style";
            doc.head.appendChild(styleTag);
        }

        styleTag.textContent = EMBEDDED_FRAME_STYLE;
    } catch (error) {
        console.warn("Khong the dong bo style cho iframe Gradio:", error);
    }
}

if (frame && loading) {
    frame.addEventListener("load", () => {
        injectEmbeddedFrameStyle();
        loading.classList.add("is-hidden");
    });
}