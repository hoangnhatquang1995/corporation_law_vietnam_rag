from pathlib import Path
from typing import cast

import gradio as gr
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from dataset import chatroomSQL
from dataset.sql import ChatroomModel
from rag.chatbot.chatbot import answer_question, load_chatroom
import uuid

APP_TITLE = "Corporation Law Vietnam RAG"
GRADIO_PATH = "/gradio"
COLLECTION_NAME = "corporation_law_vietnam"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def create_gradio_app():
    with gr.Blocks(css=".gradio-container {max-width: 100% !important}") as demo:
        current_room_id = gr.State("")

        with gr.Row():
            # SIDEBAR
            with gr.Column(scale=1, variant="panel"):
                gr.Markdown("### 💬 Lịch sử hội thoại")
                new_chat_btn = gr.Button("➕ Cuộc trò chuyện mới", variant="primary")
                
                chat_list = gr.Radio(
                    label="Chọn cuộc hội thoại",
                    choices=[], 
                    interactive=True
                )
                refresh_btn = gr.Button("🔄 Làm mới danh sách", size="sm")

            # CHAT AREA
            with gr.Column(scale=4):
                chat_title = gr.Markdown("### 🤖 Trợ lý Luật Doanh Nghiệp")
                chatbot = gr.Chatbot(height=600)
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Nhập câu hỏi tại đây...",
                        scale=9, container=False
                    )
                    send_btn = gr.Button("Gửi", scale=1)

        # --- XỬ LÝ LOGIC ---

        def start_new_chat():
            new_id = str(uuid.uuid4())
            return new_id, [], f"### 🟢 Phiên chat mới", ""

        def update_sidebar():
            rooms = cast(list[ChatroomModel], chatroomSQL.all())
            choices = [(room.roomId, room.name or room.roomId) for room in rooms]
            return gr.update(choices=choices)

        # 1. Bấm tạo chat mới
        new_chat_btn.click(
            start_new_chat, 
            outputs=[current_room_id, chatbot, chat_title, msg_input]
        )

        # 2. Chọn một chat có sẵn từ Sidebar
        chat_list.change(
            load_chatroom, 
            inputs=[chat_list], 
            outputs=[chatbot, chat_title]
        ).then(lambda x: x, inputs=[chat_list], outputs=[current_room_id])

        # 3. Gửi tin nhắn (Cập nhật cả room_id từ return của answer_question)
        msg_submit_event = msg_input.submit(
            answer_question, 
            inputs=[msg_input, chatbot, current_room_id], 
            outputs=[chatbot, msg_input, current_room_id]
        ).then(update_sidebar, outputs=chat_list)

        send_btn.click(
            answer_question, 
            inputs=[msg_input, chatbot, current_room_id], 
            outputs=[chatbot, msg_input, current_room_id]
        ).then(update_sidebar, outputs=chat_list)

        refresh_btn.click(update_sidebar, outputs=chat_list)
        
        # Khi trang web load lần đầu
        demo.load(update_sidebar, outputs=chat_list)

    return demo

def create_app():
    app = FastAPI(title=APP_TITLE)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request):
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "request": request,
                "app_title": APP_TITLE,
                "gradio_path": GRADIO_PATH,
            },
        )

    return gr.mount_gradio_app(
        app,
        create_gradio_app(),
        path=GRADIO_PATH,
        footer_links=[],
    )


app = create_app()
