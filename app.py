import gradio as gr
from agent_core import ShopPulseAgent

agent = ShopPulseAgent(check_boundary_coverage=True)


def handle_message(user_message, history):
    if not user_message or not user_message.strip():
        return history, {}, None
    output = agent.run_turn(user_message)
    history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": output["response"]},
    ]
    return history, output["trace"], output["trace"].get("chart_path")


with gr.Blocks(title="ShopPulse Ops Agent — WQF7002 Assignment 1", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# ShopPulse Ops Agent\n"
        "**E-Commerce Sales & Inventory Decision-Support Agent** | Model: `Qwen/Qwen2.5-7B-Instruct` | Dataset: `2026-09-01` to `2026-10-30` (720 records)"
    )
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Agent Conversation", height=430, type="messages")
            msg_input = gr.Textbox(
                label="Ask a sales, inventory, or chart question",
                placeholder="e.g., What was the net revenue for Electronics in Klang Valley during October 2026?",
            )
            with gr.Row():
                send_btn = gr.Button("Send Query", variant="primary")
                clear_btn = gr.Button("Clear")
        with gr.Column(scale=2):
            trace_box = gr.JSON(label="Tool Execution Inspector (Function Name, Arguments & Result)")
            chart_box = gr.Image(label="Generated Analytics Chart", type="filepath")

    send_btn.click(handle_message, inputs=[msg_input, chatbot], outputs=[chatbot, trace_box, chart_box]).then(
        lambda: "", outputs=msg_input
    )
    msg_input.submit(handle_message, inputs=[msg_input, chatbot], outputs=[chatbot, trace_box, chart_box]).then(
        lambda: "", outputs=msg_input
    )
    clear_btn.click(lambda: ([], {}, None), outputs=[chatbot, trace_box, chart_box])

if __name__ == "__main__":
    demo.launch()
