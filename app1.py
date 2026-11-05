import gradio as gr

def greet(name, excited):
    return  f"Hello {name}!" + ("!" if excited else "")

inputs = [gr.Textbox(label="Name"), gr.Checkbox(label="Excited?")]
outputs = gr.Textbox(label="Greeting")

demo = gr.Interface(
    fn=greet, 
    inputs=inputs, 
    outputs=outputs)

demo.launch()