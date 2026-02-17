import gradio as gr
import matplotlib.pyplot as plt
from core.predictor import predict_image

def run_prediction(image):

    disease, confidence, prediction = predict_image(image)

    fig = plt.figure()
    plt.bar(['glioma', 'meningioma', 'notumor', 'pituitary'], prediction)
    plt.xlabel("Tumor Type")
    plt.ylabel("Probability")
    plt.title("Prediction Probabilities")

    return disease, f"{confidence:.2f} %", fig


with gr.Blocks() as app:
    gr.Markdown("# 🧠 BSD-Pro – Brain Scan Detector")

    img_input = gr.Image(type="pil")

    output1 = gr.Textbox(label="Predicted Tumor Type")
    output2 = gr.Textbox(label="Confidence Score")
    output3 = gr.Plot(label="Probability Graph")

    predict_btn = gr.Button("Predict")

    predict_btn.click(
        run_prediction,
        inputs=img_input,
        outputs=[output1, output2, output3]
    )

app.launch()
