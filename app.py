# For plotting
import numpy as np

# For utilities
from timeit import default_timer as timer

# For conversion
import opencv_transforms.transforms as TF
import opencv_transforms.functional as FF

# For everything
import torch

# For our model
import mymodels

# For demo api
import gradio as gr

# To ignore warning
import warnings

warnings.simplefilter("ignore", UserWarning)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ncluster = 9
nc = 3 * (ncluster + 1)
netC2S = mymodels.Color2Sketch(pretrained=True).to(device)
netG = mymodels.Sketch2Color(nc=nc, pretrained=True).to(device)
transform = TF.Resize((512, 512))


def make_tensor(img):
    img = FF.to_tensor(img)
    img = FF.normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    return img


def predictC2S(img):
    final_transform = TF.Resize((img.size[0], img.size[1]))
    img = np.array(img)
    img = transform(img)
    img = make_tensor(img)
    start_time = timer()
    with torch.inference_mode():
        img_edge = netC2S(img.unsqueeze(0).to(device)).squeeze().permute(1, 2, 0).cpu().numpy()
        img_edge = FF.to_grayscale(img_edge, num_output_channels=3)
        img = FF.to_tensor(img_edge).permute(1, 2, 0).cpu().numpy()
    end_time = timer()
    img = final_transform(img)
    return img, round(end_time - start_time, 3)


def predictS2C(img, ref):
    final_transform = TF.Resize((img.size[0], img.size[1]))
    img = np.array(img)
    ref = np.array(ref)
    ref = transform(ref)
    img = transform(img)
    img = make_tensor(img)
    color_palette = mymodels.color_cluster(ref)
    for i in range(0, len(color_palette)):
        color = color_palette[i]
        color_palette[i] = make_tensor(color)
    start_time = timer()
    with torch.inference_mode():
        img_edge = netC2S(img.unsqueeze(0).to(device)).squeeze().permute(1, 2, 0).cpu().numpy()
        img_edge = FF.to_grayscale(img_edge, num_output_channels=3)
        img = FF.to_tensor(img_edge)
    input_tensor = torch.cat([img.cpu()] + color_palette, dim=0).to(device)
    with torch.inference_mode():
        fake = netG(input_tensor.unsqueeze(0).to(device)).squeeze().permute(1, 2, 0).cpu().numpy()
    end_time = timer()
    fake = final_transform(fake)
    return fake, round(end_time - start_time, 3)


example_list1 = [["./examples/img1.jpg", "./examples/ref1.jpg"],
                 ["./examples/img4.jpg", "./examples/ref4.jpg"],
                 ["./examples/img3.jpg", "./examples/ref3.jpg"],
                 ["./examples/img2.jpg", "./examples/ref2.jpg"]]
example_list2 = [["./examples/sketch1.jpg"],
                 ["./examples/sketch2.jpg"],
                 ["./examples/sketch3.jpg"],
                 ["./examples/sketch4.jpg"]]

with gr.Blocks() as demo:
    gr.Markdown("# Color2Sketch & Sketch2Color")
    with gr.Tab("Sketch To Color"):
        gr.Markdown("### Enter the **Sketch** & **Reference** on the left side. You can use example list.")
        with gr.Row():
            with gr.Column():
                input1 = [gr.inputs.Image(type="pil", label="Sketch"), gr.inputs.Image(type="pil", label="Reference")]
                with gr.Row():
                    # Clear Button
                    gr.ClearButton(input1)
                    btn1 = gr.Button("Submit")
                gr.Examples(examples=example_list1, inputs=input1)
            with gr.Column():
                output1 = [gr.inputs.Image(type="pil", label="Colored Sketch"), gr.Number(label="Prediction time (s)")]
    with gr.Tab("Color To Sketch"):
        gr.Markdown(
            "### Enter the **Colored Sketch** on the left side. You can use example list.")
        with gr.Row():
            with gr.Column():
                input2 = gr.inputs.Image(type="pil", label="Color Sketch")
                with gr.Row():
                    # Clear Button
                    gr.ClearButton(input2)
                    btn2 = gr.Button("Submit")
                gr.Examples(example_list2, inputs=input2)
            with gr.Column():
                output2 = [gr.inputs.Image(type="pil", label="Sketch"), gr.Number(label="Prediction time (s)")]
    btn1.click(predictS2C, inputs=input1, outputs=output1)
    btn2.click(predictC2S, inputs=input2, outputs=output2)
    gr.Markdown("""
    ### The model is taken from [this GitHub Repo.](https://github.com/delta6189/Anime-Sketch-Colorizer)
    
    Email : rajatsingh072002@gmail.com | My [GitHub Repo](https://github.com/Rajatsingh24/Anime-Sketch2Color-Color2Sketch)
    """)
if __name__ == "__main__":
    demo.launch(debug=False)
